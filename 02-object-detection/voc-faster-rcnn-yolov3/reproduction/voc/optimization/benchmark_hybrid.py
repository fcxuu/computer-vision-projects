"""Compare isolated CPU and hybrid MPS/CPU Faster R-CNN runs on matched VOC batches."""
import argparse
import json
import math
import platform
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
WORKER = Path(__file__).with_name("benchmark_worker.py")
DEVICE_ERRORS = re.compile(
    r"command buffer.*error|ImpactingInteractivity|MPS backend out of memory|"
    r"SIGABRT|SIGSEGV|libc\+\+abi: terminating", re.IGNORECASE,
)


def valid_metrics(metrics, args, device):
    if metrics["device"] != device or len(metrics["batch_ids"]) != args.warm_up + args.steps:
        return False
    if not all(len(batch) == args.batch for batch in metrics["batch_ids"]):
        return False
    groups = {
        "warmup_seconds": args.warm_up, "timed_seconds": args.steps,
        "losses": args.warm_up + args.steps, "grad_norms": args.warm_up + args.steps,
    }
    for key, count in groups.items():
        if len(metrics[key]) != count or not all(math.isfinite(value) and value >= 0 for value in metrics[key]):
            return False
    return (
        all(value > 0 for value in metrics["timed_seconds"])
        and metrics["parameter_changed"] is True
        and math.isfinite(metrics["validation_loss"])
        and metrics["validation_loss"] >= 0
        and bool(metrics["inference_counts"]) and sum(metrics["inference_counts"]) > 0
        and bool(metrics["proposal_counts"]) and sum(metrics["proposal_counts"]) > 0
    )


def run_worker(args, round_number, device, seed):
    stem = f"round-{round_number}-{device}"
    raw_path = args.out / (stem + ".json")
    log_path = args.out / (stem + ".log")
    options = {
        "device": device, "data": args.data, "checkpoint": args.checkpoint,
        "image-size": args.image_size, "batch": args.batch,
        "warm-up": args.warm_up, "steps": args.steps, "seed": seed, "out": raw_path,
    }
    command = [sys.executable, "-u", str(WORKER), *(
        token for key, value in options.items() for token in ("--" + key, str(value))
    )]
    with log_path.open("w", encoding="utf-8") as log:
        try:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=150)
        except subprocess.TimeoutExpired:
            return {"stable": False, "failure": "Worker timed out", "log": log_path.name}
    errors = DEVICE_ERRORS.search(log_path.read_text(errors="replace"))
    if result.returncode or errors or not raw_path.is_file():
        return {
            "stable": False, "failure": errors.group() if errors else "Worker did not complete successfully",
            "return_code": result.returncode, "log": log_path.name,
        }
    try:
        metrics = json.loads(raw_path.read_text())
        complete = valid_metrics(metrics, args, device)
    except (KeyError, TypeError, ValueError):
        complete = False
    if not complete:
        return {"stable": False, "failure": "Invalid or incomplete stability metrics", "log": log_path.name}
    return {
        "stable": True, "log": log_path.name, "metrics": metrics,
        "median_seconds": statistics.median(metrics["timed_seconds"]),
        "total_timed_seconds": sum(metrics["timed_seconds"]),
    }


def paired(cpu, mps):
    return (
        cpu["stable"] and mps["stable"]
        and cpu["metrics"]["batch_ids"] == mps["metrics"]["batch_ids"]
        and cpu["metrics"]["validation_ids"] == mps["metrics"]["validation_ids"]
    )


def benchmark(args):
    args.out.mkdir(parents=True, exist_ok=False)
    rounds = []
    for number, order in enumerate((("cpu", "mps"), ("mps", "cpu")), start=1):
        outcomes = {}
        for device in order:
            stopped = rounds and not rounds[0]["outcomes"][device]["stable"]
            if stopped:
                outcomes[device] = {"stable": False, "failure": "Skipped after an unsuccessful first round"}
                continue
            print(f"Round {number}/2: {device} short test", flush=True)
            outcomes[device] = run_worker(args, number, device, args.seed + number - 1)
            print(json.dumps({key: value for key, value in outcomes[device].items() if key != "metrics"}), flush=True)
        cpu, mps = outcomes["cpu"], outcomes["mps"]
        match = paired(cpu, mps)
        rounds.append({
            "order": list(order), "seed": args.seed + number - 1,
            "outcomes": outcomes, "matched": match,
            "cpu_to_hybrid_speedup": cpu["median_seconds"] / mps["median_seconds"] if match else None,
        })
    compared = all(row["matched"] for row in rounds)
    faster = compared and all(
        row["cpu_to_hybrid_speedup"] > 1.0 and
        row["outcomes"]["mps"]["total_timed_seconds"] < row["outcomes"]["cpu"]["total_timed_seconds"]
        for row in rounds
    )
    result = {
        "configuration": {
            "image_size": args.image_size, "batch": args.batch,
            "warm_up": args.warm_up, "timed_steps_per_round": args.steps, "platform": platform.platform(),
        },
        "rounds": rounds, "comparison_valid": compared,
        "hybrid_passed": faster,
        "decision": "Hybrid path eligible for full training" if faster else "Defer full CPU training",
    }
    (args.out / "gate.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(result["decision"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "voc2007")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs" / "voc2007-full" / "faster_rcnn" / "best.pt")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--warm-up", type=int, default=2)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2020)
    args = parser.parse_args()
    if args.image_size < 128 or args.image_size % 32 or min(args.batch, args.warm_up, args.steps) < 1 or args.seed < 0:
        parser.error("Use an image size >= 128 divisible by 32, positive batch and step counts, and a nonnegative seed")
    benchmark(args)


if __name__ == "__main__":
    main()
