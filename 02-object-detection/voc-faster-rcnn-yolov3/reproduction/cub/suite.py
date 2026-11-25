"""Run the four CUB-200-2011 trials in order and evaluate frozen checkpoints."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from dataset import prepare, read_samples
from experiment import ROOT, SEED, TRIALS, score, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "cub200")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.epochs < 2 or args.image_size < 32 or args.batch < 2 or args.seed < 0:
        parser.error("Use at least two epochs, image size >= 32, batch size >= 2, and a nonnegative seed")
    script = Path(__file__).with_name("experiment.py")
    base = [sys.executable, "-u", str(script), "--data", str(args.data), "--device", args.device]
    source = read_samples(prepare(args.data))
    classes = dict((sample.label, sample.path.parent.name.partition(".")[2].replace("_", " "))
                   for sample in source)
    args.out.mkdir(parents=True, exist_ok=True)
    save_json(args.out / "classes.json", [classes[idx] for idx in sorted(classes)])
    trials = {}
    for trial in TRIALS:
        run = args.out / trial
        meta = run / "train.json"
        config = {"trial": trial, "seed": args.seed, "epochs": args.epochs,
                  "batch": args.batch, "image_size": args.image_size}
        if not meta.exists():
            if run.exists():
                raise FileExistsError(f"Incomplete experiment {run}; preserve it before retrying")
            options = [item for key, value in config.items()
                       for item in ("--" + key.replace("_", "-"), str(value))]
            subprocess.run([*base, "train", "--out", str(run), *options], check=True)
        trained = json.loads(meta.read_text())
        if any(trained[key] != value for key, value in config.items()):
            raise ValueError(f"Existing run differs from requested setup: {run}")
        trials[trial] = trained
    best = max((name for name in trials if name != "scratch"), key=lambda name: score(trials[name]["best"]))
    print(f"Validation-selected transfer model: {best}", flush=True)
    for trial in TRIALS:
        run = args.out / trial
        subprocess.run([*base, "test", "--checkpoint", str(run / "best.pt"), "--metrics",
                        str(run / "test.json"), "--batch", str(args.batch)], check=True)


if __name__ == "__main__":
    main()
