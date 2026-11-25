"""Run matched SimCLR, frozen-probe, and supervised CIFAR-100 comparisons."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from experiment import PRETRAIN, ROOT, SEED


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "cifar100")
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--size", type=int, choices=(32, 64), default=64)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if min(args.epochs, args.steps) < 1 or args.batch < 2 or args.seed < 0:
        parser.error("Use positive counts, batch size at least two, and a nonnegative seed")
    script = Path(__file__).with_name("experiment.py")
    base = [sys.executable, "-u", str(script), "--data", str(args.data), "--device", args.device]
    shared = ["--batch", str(args.batch), "--seed", str(args.seed), "--size", str(args.size)]

    def fit(folder, summary, configuration, command):
        path = args.out / folder
        result = path / summary
        if result.is_file():
            record = json.loads(result.read_text())
            if any(record[k] != v for k, v in configuration.items()):
                raise ValueError(f"Existing run configuration differs: {path}")
        elif path.exists():
            raise FileExistsError(f"Incomplete experiment {path}; preserve it before retrying")
        else:
            subprocess.run([*base, *command, *shared, "--out", str(path)], check=True)

    common = {"batch": args.batch, "seed": args.seed, "size": args.size}
    for name in PRETRAIN:
        fit(f"pretrain-{name}", "run.json", {**common, "steps": args.steps, "variant": name},
            ["pretrain", "--variant", name, "--steps", str(args.steps)])
    sources = ("scratch", "imagenet", *(f"simclr_{key}" for key in PRETRAIN))
    for source in sources:
        command = ["train", "--source", source, "--epochs", str(args.epochs)]
        if source.startswith("simclr_"):
            encoder = args.out / f"pretrain-{source.removeprefix('simclr_')}" / "pretrained.pt"
            command += ["--pretrained", str(encoder)]
        fit(source, "train.json", {**common, "epochs": args.epochs, "source": source}, command)
    for source in sources:
        run = args.out / source
        subprocess.run([*base, "test", "--checkpoint", str(run / "best.pt"), "--metrics", str(run / "test.json"),
                        "--batch", str(args.batch)], check=True)


if __name__ == "__main__":
    main()
