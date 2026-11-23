"""Run the full CIFAR-10 protocol as serial, independently recorded trials."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from data import ROOT
from model import VARIANTS

SCRIPT = Path(__file__).with_name("experiment.py")
SWEEP = ("0.0001", "0.0005", "0.001", "0.002")


def run_trials(args):
    base = [sys.executable, "-u", str(SCRIPT), "--data", str(args.data), "--device", args.device]
    common = {"batch": args.batch, "seed": args.seed, "weight_decay": 0.0001, "optimizer": "adamw",
              "activation": "relu", "loss": "cross_entropy", "lr": 0.001, "schedule": "cosine"}

    def train(name, **overrides):
        out = args.out / name
        config = {**common, **overrides}
        metadata = out / "run.json"
        if metadata.is_file():
            saved = json.loads(metadata.read_text())
            if any(saved[key] != value for key, value in config.items()) or not (out / "best.pt").is_file():
                raise ValueError(f"Existing trial configuration differs: {out}")
            return
        if out.exists():
            raise FileExistsError(f"Incomplete trial directory: {out}")
        options = [value for key, setting in config.items()
                   for value in ("--" + key.replace("_", "-"), str(setting))]
        subprocess.run([*base, "train", "--out", str(out), *options], check=True)

    train("main-vgg_a", variant="vgg_a", width=0.5, epochs=args.epochs)
    for variant in VARIANTS:
        train(f"full-{variant}", variant=variant, width=1.0, epochs=args.epochs)
    for variant in VARIANTS[:2]:
        for lr in SWEEP:
            train(f"sweep-{variant}-{lr}", variant=variant, width=0.5, epochs=args.sweep_epochs,
                  lr=float(lr), schedule="constant")
    probes = {
        "gelu": {"activation": "gelu"},
        "focal": {"loss": "focal"},
        "decay": {"weight_decay": 0.001},
        "sgd": {"optimizer": "sgd", "lr": 0.03},
    }
    for name, change in probes.items():
        train(f"probe-{name}", variant="vgg_a_bn", width=0.5, epochs=args.sweep_epochs,
              schedule="constant", **change)
    for name in ("main-vgg_a", *(f"full-{variant}" for variant in VARIANTS)):
        run = args.out / name
        subprocess.run([*base, "test", "--checkpoint", str(run / "best.pt"),
                        "--metrics", str(run / "test.json"), "--batch", str(args.batch)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "cifar10")
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cpu", "cuda"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--sweep-epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()
    run_trials(args)


if __name__ == "__main__":
    main()
