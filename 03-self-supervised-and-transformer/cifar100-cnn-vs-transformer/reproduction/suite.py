"""Train both architectures under both settings before the official test."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from experiment import MODELS, ROOT, SEED, SETTINGS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "cifar100")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    script = Path(__file__).with_name("experiment.py")
    command = [sys.executable, "-u", str(script), "--data", str(args.data), "--device", args.device]
    runs = []
    for setting in SETTINGS:
        for variant in MODELS:
            output = args.out / f"{variant}-{setting}"
            runs.append(output)
            meta = output / "train.json"
            desired = {"variant": variant, "setting": setting, "batch": args.batch,
                       "epochs": args.epochs, "seed": args.seed}
            if meta.is_file():
                record = json.loads(meta.read_text())
                if any(record[field] != value for field, value in desired.items()):
                    raise ValueError(f"Existing run configuration mismatch: {output}")
                continue
            if output.exists():
                raise FileExistsError(f"Incomplete experiment {output}; preserve it before retrying")
            options = [item for key, value in desired.items() for item in ("--" + key, str(value))]
            subprocess.run([*command, "train", "--out", str(output), *options], check=True)
    for output in runs:
        subprocess.run([*command, "test", "--checkpoint", str(output / "best.pt"),
                        "--metrics", str(output / "test.json"), "--batch", str(args.batch)], check=True)


if __name__ == "__main__":
    main()
