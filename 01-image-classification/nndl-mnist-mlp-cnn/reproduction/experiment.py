"""Full-data MNIST training and evaluation for hand-written MLP and CNN."""
import argparse
import csv
import json
import platform
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from dataset import CHECKSUMS, augment, load, split_indices
from networks import GradientDescent, Network, scheduler

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Trial:
    network: str
    augmented: bool
    lr: float
    l2: float
    momentum: float
    schedule: str


MLP = Trial("mlp", False, 0.04, 0.0001, 0.9, "exponential")
CNN = Trial("cnn", False, 0.02, 0.00001, 0.9, "multistep")
TRIALS = {
    "mlp_plain": MLP,
    "mlp_affine": replace(MLP, augmented=True),
    "mlp_sgd": replace(MLP, augmented=True, lr=0.12, momentum=0.0),
    "mlp_step": replace(MLP, augmented=True, lr=0.02, schedule="step"),
    "cnn_plain": CNN,
    "cnn_affine": replace(CNN, augmented=True),
}


def log_probabilities(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def loss_and_derivative(logits, labels):
    rows = np.arange(len(labels))
    log_prob = log_probabilities(logits)
    loss = float(-log_prob[rows, labels].sum(dtype=np.float64))
    gradient = np.exp(log_prob)
    gradient[rows, labels] -= 1
    return loss, gradient / len(labels)


def stats(model, x, y, batch):
    confusion = np.zeros((10, 10), dtype=np.int64)
    ce = 0.0
    for start in range(0, len(y), batch):
        labels = y[start:start + batch]
        logits = model(x[start:start + batch])
        ce -= log_probabilities(logits)[np.arange(len(labels)), labels].sum(dtype=np.float64)
        np.add.at(confusion, (labels, logits.argmax(axis=1)), 1)
    return {
        "cross_entropy": float(ce / len(y)),
        "accuracy": float(confusion.trace() / len(y)),
        "correct": int(confusion.trace()),
        "samples": len(y),
        "confusion": confusion.tolist(),
    }


def score(metrics):
    return metrics["val_accuracy"], -metrics["val_ce"]


def train_trial(name, config, args, train_data, validation):
    x, y = train_data
    model = Network(config.network, args.activation, args.seed, args.mlp_width)
    optimizer = GradientDescent(model, config.lr, config.momentum, config.l2)
    schedule = scheduler(optimizer, config.schedule, args.epochs)
    shuffle = np.random.default_rng(args.seed + 1)
    augmentation = np.random.default_rng(args.seed + 2)
    best, elapsed = None, time.perf_counter()
    checkpoint = args.out / f"{name}.npz"
    fields = ["epoch", "lr", "train_ce", "train_accuracy", "val_ce", "val_accuracy"]
    with (args.out / f"{name}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            order = shuffle.permutation(len(y))
            train_ce, train_correct = 0.0, 0
            for start in range(0, len(y), args.batch):
                rows = order[start:start + args.batch]
                pixels, labels = x[rows], y[rows]
                if config.augmented:
                    pixels = augment(pixels, augmentation)
                logits = model(pixels)
                loss, derivative = loss_and_derivative(logits, labels)
                model.backward(derivative)
                optimizer.step()
                train_ce += loss
                train_correct += int(np.count_nonzero(logits.argmax(axis=1) == labels))
            metrics = stats(model, *validation, args.batch)
            record = {
                "epoch": epoch, "lr": optimizer.init_lr,
                "train_ce": train_ce / len(y), "train_accuracy": train_correct / len(y),
                "val_ce": metrics["cross_entropy"], "val_accuracy": metrics["accuracy"],
            }
            if not all(np.isfinite(value) for value in record.values()):
                raise FloatingPointError(f"Non-finite metrics in {name} epoch {epoch}")
            if best is None or score(record) > score(best):
                best = record
                model.save(checkpoint)
            writer.writerow(record)
            stream.flush()
            print(f"{name:12} epoch {epoch:02}/{args.epochs} "
                  f"train {record['train_accuracy']:.2%} val {record['val_accuracy']:.2%} "
                  f"val CE {record['val_ce']:.4f}", flush=True)
            schedule.step()
            if args.patience and epoch - best["epoch"] >= args.patience:
                break
    loaded = Network.load(checkpoint)
    replay = stats(loaded, *validation, args.batch)
    if score(best) != (replay["accuracy"], -replay["cross_entropy"]):
        raise RuntimeError(f"Checkpoint validation mismatch for {name}")
    return {
        "config": asdict(config), "activation": args.activation, "epochs_run": epoch,
        "mlp_width": args.mlp_width,
        "parameters": sum(array.size for _, array, _ in model.parameters()),
        "best": best, "seconds": round(time.perf_counter() - elapsed, 2),
    }


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def train(args):
    args.out.mkdir(parents=True, exist_ok=False)
    images, labels = load(args.data, "train")
    train_ids, val_ids = split_indices(labels, args.seed)
    np.savez_compressed(args.out / "split.npz", train=train_ids, validation=val_ids)
    training, validation = (images[train_ids], labels[train_ids]), (images[val_ids], labels[val_ids])
    chosen = dict.fromkeys(args.trials)
    archive = {}
    for name in chosen:
        archive[name] = train_trial(name, TRIALS[name], args, training, validation)
    save_json(args.out / "train.json", {
        "python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__,
        "seed": args.seed, "batch": args.batch, "epochs": args.epochs, "patience": args.patience,
        "train_samples": len(train_ids), "validation_samples": len(val_ids),
        "idx_md5": CHECKSUMS, "trials": archive,
    })


def test(args):
    images, labels = load(args.data, "t10k")
    if args.run:
        meta = json.loads((args.run / "train.json").read_text())
        results = {
            name: stats(Network.load(args.run / f"{name}.npz"), images, labels, args.batch)
            for name in meta["trials"]
        }
        save_json(args.run / "test.json", results)
    else:
        results = {args.checkpoint.stem: stats(Network.load(args.checkpoint), images, labels, args.batch)}
    for name, result in results.items():
        print(f"{name:12} {result['accuracy']:.2%} ({result['correct']}/{result['samples']})"
              f"  CE {result['cross_entropy']:.5f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "mnist")
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train")
    fit.add_argument("--out", type=Path, required=True)
    fit.add_argument("--trials", nargs="+", choices=TRIALS, default=list(TRIALS))
    fit.add_argument("--activation", choices=("relu", "sigmoid"), default="relu")
    fit.add_argument("--seed", type=int, default=309)
    fit.add_argument("--mlp-width", type=int, default=600)
    fit.add_argument("--epochs", type=int, default=15)
    fit.add_argument("--batch", type=int, default=128)
    fit.add_argument("--patience", type=int, default=0)
    assess = sub.add_parser("test")
    choice = assess.add_mutually_exclusive_group(required=True)
    choice.add_argument("--run", type=Path)
    choice.add_argument("--checkpoint", type=Path)
    assess.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("batch must be positive")
    if args.command == "train":
        if min(args.epochs, args.mlp_width) < 1 or min(args.patience, args.seed) < 0:
            parser.error("epochs and mlp-width must be positive and patience and seed nonnegative")
        train(args)
    else:
        test(args)


if __name__ == "__main__":
    main()
