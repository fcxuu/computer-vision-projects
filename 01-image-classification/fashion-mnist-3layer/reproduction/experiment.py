"""Train, search and evaluate a Fashion-MNIST NumPy classifier."""
import argparse
import csv
import json
import platform
import shutil
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from data_io import DIGESTS, LABELS, load, stratified_split
from model import (
    ACTIVATIONS, forward, initialize, l2_penalty, load_checkpoint,
    log_probabilities, objective_and_gradients, save_checkpoint,
)

DEFAULT_DATA = Path(__file__).resolve().parents[3] / "data" / "fashion-mnist"


@dataclass(frozen=True)
class Configuration:
    hidden: int = 256
    lr: float = 0.12
    l2: float = 0.0001
    activation: str = "relu"


def evaluate(model, activation, x, y):
    loss_sum = 0.0
    confusion = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    for start in range(0, len(y), 1024):
        targets = y[start:start + 1024]
        logits, _ = forward(model, x[start:start + 1024], activation)
        loss_sum -= float(log_probabilities(logits)[np.arange(len(targets)), targets].sum())
        np.add.at(confusion, (targets, logits.argmax(axis=1)), 1)
    return {"cross_entropy": loss_sum / len(y), "accuracy": float(confusion.trace() / len(y))}, confusion


def train(name, config, train_set, val_set, arguments):
    x, y = train_set
    rng = np.random.default_rng(arguments.seed)
    model = initialize(x.shape[1], config.hidden, len(LABELS), config.activation, rng)
    history = []
    path = arguments.out / f"{name}.npz"
    started = time.perf_counter()
    best = None
    for epoch in range(1, arguments.epochs + 1):
        order = rng.permutation(len(y))
        lr = config.lr * arguments.decay ** (epoch - 1)
        for start in range(0, len(order), arguments.batch):
            rows = order[start:start + arguments.batch]
            _, gradients = objective_and_gradients(model, x[rows], y[rows], config.activation, config.l2)
            for key, gradient in gradients.items():
                model[key] -= lr * gradient
        train_metrics, _ = evaluate(model, config.activation, *train_set)
        val_metrics, _ = evaluate(model, config.activation, *val_set)
        record = {
            "epoch": epoch, "learning_rate": lr, "train_ce": train_metrics["cross_entropy"],
            "val_ce": val_metrics["cross_entropy"], "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "train_objective": train_metrics["cross_entropy"] + l2_penalty(model, config.l2),
        }
        if not all(np.isfinite(value) for value in record.values()):
            raise FloatingPointError(f"Non-finite metrics in {name} epoch {epoch}")
        history.append(record)
        if best is None or score(record) > score(best):
            save_checkpoint(path, model, config.activation)
            best = record
        print(f"{name:12} {epoch:02}/{arguments.epochs:02} "
              f"train CE {record['train_ce']:.4f} val CE {record['val_ce']:.4f} "
              f"val acc {record['val_accuracy']:.2%}", flush=True)
    write_csv(arguments.out / f"{name}.csv", history)
    return {
        "name": name, "configuration": asdict(config), "best": best,
        "seconds": round(time.perf_counter() - started, 2),
    }


def score(record):
    return record["val_accuracy"], -record["val_ce"]


def write_csv(path, records):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def save_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def run(arguments):
    arguments.out.mkdir(parents=True, exist_ok=False)
    x, y = load(arguments.data, "train")
    train_idx, val_idx = stratified_split(y, arguments.seed)
    train_set, val_set = (x[train_idx], y[train_idx]), (x[val_idx], y[val_idx])
    base = Configuration(arguments.hidden, arguments.lr, arguments.l2, arguments.activation)
    configurations = {"baseline": base}
    if arguments.command == "search":
        configurations.update({
            "lower_lr": replace(base, lr=base.lr / 2),
            "higher_lr": replace(base, lr=base.lr * 2),
            "narrow": replace(base, hidden=max(1, base.hidden // 2)),
            "wider": replace(base, hidden=base.hidden * 2),
            "higher_l2": replace(base, l2=base.l2 * 10),
            "sigmoid": replace(base, activation="sigmoid"),
        })
    trials = [train(name, cfg, train_set, val_set, arguments) for name, cfg in configurations.items()]
    winner = max(trials, key=lambda trial: score(trial["best"]))
    selected = arguments.out / "best.npz"
    shutil.copyfile(arguments.out / f"{winner['name']}.npz", selected)
    model, activation = load_checkpoint(selected)
    validation, _ = evaluate(model, activation, *val_set)
    if (validation["accuracy"], -validation["cross_entropy"]) != score(winner["best"]):
        raise RuntimeError("Reloaded checkpoint does not reproduce best validation metrics")
    save_json(arguments.out / "search.json", {
        "python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__,
        "seed": arguments.seed, "epochs": arguments.epochs, "batch": arguments.batch,
        "lr_decay": arguments.decay, "train_size": len(train_idx), "validation_size": len(val_idx),
        "official_md5": DIGESTS, "trials": trials, "selected": winner["name"],
    })
    print(f"Selected {winner['name']} on validation at epoch {winner['best']['epoch']}: "
          f"{validation['accuracy']:.2%}", flush=True)


def test(arguments):
    model, activation = load_checkpoint(arguments.checkpoint)
    x, y = load(arguments.data, "t10k")
    metrics, confusion = evaluate(model, activation, x, y)
    result = {**metrics, "samples": len(y), "classes": LABELS, "confusion": confusion.tolist()}
    if arguments.metrics:
        save_json(arguments.metrics, result)
    print(json.dumps({**metrics, "samples": len(y)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "search"):
        fit = sub.add_parser(command)
        fit.add_argument("--out", type=Path, required=True)
        fit.add_argument("--epochs", type=int, default=36)
        fit.add_argument("--batch", type=int, default=256)
        fit.add_argument("--seed", type=int, default=23)
        fit.add_argument("--hidden", type=int, default=256)
        fit.add_argument("--lr", type=float, default=0.12)
        fit.add_argument("--l2", type=float, default=0.0001)
        fit.add_argument("--decay", type=float, default=0.97)
        fit.add_argument("--activation", choices=ACTIVATIONS, default="relu")
    assess = sub.add_parser("test")
    assess.add_argument("--checkpoint", type=Path, required=True)
    assess.add_argument("--metrics", type=Path)
    args = parser.parse_args()
    if args.command == "test":
        test(args)
    else:
        if min(args.hidden, args.batch, args.epochs) <= 0 or args.seed < 0:
            parser.error("hidden, batch and epochs must be positive and seed nonnegative")
        if not (0 < args.lr < float("inf") and 0 <= args.l2 < float("inf") and 0 < args.decay <= 1):
            parser.error("lr must be positive, l2 nonnegative and decay in (0, 1]")
        run(args)


if __name__ == "__main__":
    main()
