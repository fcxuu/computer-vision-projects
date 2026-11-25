"""Train VGG-A variants with separate validation selection and official testing."""
import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from data import ROOT, device_for, test_partition, training_partitions
from model import VARIANTS, VGG, checkpoint

EPOCH_FIELDS = ["epoch", "lr", "train_ce", "train_accuracy", "val_ce", "val_accuracy"]
BATCH_FIELDS = ["step", "epoch", "loss", "objective", "first_conv_grad", "output_grad"]


@torch.inference_mode()
def assess(model, partition, batch):
    model.eval()
    total_loss = torch.zeros((), device=partition.images.device)
    total_correct = torch.zeros((), dtype=torch.int32, device=partition.images.device)
    for x, y in partition.batches(batch):
        logits = model(x)
        total_loss += nn.functional.cross_entropy(logits, y, reduction="sum")
        total_correct += (logits.argmax(1) == y).sum()
    return float(total_loss) / len(partition), int(total_correct), len(partition)


def write_json(path, value):
    partial = path.with_suffix(".tmp")
    partial.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    partial.replace(path)


def save_best(path, model, args, epoch):
    partial = path.with_suffix(".tmp")
    torch.save({
        "variant": args.variant, "width": args.width, "activation": args.activation, "epoch": epoch,
        "weights": {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()},
    }, partial)
    partial.replace(path)


def make_optimizer(model, args):
    if args.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def train(args):
    device = device_for(args.device)
    args.out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    training, validation, indices = training_partitions(args.data, args.seed, device)
    model = VGG(args.variant, args.width, args.activation).to(device)
    first, last = model.features[0].weight, model.classifier[-1].weight
    optimizer = make_optimizer(model, args)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr * 0.02)
    rng = torch.Generator().manual_seed(args.seed + 1)
    np.savez_compressed(args.out / "split.npz", **indices)
    best = None
    step = 0
    started = time.perf_counter()
    with (
        (args.out / "epochs.csv").open("w", newline="") as epoch_file,
        (args.out / "batches.csv").open("w", newline="") as batch_file,
        SummaryWriter(log_dir=args.out / "tensorboard") as writer,
    ):
        epochs = csv.DictWriter(epoch_file, EPOCH_FIELDS, lineterminator="\n")
        batches = csv.DictWriter(batch_file, BATCH_FIELDS, lineterminator="\n")
        epochs.writeheader()
        batches.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            lr = optimizer.param_groups[0]["lr"]
            batch_records = []
            skipped = 0
            for x, y in training.batches(args.batch, rng=rng):
                if len(y) == 1 and args.variant == "vgg_a_bn":
                    skipped += 1
                    continue
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                per_example = nn.functional.cross_entropy(logits, y, reduction="none")
                ce = per_example.mean()
                if args.loss == "focal":
                    loss = ((1 - torch.exp(-per_example)).square() * per_example).mean()
                else:
                    loss = ce
                loss.backward()
                batch_records.append(torch.stack((
                    ce.detach(), loss.detach(), first.grad.detach().norm(), last.grad.detach().norm(),
                    (logits.detach().argmax(1) == y).sum(), ce.new_tensor(len(y)),
                )))
                optimizer.step()
            recorded = torch.stack(batch_records).cpu().numpy()
            if not np.isfinite(recorded).all():
                raise FloatingPointError(f"Non-finite training metrics in epoch {epoch}")
            count = int(recorded[:, 5].sum())
            assert count == len(training) - skipped
            for ce, objective, front, output, _, _ in recorded:
                step += 1
                batches.writerow({
                    "step": step, "epoch": epoch, "loss": ce, "objective": objective,
                    "first_conv_grad": front, "output_grad": output,
                })
            batch_file.flush()
            val_ce, val_correct, val_samples = assess(model, validation, args.batch)
            record = {
                "epoch": epoch, "lr": lr,
                "train_ce": float(recorded[:, 0] @ recorded[:, 5]) / count,
                "train_accuracy": float(recorded[:, 4].sum()) / count,
                "val_ce": val_ce, "val_accuracy": val_correct / val_samples,
            }
            if not np.isfinite(val_ce):
                raise FloatingPointError(f"Non-finite validation metrics in epoch {epoch}")
            epochs.writerow(record)
            epoch_file.flush()
            for key in EPOCH_FIELDS[1:]:
                writer.add_scalar(key, record[key], epoch)
            writer.add_scalar("grad/first_conv_median", float(np.median(recorded[:, 2])), epoch)
            writer.add_scalar("grad/output_median", float(np.median(recorded[:, 3])), epoch)
            writer.flush()
            if best is None or score(record) > score(best):
                best = record
                save_best(args.out / "best.pt", model, args, epoch)
            if args.schedule == "cosine":
                schedule.step()
            print(f"{args.variant} lr={args.lr:g} {epoch:02}/{args.epochs} "
                  f"train CE {record['train_ce']:.4f} val CE {val_ce:.4f} "
                  f"val {record['val_accuracy']:.2%} {time.perf_counter() - started:.1f}s", flush=True)
    write_json(args.out / "run.json", {
        "variant": args.variant, "width": args.width, "lr": args.lr, "batch": args.batch,
        "epochs": args.epochs, "schedule": args.schedule, "optimizer": args.optimizer,
        "weight_decay": args.weight_decay, "activation": args.activation, "loss": args.loss,
        "seed": args.seed, "device": str(device),
        "torch": torch.__version__, "python": platform.python_version(), "platform": platform.platform(),
        "train_samples": len(training), "validation_samples": len(validation),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best": best, "steps": step, "seconds": time.perf_counter() - started,
    })


def score(record):
    return record["val_accuracy"], -record["val_ce"]


def test(args):
    device = device_for(args.device)
    model, state = checkpoint(args.checkpoint, device)
    test_data = test_partition(args.data, device)
    loss, correct, count = assess(model, test_data, args.batch)
    payload = {
        "variant": state["variant"], "width": state["width"], "epoch": state["epoch"],
        "test_ce": loss, "test_accuracy": correct / count, "test_correct": correct, "test_samples": count,
    }
    if args.metrics:
        write_json(args.metrics, payload)
    print(json.dumps(payload))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "cifar10")
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cpu", "cuda"))
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train")
    fit.add_argument("--out", type=Path, required=True)
    fit.add_argument("--variant", choices=VARIANTS, required=True)
    fit.add_argument("--optimizer", choices=("adamw", "sgd"), default="adamw")
    fit.add_argument("--activation", choices=("relu", "gelu"), default="relu")
    fit.add_argument("--loss", choices=("cross_entropy", "focal"), default="cross_entropy")
    fit.add_argument("--schedule", choices=("cosine", "constant"), default="cosine")
    fit.add_argument("--seed", type=int, default=2020)
    fit.add_argument("--epochs", type=int, default=20)
    fit.add_argument("--batch", type=int, default=128)
    fit.add_argument("--lr", type=float, default=0.001)
    fit.add_argument("--width", type=float, default=0.5)
    fit.add_argument("--weight-decay", type=float, default=0.0001)
    assess = sub.add_parser("test")
    assess.add_argument("--checkpoint", type=Path, required=True)
    assess.add_argument("--metrics", type=Path)
    assess.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()
    if args.batch < 2:
        parser.error("batch size must be at least two")
    if args.command == "train":
        if min(args.lr, args.width) <= 0 or args.epochs < 1 or min(args.seed, args.weight_decay) < 0:
            parser.error("width, lr and epochs must be positive; seed and weight decay must be nonnegative")
        train(args)
    else:
        test(args)


if __name__ == "__main__":
    main()
