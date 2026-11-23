"""Local CUB-200-2011 ResNet-18 transfer and scratch experiments."""
import argparse
import csv
import hashlib
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import ResNet18_Weights, resnet18

from dataset import image_tensors, prepare, read_samples, split_training

ROOT = Path(__file__).resolve().parents[4]
SEED = 2020
N_CLASSES = 200


@dataclass(frozen=True)
class Trial:
    initialization: str
    body_lr: float
    head_lr: float
    warmup: int


TRIALS = {
    "scratch": Trial("random", 0.001, 0.002, 0),
    "transfer": Trial("imagenet", 0.0002, 0.002, 1),
    "transfer_low": Trial("imagenet", 0.00007, 0.001, 1),
    "transfer_high": Trial("imagenet", 0.0005, 0.003, 1),
}


def select_device(name):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


def make_network(pretrained, seed):
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    torch.manual_seed(seed)
    model = resnet18(weights=weights)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 10)
        model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
    return model


def save_json(path, data):
    partial = path.with_suffix(".tmp")
    partial.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    partial.replace(path)


def sample_cache(root, samples, size, device):
    ids = np.asarray([sample.id for sample in samples], dtype="<i4")
    sha = hashlib.sha256(ids.tobytes()).hexdigest()[:20]
    folder = root / "tensors"
    folder.mkdir(exist_ok=True)
    cache = folder / f"rgb-{size}-{sha}.pt"
    if cache.exists():
        payload = torch.load(cache, map_location="cpu", weights_only=True)
        x, y = payload["pixels"], payload["labels"]
        if (
            tuple(x.shape) != (len(samples), 3, size, size) or x.dtype != torch.uint8
            or y.tolist() != [sample.label for sample in samples]
        ):
            raise ValueError(f"Cached image shape or labels do not match {cache}")
    else:
        x, y = image_tensors(samples, size)
        partial = cache.with_suffix(".tmp")
        torch.save({"pixels": x, "labels": y}, partial)
        partial.replace(cache)
    return x.to(device), y.to(device)


def batches(data, batch, rng=None):
    images, labels = data
    device = images.device
    order = torch.randperm(len(labels), generator=rng) if rng else torch.arange(len(labels))
    mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
    for rows in order.split(batch):
        rows = rows.to(device)
        x = images[rows].float() / 255
        if rng:
            count = len(rows)
            scale = 0.78 + 0.22 * torch.rand(count, generator=rng)
            mirror = 2 * (torch.rand(count, generator=rng) < 0.5).float() - 1
            translate = (2 * torch.rand(count, 2, generator=rng) - 1) * (1 - scale[:, None])
            theta = torch.zeros(count, 2, 3)
            theta[:, 0, 0] = scale * mirror
            theta[:, 1, 1] = scale
            theta[:, :, 2] = translate
            grid = F.affine_grid(theta.to(device), x.shape, align_corners=False)
            edge = 1 - 1 / x.shape[-1]
            grid.clamp_(-edge, edge)
            x = F.grid_sample(x, grid, mode="bilinear", align_corners=False)
        yield (x - mean) / std, labels[rows]


@torch.inference_mode()
def evaluate(model, data, batch):
    model.eval()
    device = data[1].device
    totals = torch.zeros(3, device=device)
    labels, predictions = [], []
    for images, target in batches(data, batch):
        logits = model(images)
        top5 = logits.topk(5, dim=1).indices
        hits = top5 == target[:, None]
        totals += torch.stack((
            F.cross_entropy(logits, target, reduction="sum"),
            hits[:, 0].sum(),
            hits.any(dim=1).sum(),
        ))
        labels.append(target)
        predictions.append(top5[:, 0])
    loss, correct, top5_correct = totals.cpu().tolist()
    count = len(data[1])
    expected = torch.cat(labels).cpu().numpy()
    observed = torch.cat(predictions).cpu().numpy()
    confusion = np.bincount(N_CLASSES * expected + observed, minlength=N_CLASSES**2).reshape(N_CLASSES, N_CLASSES)
    if int(confusion.trace()) != int(correct):
        raise RuntimeError("Top-1 counts disagree with the confusion matrix")
    return {
        "loss": loss / count, "accuracy": correct / count, "top5_accuracy": top5_correct / count,
        "correct": int(correct), "samples": count,
    }, confusion


def save_checkpoint(path, model, trial, size, epoch):
    partial = path.with_suffix(".tmp")
    torch.save({
        "model": {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()},
        "trial": trial, "image_size": size, "epoch": epoch,
    }, partial)
    partial.replace(path)


def checkpoint(path, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = make_network(False, SEED)
    model.load_state_dict(state["model"])
    return model.to(device), state


def score(row):
    return row["val_accuracy"], -row["val_loss"]


def train(args, source, device):
    config = TRIALS[args.trial]
    args.out.mkdir(parents=True, exist_ok=False)
    training, validation = split_training(source, args.seed)
    train_data = sample_cache(args.data, training, args.image_size, device)
    val_data = sample_cache(args.data, validation, args.image_size, device)
    np.savez_compressed(args.out / "split.npz",
                        train=[x.id for x in training], validation=[x.id for x in validation])
    torch.hub.set_dir(str(args.data / "torch-hub"))
    model = make_network(config.initialization == "imagenet", args.seed).to(device)
    body = [parameter for name, parameter in model.named_parameters() if not name.startswith("fc.")]
    optimizer = torch.optim.AdamW([
        {"params": body, "lr": config.body_lr},
        {"params": model.fc.parameters(), "lr": config.head_lr},
    ], weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    rng = torch.Generator().manual_seed(args.seed + 1)
    best = None
    fields = (
        "epoch", "body_lr", "head_lr", "frozen_body", "train_loss", "train_accuracy",
        "val_loss", "val_accuracy", "val_top5_accuracy",
    )
    started = time.perf_counter()
    with (
        (args.out / "history.csv").open("w", newline="") as stream,
        SummaryWriter(str(args.out / "tensorboard")) as logs,
    ):
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            frozen = epoch <= config.warmup
            model.train(not frozen)
            for param in body:
                param.requires_grad_(not frozen)
            totals = torch.zeros(2, device=device)
            for pixels, targets in batches(train_data, args.batch, rng):
                optimizer.zero_grad(set_to_none=True)
                logits = model(pixels)
                loss = F.cross_entropy(logits, targets)
                loss.backward()
                optimizer.step()
                totals += torch.stack((loss.detach() * len(targets),
                                       (logits.detach().argmax(1) == targets).sum()))
            train_loss, train_correct = (totals / len(training)).cpu().tolist()
            metrics, _ = evaluate(model, val_data, args.batch)
            record = {
                "epoch": epoch, "body_lr": optimizer.param_groups[0]["lr"],
                "head_lr": optimizer.param_groups[1]["lr"], "frozen_body": int(frozen),
                "train_loss": train_loss, "train_accuracy": train_correct,
                "val_loss": metrics["loss"], "val_accuracy": metrics["accuracy"],
                "val_top5_accuracy": metrics["top5_accuracy"],
            }
            if not all(np.isfinite(item) for item in record.values()):
                raise FloatingPointError(f"Non-finite metrics at {args.trial} epoch {epoch}")
            writer.writerow(record)
            stream.flush()
            for key, value in record.items():
                if key != "epoch":
                    logs.add_scalar(key, value, epoch)
            logs.flush()
            if best is None or score(record) > score(best):
                best = record
                save_checkpoint(args.out / "best.pt", model, args.trial, args.image_size, epoch)
            scheduler.step()
            print(f"{args.trial:15s} {epoch:02}/{args.epochs} "
                  f"train CE {train_loss:.4f} val CE {metrics['loss']:.4f} "
                  f"val acc {metrics['accuracy']:.2%} {time.perf_counter() - started:.1f}s", flush=True)
    model, saved = checkpoint(args.out / "best.pt", device)
    replay, _ = evaluate(model, val_data, args.batch)
    if saved["epoch"] != best["epoch"] or replay["accuracy"] != best["val_accuracy"]:
        raise RuntimeError("Reloaded checkpoint disagrees with validation record")
    if abs(replay["loss"] - best["val_loss"]) > 1e-4:
        raise RuntimeError("Reloaded checkpoint validation loss differs")
    save_json(args.out / "train.json", {
        "trial": args.trial, "initialization": config.initialization, "body_lr": config.body_lr,
        "head_lr": config.head_lr, "warmup": config.warmup, "optimizer": "AdamW",
        "weight_decay": 0.0001, "epochs": args.epochs, "batch": args.batch, "seed": args.seed,
        "image_size": args.image_size, "num_classes": N_CLASSES, "n_train": len(training),
        "n_val": len(validation), "n_parameters": sum(param.numel() for param in model.parameters()),
        "best": best, "device": str(device), "torch": str(torch.__version__),
        "python": platform.python_version(), "seconds": time.perf_counter() - started,
    })


def test(args, source, device):
    model, saved = checkpoint(args.checkpoint, device)
    samples = [sample for sample in source if not sample.official_train]
    data = sample_cache(args.data, samples, saved["image_size"], device)
    result, confusion = evaluate(model, data, args.batch)
    result.update({"trial": saved["trial"], "epoch": saved["epoch"], "image_size": saved["image_size"],
                   "macro_recall": float((confusion.diagonal() / confusion.sum(axis=1)).mean()),
                   "confusion": confusion.tolist()})
    if args.metrics:
        save_json(args.metrics, result)
    print(f"{saved['trial']}: {result['correct']}/{result['samples']} "
          f"top-1 {result['accuracy']:.2%}; top-5 {result['top5_accuracy']:.2%}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "cub200")
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train")
    fit.add_argument("--out", required=True, type=Path)
    fit.add_argument("--trial", required=True, choices=TRIALS)
    fit.add_argument("--image-size", type=int, default=160)
    fit.add_argument("--epochs", type=int, default=16)
    fit.add_argument("--seed", type=int, default=SEED)
    fit.add_argument("--batch", type=int, default=48)
    assess = sub.add_parser("test")
    assess.add_argument("--checkpoint", required=True, type=Path)
    assess.add_argument("--metrics", type=Path)
    assess.add_argument("--batch", type=int, default=48)
    args = parser.parse_args()
    if args.batch < 2:
        parser.error("Batch size must be at least two")
    if args.command == "train" and (args.epochs < 1 or args.image_size < 32 or args.seed < 0):
        parser.error("Image size must be at least 32, epochs positive and seed nonnegative")
    torch.set_num_threads(4)
    device = select_device(args.device)
    source = read_samples(prepare(args.data))
    if args.command == "train":
        train(args, source, device)
    else:
        test(args, source, device)


if __name__ == "__main__":
    main()
