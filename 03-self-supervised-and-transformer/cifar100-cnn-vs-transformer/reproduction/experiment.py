"""Compare parameter-matched CIFAR-100 models using shared CutMix training."""
import argparse
import csv
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from models import build_model

ROOT = Path(__file__).resolve().parents[3]
DATA_MODULE = Path(__file__).resolve().parents[2] / "cifar100-self-vs-supervised" / "reproduction"
sys.path.insert(0, str(DATA_MODULE))
try:
    from cifar100_data import test_partition, training_partitions, transform
finally:
    sys.path.pop(0)

SEED = 2020
MODELS = ("cnn", "vit")
SETTINGS = {"adamw_1e-3": (0.001, 0.03), "adamw_5e-4": (0.0005, 0.01)}
MIX_PROBABILITY = 0.5
MIX_ALPHA = 1.0


def device_for(name):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


def cutmix(images, labels, rng, probability=MIX_PROBABILITY, alpha=MIX_ALPHA):
    """Replace one clipped rectangle with another image and return its measured area weight."""
    if len(images) < 2 or rng.random() >= probability:
        return images, labels, 1.0
    _, _, height, width = images.shape
    scale = math.sqrt(1 - rng.beta(alpha, alpha))
    box_w, box_h = round(width * scale), round(height * scale)
    center_x, center_y = rng.integers(width), rng.integers(height)
    x0, y0 = max(0, center_x - box_w // 2), max(0, center_y - box_h // 2)
    x1 = min(width, center_x + (box_w + 1) // 2)
    y1 = min(height, center_y + (box_h + 1) // 2)
    area = (x1 - x0) * (y1 - y0)
    if area == 0:
        return images, labels, 1.0
    permutation = torch.as_tensor(rng.permutation(len(images)), device=images.device)
    mixed = images.clone()
    mixed[:, :, y0:y1, x0:x1] = images[permutation, :, y0:y1, x0:x1]
    return mixed, labels[permutation], 1 - area / (height * width)


def param_counts():
    return {key: sum(p.numel() for p in build_model(key).parameters() if p.requires_grad) for key in MODELS}


def lr_multiplier(epoch, epochs, warmup):
    if epoch <= warmup:
        return epoch / warmup
    progress = (epoch - warmup) / max(1, epochs - warmup)
    return 0.02 + 0.98 * (1 + math.cos(math.pi * progress)) / 2


def save_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def save_checkpoint(path, model, variant, setting, epoch):
    temp = path.with_suffix(".tmp")
    torch.save({
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "variant": variant, "setting": setting, "epoch": epoch,
    }, temp)
    temp.replace(path)


@torch.no_grad()
def evaluate(model, partition, batch):
    model.eval()
    device = partition.images.device
    totals = torch.zeros(3, device=device)
    for rgb, target in partition.batches(batch):
        logits = model(transform(rgb, 32))
        matches = logits.topk(5, dim=1).indices == target[:, None]
        totals += torch.stack((F.cross_entropy(logits, target, reduction="sum"),
                               matches[:, 0].sum(), matches.any(dim=1).sum()))
    loss, correct, top5 = totals.cpu().tolist()
    samples = len(partition.targets)
    return {"loss": loss / samples, "accuracy": correct / samples, "top5": top5 / samples,
            "correct": int(correct), "top5_correct": int(top5), "samples": samples}


def rank(record):
    return record["val_accuracy"], -record["val_loss"]


def train(args, device):
    torch.manual_seed(args.seed)
    counts = param_counts()
    relative_gap = (max(counts.values()) - min(counts.values())) / min(counts.values())
    if relative_gap > 0.03:
        raise ValueError(f"Model sizes differ by more than 3%: {counts}")
    train_data, validation, split = training_partitions(args.data, args.seed, device)
    args.out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.out / "split.npz", **split)
    torch.manual_seed(args.seed)
    model = build_model(args.variant).to(device)
    lr, decay = SETTINGS[args.setting]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)
    shuffle = torch.Generator().manual_seed(args.seed + 1)
    augmentation = torch.Generator().manual_seed(args.seed + 2)
    mixing = np.random.default_rng(args.seed + 3)
    warmup = min(2, args.epochs)
    fields = ("epoch", "lr", "train_loss", "train_weighted_accuracy", "val_loss", "val_accuracy",
              "val_top5", "cutmix_batches", "mean_replaced_area")
    best = None
    started = time.perf_counter()
    with (
        (args.out / "history.csv").open("w", newline="") as stream,
        SummaryWriter(str(args.out / "tensorboard")) as logs,
    ):
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            rate = lr * lr_multiplier(epoch, args.epochs, warmup)
            for group in optimizer.param_groups:
                group["lr"] = rate
            totals = torch.zeros(2, device=device)
            applied, covered, batches = 0, 0.0, 0
            for raw, labels in train_data.batches(args.batch, rng=shuffle):
                images = transform(raw, 32, rng=augmentation)
                images, other, weight = cutmix(images, labels, mixing)
                applied += int(weight < 1)
                covered += 1 - weight
                batches += 1
                optimizer.zero_grad(set_to_none=True)
                predictions = model(images)
                loss = F.cross_entropy(predictions, labels)
                if weight < 1:
                    loss = weight * loss + (1 - weight) * F.cross_entropy(predictions, other)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                pred = predictions.detach().argmax(1)
                hits = weight * (pred == labels).sum() + (1 - weight) * (pred == other).sum()
                totals += torch.stack((loss.detach() * len(labels), hits))
            train_loss, train_accuracy = (totals / len(train_data.targets)).cpu().tolist()
            val = evaluate(model, validation, args.batch)
            record = {
                "epoch": epoch, "lr": rate, "train_loss": train_loss, "train_weighted_accuracy": train_accuracy,
                "val_loss": val["loss"], "val_accuracy": val["accuracy"], "val_top5": val["top5"],
                "cutmix_batches": applied, "mean_replaced_area": covered / batches,
            }
            if not all(np.isfinite(value) for value in record.values()):
                raise FloatingPointError(f"Non-finite metrics in {args.variant} epoch {epoch}")
            writer.writerow(record)
            stream.flush()
            for key in fields[1:]:
                logs.add_scalar(key, record[key], epoch)
            logs.flush()
            if best is None or rank(record) > rank(best):
                best = record
                save_checkpoint(args.out / "best.pt", model, args.variant, args.setting, epoch)
            print(f"{args.variant} {args.setting} {epoch:02}/{args.epochs} "
                  f"train CE {train_loss:.4f} val CE {val['loss']:.4f} top1 {val['accuracy']:.2%} "
                  f"({time.perf_counter() - started:.1f}s)", flush=True)
    save_json(args.out / "train.json", {
        "variant": args.variant, "setting": args.setting, "epochs": args.epochs, "batch": args.batch,
        "size": 32, "seed": args.seed, "optimizer": "AdamW", "initial_lr": lr, "weight_decay": decay,
        "warmup_epochs": warmup, "schedule": "cosine", "clip_norm": 5.0, "cutmix_alpha": MIX_ALPHA,
        "cutmix_probability": MIX_PROBABILITY, "n_train": len(train_data.targets),
        "n_val": len(validation.targets), "parameters": counts[args.variant], "parameter_gap": relative_gap,
        "device": str(device), "torch": str(torch.__version__), "python": platform.python_version(),
        "best": best, "seconds": time.perf_counter() - started,
    })


def test(args, device):
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = build_model(saved["variant"]).to(device)
    model.load_state_dict(saved["model"], strict=True)
    partition = test_partition(args.data, device)
    values = evaluate(model, partition, args.batch)
    values.update({key: saved[key] for key in ("variant", "setting", "epoch")})
    if args.metrics:
        save_json(args.metrics, values)
    print(f"{saved['variant']} {saved['setting']} test {values['accuracy']:.2%} "
          f"({values['correct']}/{values['samples']}), top5 {values['top5']:.2%}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "cifar100")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train")
    fit.add_argument("--variant", choices=MODELS, required=True)
    fit.add_argument("--setting", choices=SETTINGS, required=True)
    fit.add_argument("--out", type=Path, required=True)
    fit.add_argument("--epochs", type=int, default=20)
    fit.add_argument("--batch", type=int, default=256)
    fit.add_argument("--seed", type=int, default=SEED)
    assess = sub.add_parser("test")
    assess.add_argument("--checkpoint", type=Path, required=True)
    assess.add_argument("--metrics", type=Path)
    assess.add_argument("--batch", type=int, default=256)
    args = parser.parse_args()
    if args.batch < 2:
        parser.error("Batch size must be at least two")
    if args.command == "train" and (args.epochs < 1 or args.seed < 0):
        parser.error("Use positive epochs and a nonnegative seed")
    torch.set_num_threads(4)
    device = device_for(args.device)
    if args.command == "train":
        train(args, device)
    else:
        test(args, device)


if __name__ == "__main__":
    main()
