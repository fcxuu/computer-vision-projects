"""Train and evaluate two VOC2007 detectors on the same official partitions."""
import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from detectors import ROOT, Detector, original_detections
from voc import CLASSES, VocDataset, evaluate, load_split

VARIANTS = ("faster_rcnn", "yolov3")
DEFAULTS = {"faster_rcnn": (4, 0.003), "yolov3": (8, 0.009)}


def select_device(name):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


def save_json(path, data):
    partial = path.with_suffix(".tmp")
    partial.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    partial.replace(path)


def stack(batch):
    pixels, targets = zip(*batch)
    return torch.stack(pixels), list(targets)


def loader(samples, size, batch, shuffle=None):
    return DataLoader(VocDataset(samples, size), batch_size=batch, collate_fn=stack,
                      shuffle=shuffle is not None, generator=shuffle, num_workers=0)


def transform_training(pixels, records, rng):
    size = pixels.shape[-1]
    for index, target in enumerate(records):
        if torch.rand((), generator=rng) < 0.5:
            pixels[index] = pixels[index].flip(-1)
            target["boxes"][:, 0::2] = size - target["boxes"][:, [2, 0]]
    return pixels


@torch.no_grad()
def assess(detector, samples, images, include_loss):
    predictions = {}
    total = torch.zeros((), device=detector.device)
    started = time.perf_counter()
    for pixels, records in images:
        data = pixels.to(detector.device, dtype=torch.float32) / 255
        targets = detector.targets(records) if include_loss else None
        outputs, loss = detector.outputs(data, targets)
        predictions.update(original_detections(outputs, records))
        if loss is not None:
            total += loss * len(records)
    if len(predictions) != len(samples):
        raise RuntimeError("Missing VOC prediction rows")
    result = evaluate(samples, predictions)
    result["images"] = len(samples)
    result["seconds"] = time.perf_counter() - started
    if include_loss:
        result["loss"] = float(total) / len(samples)
    return result, predictions


def save_checkpoint(path, detector, epoch):
    partial = path.with_suffix(".tmp")
    torch.save({
        "name": detector.name, "size": detector.size, "epoch": epoch,
        "model": {key: tensor.detach().cpu() for key, tensor in detector.net.state_dict().items()},
    }, partial)
    partial.replace(path)


def load_checkpoint(path, device, vendor, data):
    state = torch.load(path, map_location="cpu", weights_only=True)
    detector = Detector(state["name"], state["size"], device, vendor, data)
    detector.net.load_state_dict(state["model"], strict=True)
    detector.net.eval()
    return detector, state


def train(args, device):
    batch, lr = DEFAULTS[args.variant]
    batch = args.batch or batch
    lr = args.lr or lr
    training, validation = load_split(args.data, "train"), load_split(args.data, "val")
    if {row.id for row in training} & {row.id for row in validation}:
        raise RuntimeError("VOC train and validation overlap")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    shuffle = torch.Generator().manual_seed(args.seed + 10)
    augment = torch.Generator().manual_seed(args.seed + 20)
    training_loader = loader(training, args.size, batch, shuffle)
    validation_loader = loader(validation, args.size, args.val_batch)
    detector = Detector(args.variant, args.size, device, args.vendor, args.data, pretrained=True)
    optimizer = torch.optim.SGD([parameter for parameter in detector.net.parameters() if parameter.requires_grad],
                                lr=lr, momentum=0.9, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=lr * 0.05)
    best = None
    fields = ["epoch", "lr", "train_loss", "val_loss", "val_map50", "fit_seconds", "val_seconds"]
    started = time.perf_counter()
    with (
        (args.out / "history.csv").open("w", newline="") as stream,
        SummaryWriter(str(args.out / "tensorboard")) as logs,
    ):
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            detector.training_mode()
            running = torch.zeros((), device=device)
            fit_start = time.perf_counter()
            for step, (pixels, records) in enumerate(training_loader, start=1):
                pixels = transform_training(pixels, records, augment)
                imgs = pixels.to(device, dtype=torch.float32) / 255
                labels = detector.targets(records)
                optimizer.zero_grad(set_to_none=True)
                loss = detector.training_loss(imgs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(detector.net.parameters(), 10.0)
                optimizer.step()
                running += loss.detach() * len(records)
                if step % 100 == 0:
                    current = float(loss.detach())
                    if not np.isfinite(current):
                        raise FloatingPointError(f"Invalid {args.variant} loss at epoch {epoch}, step {step}")
                    print(f"{args.variant} epoch {epoch} {step}/{len(training_loader)} batch loss {current:.4f}", flush=True)
            training_loss = float(running) / len(training)
            fit_duration = time.perf_counter() - fit_start
            val, _ = assess(detector, validation, validation_loader, include_loss=True)
            record = {
                "epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
                "train_loss": training_loss, "val_loss": val["loss"], "val_map50": val["map50"],
                "fit_seconds": fit_duration, "val_seconds": val["seconds"],
            }
            if not all(np.isfinite(value) for value in record.values()):
                raise FloatingPointError(f"Invalid {args.variant} epoch metrics at epoch {epoch}")
            writer.writerow(record)
            stream.flush()
            for key in ("train_loss", "val_loss", "val_map50", "lr"):
                logs.add_scalar(key, record[key], epoch)
            logs.flush()
            if best is None or (record["val_map50"], -record["val_loss"]) > (best["val_map50"], -best["val_loss"]):
                best = record
                save_checkpoint(args.out / "best.pt", detector, epoch)
                save_json(args.out / "best-val.json", val)
            scheduler.step()
            print(f"{args.variant} {epoch:02}/{args.epochs} train {training_loss:.4f} "
                  f"val {val['loss']:.4f} mAP50 {val['map50']:.2%} "
                  f"(fit {fit_duration:.1f}s, val {val['seconds']:.1f}s)", flush=True)
    save_json(args.out / "train.json", {
        "variant": args.variant, "size": args.size, "seed": args.seed, "batch": batch, "val_batch": args.val_batch,
        "epochs": args.epochs, "initial_lr": lr, "optimizer": "SGD", "momentum": 0.9, "weight_decay": 0.0001,
        "n_train": len(training), "n_validation": len(validation), "best": best,
        "total_parameters": sum(p.numel() for p in detector.net.parameters()),
        "trainable_parameters": sum(p.numel() for p in detector.net.parameters() if p.requires_grad),
        "device": str(device), "torch": str(torch.__version__), "platform": platform.platform(),
        "elapsed_seconds": time.perf_counter() - started,
    })


def save_predictions(path, predictions):
    ids = sorted(predictions)
    rows = [predictions[image_id] for image_id in ids]
    offsets = np.r_[0, np.cumsum([len(row["boxes"]) for row in rows])]
    np.savez_compressed(path, ids=ids, offsets=offsets,
                        boxes=np.concatenate([row["boxes"] for row in rows]),
                        labels=np.concatenate([row["labels"] for row in rows]).astype(np.int8),
                        scores=np.concatenate([row["scores"] for row in rows]))


def test(args, device):
    detector, state = load_checkpoint(args.checkpoint, device, args.vendor, args.data)
    samples = load_split(args.data, "test")
    result, predictions = assess(detector, samples, loader(samples, state["size"], args.batch), include_loss=False)
    result.update({"variant": state["name"], "epoch": state["epoch"], "size": state["size"]})
    if args.metrics:
        save_json(args.metrics, result)
    if args.predictions:
        save_predictions(args.predictions, predictions)
    print(f"{state['name']} VOC2007 test mAP50 {result['map50']:.2%} "
          f"over {result['images']} images ({result['seconds']:.1f}s)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "voc2007")
    parser.add_argument("--vendor", type=Path, default=ROOT / "data" / "vendor" / "yolov3")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    fit.add_argument("--variant", choices=VARIANTS, required=True)
    fit.add_argument("--out", type=Path, required=True)
    fit.add_argument("--epochs", type=int, default=8)
    fit.add_argument("--size", type=int, default=320)
    fit.add_argument("--batch", type=int)
    fit.add_argument("--val-batch", type=int, default=8)
    fit.add_argument("--seed", type=int, default=2020)
    fit.add_argument("--lr", type=float)
    assess = commands.add_parser("test")
    assess.add_argument("--checkpoint", type=Path, required=True)
    assess.add_argument("--metrics", type=Path)
    assess.add_argument("--predictions", type=Path)
    assess.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = select_device(args.device)
    if args.batch is not None and args.batch < 1:
        parser.error("Batch size must be positive")
    if args.command == "train":
        if args.size < 128 or args.size % 32 or min(args.epochs, args.val_batch) < 1:
            parser.error("Use size divisible by 32 and at least 128, positive epochs and val batch")
        if args.seed < 0 or (args.lr is not None and not (0 < args.lr < float("inf"))):
            parser.error("Use a nonnegative seed and a finite positive learning rate")
        train(args, device)
    else:
        test(args, device)


if __name__ == "__main__":
    main()
