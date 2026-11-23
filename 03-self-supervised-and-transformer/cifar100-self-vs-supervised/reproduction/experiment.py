"""Run SimCLR pretraining and supervised CIFAR-100 classification experiments."""
import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from cifar100_data import test_partition, training_partitions, transform
from models import Classifier, SimCLR, encoder, fingerprint, ntxent

ROOT = Path(__file__).resolve().parents[3]
SEED = 2020
PRETRAIN = {"5k": (5000, 0.2), "20k": (20000, 0.2), "45k": (45000, 0.2), "20k_t05": (20000, 0.5)}


def device_for(name):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_weights(path, model, **configuration):
    temporary = path.with_suffix(".tmp")
    torch.save({**configuration, "weights": {key: value.detach().cpu() for key, value in model.state_dict().items()}},
               temporary)
    temporary.replace(path)


def model_context(args, device):
    return {
        "seed": args.seed, "size": args.size, "batch": args.batch,
        "device": str(device), "torch": str(torch.__version__), "python": platform.python_version(),
        "platform": platform.platform(),
    }


def pretrain(args, train, split, device):
    size, temperature = PRETRAIN[args.variant]
    if args.batch > size:
        raise ValueError("Batch is larger than the SimCLR subset")
    subset = np.random.default_rng(args.seed + 1).permutation(len(train.targets))[:size]
    np.save(args.out / "pretrain-indices.npy", split["train"][subset], allow_pickle=False)
    model = SimCLR().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=args.lr * 0.02)
    shuffle = torch.Generator().manual_seed(args.seed + 2)
    view_rng = torch.Generator().manual_seed(args.seed + 3)
    fields = ("step", "learning_rate", "ntxent", "pair_top1")
    progress = 0
    started = time.perf_counter()
    accumulated = torch.zeros(2, device=device)
    sample_steps = 0
    with (
        (args.out / "history.csv").open("w", newline="") as stream,
        SummaryWriter(str(args.out / "tensorboard")) as logs,
    ):
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        model.train()
        while progress < args.steps:
            for raw, _ in train.batches(args.batch, rng=shuffle, indices=subset):
                if len(raw) != args.batch:
                    continue
                first, second = (transform(raw, args.size, rng=view_rng, strong=True) for _ in range(2))
                optimizer.zero_grad(set_to_none=True)
                first, second = model(torch.cat((first, second))).chunk(2, dim=0)
                loss, retrieval = ntxent(first, second, temperature)
                loss.backward()
                optimizer.step()
                accumulated += torch.stack((loss.detach(), retrieval.detach()))
                sample_steps += 1
                progress += 1
                if progress % 50 == 0 or progress == args.steps:
                    mean_loss, mean_retrieval = (accumulated / sample_steps).cpu().tolist()
                    values = {"step": progress, "learning_rate": optimizer.param_groups[0]["lr"],
                              "ntxent": mean_loss, "pair_top1": mean_retrieval}
                    if not all(np.isfinite(value) for value in values.values()):
                        raise FloatingPointError(f"Non-finite contrastive metrics at step {progress}")
                    writer.writerow(values)
                    stream.flush()
                    logs.add_scalar("pretrain/ntxent", mean_loss, progress)
                    logs.add_scalar("pretrain/pair_top1", mean_retrieval, progress)
                    logs.flush()
                    print(f"SimCLR {args.variant:8s} {progress:4d}/{args.steps} "
                          f"loss {mean_loss:.4f} retrieval {mean_retrieval:.2%} "
                          f"{time.perf_counter() - started:.1f}s", flush=True)
                    accumulated.zero_()
                    sample_steps = 0
                schedule.step()
                if progress == args.steps:
                    break
    save_weights(args.out / "pretrained.pt", model, variant=args.variant, size=args.size,
                 seed=args.seed, steps=args.steps, temperature=temperature, samples=size)
    write_json(args.out / "run.json", {
        **model_context(args, device), "variant": args.variant, "pretrain_samples": size,
        "temperature": temperature, "steps": progress, "lr": args.lr, "optimizer": "AdamW",
        "weight_decay": 0.0001, "seconds": time.perf_counter() - started,
        "backbone_digest": fingerprint(model.backbone),
    })


def prepared_backbone(args):
    if args.source == "imagenet":
        return encoder(imagenet=True)
    if args.source == "scratch":
        return encoder()
    if args.pretrained is None:
        raise ValueError("Pass --pretrained for a SimCLR source")
    state = torch.load(args.pretrained, map_location="cpu", weights_only=True)
    if state["variant"] != args.source.removeprefix("simclr_") or state["size"] != args.size:
        raise ValueError("SimCLR checkpoint and probe source or image size disagree")
    backbone = encoder()
    prefix = "backbone."
    backbone.load_state_dict({key.removeprefix(prefix): value for key, value in state["weights"].items()
                              if key.startswith(prefix)}, strict=True)
    return backbone


@torch.no_grad()
def freeze_features(backbone, partition, size, batch):
    backbone.requires_grad_(False).eval()
    vectors = []
    for pixels, _ in partition.batches(batch):
        vectors.append(backbone(transform(pixels, size)))
    return torch.cat(vectors)


def feature_batches(vectors, targets, batch, rng=None):
    number = len(targets)
    indices = torch.arange(number) if rng is None else torch.randperm(number, generator=rng)
    for rows in indices.split(batch):
        rows = rows.to(targets.device)
        yield vectors[rows], targets[rows]


def image_batches(partition, size, batch, shuffle=None, aug=None):
    for pixels, labels in partition.batches(batch, rng=shuffle):
        yield transform(pixels, size, rng=aug), labels


@torch.no_grad()
def evaluate(model, batches, count, device):
    model.eval()
    sums = torch.zeros(3, device=device)
    for images, labels in batches:
        logits = model(images)
        top = logits.topk(5, dim=1).indices == labels[:, None]
        sums += torch.stack((F.cross_entropy(logits, labels, reduction="sum"),
                             top[:, 0].sum(), top.any(dim=1).sum()))
    ce, correct, top5 = sums.cpu().tolist()
    return {"loss": ce / count, "accuracy": correct / count, "top5": top5 / count,
            "correct": int(correct), "top5_correct": int(top5), "samples": count}


def select(record):
    return record["val_accuracy"], -record["val_loss"]


def classifier_fit(args, train, validation, device):
    torch.hub.set_dir(str(args.data / "torch-hub"))
    backbone = prepared_backbone(args)
    torch.manual_seed(args.seed + 7)
    classifier = Classifier(backbone).to(device)
    frozen = args.source != "scratch"
    before = fingerprint(backbone) if frozen else None
    rng = torch.Generator().manual_seed(args.seed + 5)
    aug = torch.Generator().manual_seed(args.seed + 6)
    if frozen:
        train_vectors = freeze_features(backbone, train, args.size, args.batch)
        validation_vectors = freeze_features(backbone, validation, args.size, args.batch)

        def training_batches():
            return feature_batches(train_vectors, train.targets, args.batch, rng)

        def validating_batches():
            return feature_batches(validation_vectors, validation.targets, args.batch)

        fitted = classifier.head
    else:
        def training_batches():
            return image_batches(train, args.size, args.batch, shuffle=rng, aug=aug)

        def validating_batches():
            return image_batches(validation, args.size, args.batch)

        fitted = classifier
    optimizer = torch.optim.AdamW(fitted.parameters(), lr=args.lr, weight_decay=0.0001)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr * 0.02)
    fields = ("epoch", "learning_rate", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "val_top5")
    best = None
    started = time.perf_counter()
    with (
        (args.out / "history.csv").open("w", newline="") as stream,
        SummaryWriter(str(args.out / "tensorboard")) as logs,
    ):
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            fitted.train()
            totals = torch.zeros(2, device=device)
            for x, y in training_batches():
                optimizer.zero_grad(set_to_none=True)
                logits = fitted(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()
                totals += torch.stack((loss.detach() * len(y), (logits.detach().argmax(1) == y).sum()))
            train_loss, train_accuracy = (totals / len(train.targets)).cpu().tolist()
            metrics = evaluate(fitted, validating_batches(), len(validation.targets), device)
            record = {
                "epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss, "train_accuracy": train_accuracy,
                "val_loss": metrics["loss"], "val_accuracy": metrics["accuracy"], "val_top5": metrics["top5"],
            }
            if not all(np.isfinite(value) for value in record.values()):
                raise FloatingPointError(f"Non-finite classifier metrics at epoch {epoch}")
            writer.writerow(record)
            stream.flush()
            for key in fields[1:]:
                logs.add_scalar(key, record[key], epoch)
            logs.flush()
            if best is None or select(record) > select(best):
                best = record
                save_weights(args.out / "best.pt", classifier, source=args.source, size=args.size, epoch=epoch)
            schedule.step()
            print(f"{args.source:13s} {epoch:02d}/{args.epochs} "
                  f"val CE {record['val_loss']:.4f} top1 {record['val_accuracy']:.2%} "
                  f"{time.perf_counter() - started:.1f}s", flush=True)
    after = fingerprint(backbone) if frozen else None
    if before != after:
        raise RuntimeError("A frozen encoder parameter or buffer changed during the linear probe")
    write_json(args.out / "train.json", {
        **model_context(args, device), "source": args.source, "protocol": "linear" if frozen else "end_to_end",
        "lr": args.lr, "optimizer": "AdamW", "weight_decay": 0.0001, "epochs": args.epochs,
        "n_train": len(train.targets), "n_validation": len(validation.targets),
        "trainable_parameters": sum(p.numel() for p in fitted.parameters() if p.requires_grad),
        "frozen_fingerprint_before": before, "frozen_fingerprint_after": after,
        "best": best, "seconds": time.perf_counter() - started,
    })


def test(args, device):
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    network = Classifier(encoder()).to(device)
    network.load_state_dict(state["weights"])
    data = test_partition(args.data, device)
    tested = evaluate(network, image_batches(data, state["size"], args.batch), len(data.targets), device)
    tested.update({"source": state["source"], "epoch": state["epoch"], "size": state["size"]})
    if args.metrics:
        write_json(args.metrics, tested)
    print(f"{tested['source']:13s} test {tested['accuracy']:.2%}, top-5 {tested['top5']:.2%} "
          f"({tested['correct']}/{tested['samples']})", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "cifar100")
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    commands = parser.add_subparsers(dest="command", required=True)
    ssl = commands.add_parser("pretrain")
    ssl.add_argument("--variant", choices=PRETRAIN, required=True)
    ssl.add_argument("--steps", type=int, default=1200)
    ssl.add_argument("--lr", type=float, default=0.001)
    fit = commands.add_parser("train")
    fit.add_argument("--source", choices=("imagenet", "scratch", *(f"simclr_{key}" for key in PRETRAIN)), required=True)
    fit.add_argument("--pretrained", type=Path)
    fit.add_argument("--epochs", type=int, default=18)
    fit.add_argument("--lr", type=float)
    for command in (ssl, fit):
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--size", type=int, choices=(32, 64), default=64)
        command.add_argument("--seed", type=int, default=SEED)
        command.add_argument("--batch", type=int, default=128)
    assess = commands.add_parser("test")
    assess.add_argument("--checkpoint", type=Path, required=True)
    assess.add_argument("--metrics", type=Path)
    assess.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = device_for(args.device)
    if args.batch < 2:
        parser.error("Use batch size at least two")
    if args.command == "test":
        test(args, device)
        return
    if args.command == "train" and args.lr is None:
        args.lr = 0.001 if args.source == "scratch" else 0.004
    if not (0 < args.lr < float("inf")) or args.seed < 0 or getattr(args, "steps", 1) < 1 or getattr(args, "epochs", 1) < 1:
        parser.error("Use positive finite LR and steps/epochs, and a nonnegative seed")
    torch.manual_seed(args.seed)
    train, validation, split = training_partitions(args.data, args.seed, device)
    args.out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.out / "split.npz", **split)
    if args.command == "pretrain":
        pretrain(args, train, split, device)
    else:
        classifier_fit(args, train, validation, device)


if __name__ == "__main__":
    main()
