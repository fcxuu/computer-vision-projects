"""Time Faster R-CNN optimizer steps on a fixed set of VOC batches."""

import argparse
import math
import platform
import sys
import time
from pathlib import Path

import torch
import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from detectors import Detector, HostROIAlign
from experiment import loader, save_json, transform_training
from voc import CLASSES, load_split


def subset(samples, count, seed):
    if count > len(samples):
        raise ValueError(f"Requested {count} samples from a split with {len(samples)} images")
    order = torch.randperm(len(samples), generator=torch.Generator().manual_seed(seed), device="cpu")
    return [samples[index] for index in order[:count].tolist()]


def batch_ids(batches):
    return [[record["id"] for record in records] for _, records in batches]


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()


def finite_scalar(value, name):
    number = float(value.detach())
    if not math.isfinite(number):
        raise FloatingPointError(f"Nonfinite {name}: {number}")
    return number


def validate_boxes(boxes, size, context):
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"Unexpected {context} box shape {boxes.shape}")
    coordinates = boxes.detach().cpu()
    if not torch.isfinite(coordinates).all():
        raise FloatingPointError(f"Nonfinite {context} box coordinates")
    if (coordinates[:, 2:] <= coordinates[:, :2]).any():
        raise ValueError(f"Degenerate {context} box")
    if (coordinates < -1e-3).any() or (coordinates > size + 1e-3).any():
        raise ValueError(f"{context} box outside {size}×{size} input")
    return len(coordinates)


@torch.no_grad()
def validate(detector, decoded, size):
    pixels, records = decoded
    images = pixels.to(detector.device, dtype=torch.float32) / 255
    predictions, loss = detector.outputs(images, detector.targets(records))
    if len(predictions) != len(records):
        raise RuntimeError("Missing validation predictions")
    inference_counts, proposal_counts = [], []
    for image, prediction, record in zip(images, predictions, records):
        image_id = record["id"]
        count = validate_boxes(prediction["boxes"], size, f"{image_id} detection")
        scores = prediction["scores"].detach().cpu()
        labels = prediction["labels"].detach().cpu()
        if scores.shape != (count,) or labels.shape != (count,):
            raise ValueError(f"Invalid score or label shape for {image_id}")
        if not torch.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
            raise ValueError(f"Invalid detection confidence for {image_id}")
        if (labels < 0).any() or (labels >= len(CLASSES)).any():
            raise ValueError(f"Invalid detection label for {image_id}")
        proposals = detector.proposal_boxes(image)
        proposal_count = validate_boxes(proposals, size, f"{image_id} proposal")
        if not proposal_count:
            raise RuntimeError(f"Empty proposals for {image_id}")
        inference_counts.append(count)
        proposal_counts.append(proposal_count)
    return inference_counts, proposal_counts, finite_scalar(loss, "validation loss")


def benchmark(args):
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    torch.manual_seed(args.seed)
    augmentation = torch.Generator().manual_seed(args.seed + 20)

    training = subset(load_split(args.data, "train"), args.batch * (args.warm_up + args.steps), args.seed + 10)
    validation = subset(load_split(args.data, "val"), args.batch, args.seed + 30)
    batches = list(loader(training, args.image_size, args.batch))
    validation_batch = next(iter(loader(validation, args.image_size, args.batch)))
    ids = batch_ids(batches)
    validation_ids = batch_ids([validation_batch])[0]

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["name"] != "faster_rcnn" or checkpoint["size"] != 256:
        raise ValueError("Expected a 256px Faster R-CNN checkpoint")
    detector = Detector("faster_rcnn", args.image_size, device, None, args.data, pretrained=False)
    detector.net.load_state_dict(checkpoint["model"], strict=True)
    if device.type == "mps" and not isinstance(detector.net.roi_heads.box_roi_pool, HostROIAlign):
        raise RuntimeError("MPS benchmark requires HostROIAlign")
    detector.training_mode()
    trainable = {name: param for name, param in detector.net.named_parameters() if param.requires_grad}
    optimizer = torch.optim.SGD(list(trainable.values()), lr=0.0015, momentum=0.9, weight_decay=0.0001)

    durations, losses, grad_norms = [], [], []
    for step, (pixels, records) in enumerate(batches, start=1):
        synchronize(device)
        started = time.perf_counter()
        pixels = transform_training(pixels, records, augmentation)
        images = pixels.to(device, dtype=torch.float32) / 255
        targets = detector.targets(records)
        optimizer.zero_grad(set_to_none=True)
        loss = detector.training_loss(images, targets)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(trainable.values(), 10.0)
        optimizer.step()
        synchronize(device)
        durations.append(time.perf_counter() - started)
        losses.append(finite_scalar(loss, f"step {step} loss"))
        grad_norms.append(finite_scalar(norm, f"step {step} gradient norm"))

    parameter_changed = any(
        not torch.equal(param.detach().cpu(), checkpoint["model"][name]) for name, param in trainable.items()
    )
    if not parameter_changed:
        raise RuntimeError("No trainable parameter was updated")
    counts, proposals, validation_loss = validate(detector, validation_batch, args.image_size)
    return {
        "device": args.device,
        "batch_ids": ids,
        "validation_ids": validation_ids,
        "warmup_seconds": durations[:args.warm_up],
        "timed_seconds": durations[args.warm_up:],
        "losses": losses,
        "grad_norms": grad_norms,
        "inference_counts": counts,
        "proposal_counts": proposals,
        "validation_loss": validation_loss,
        "parameter_changed": parameter_changed,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "python_version": platform.python_version(),
        "image_size": args.image_size,
        "batch_size": args.batch,
        "seed": args.seed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--warm-up", type=int, default=2)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(args.batch, args.image_size, args.steps) < 1 or args.warm_up < 0:
        parser.error("image size, batch and step count must be positive and warm-up nonnegative")
    result = benchmark(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.out, result)


if __name__ == "__main__":
    main()
