"""Render VOC2007 RPN examples and attributed two-detector comparisons."""

import argparse
import json
from pathlib import Path
from urllib.parse import quote

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
import torch

from detectors import ROOT, original_detections
from voc import CLASSES, letterbox, load_split, unletterbox

THRESHOLD = 0.20
PROPOSAL_LIMIT = 24
EXAMPLES = Path(__file__).with_name("examples")
COMMONS = "https://commons.wikimedia.org/wiki/File:"
SOURCES = {
    "bicycle": ("Bike-side.jpg", "MartinThoma", "Bike-side.jpg", "CC0 1.0",
                "https://creativecommons.org/publicdomain/zero/1.0/"),
    "cat": ("Cat-animal.jpg", "Narmadhashree", "Cat-animal.jpg", "CC BY-SA 4.0",
            "https://creativecommons.org/licenses/by-sa/4.0/"),
    "cow": ("A COW GRAZING.jpg", "Valtino44", "A_COW_GRAZING.jpg", "CC BY-SA 4.0",
            "https://creativecommons.org/licenses/by-sa/4.0/"),
}
MODIFICATIONS = "Source resized and re-encoded as JPEG, repeated in two panels, and annotated with model predictions."
SELECTION = (
    "Greedily maximize newly covered non-difficult ground-truth classes, then total unique "
    "non-difficult classes, then non-difficult object count; break ties by ascending VOC image ID."
)


def select_rpn_samples(samples, count=4):
    """Choose examples from annotations alone, favoring class coverage."""
    candidates = []
    for sample in samples:
        labels = sample.labels[~sample.difficult]
        if len(labels):
            candidates.append((sample, set(labels.tolist()), len(labels)))
    if len(candidates) < count:
        raise ValueError("Too few test samples with non-difficult annotations")
    chosen = []
    covered = set()
    for _ in range(count):
        best = min(range(len(candidates)), key=lambda i: (
            -len(candidates[i][1] - covered), -len(candidates[i][1]),
            -candidates[i][2], candidates[i][0].id,
        ))
        sample, labels, _ = candidates.pop(best)
        chosen.append(sample)
        covered.update(labels)
    return chosen


def detector_image(image, model, device):
    pixels, geometry = letterbox(image, model.size)
    return pixels.to(device=device, dtype=torch.float32).div_(255.0), geometry


@torch.inference_mode()
def detections(model, pixels, geometry, image_id):
    outputs, _ = model.outputs(pixels.unsqueeze(0))
    prediction = original_detections(outputs, [{"id": image_id, "geometry": geometry}])[image_id]
    records = []
    for box, label, score in zip(prediction["boxes"], prediction["labels"], prediction["scores"]):
        if score >= THRESHOLD:
            records.append({"box": box.tolist(), "label": CLASSES[int(label)], "score": float(score)})
    return sorted(records, key=lambda record: -record["score"])


@torch.inference_mode()
def proposals(model, pixels, geometry):
    square_boxes = model.proposal_boxes(pixels, limit=PROPOSAL_LIMIT).detach().cpu().numpy()
    original = unletterbox(square_boxes, geometry)
    records = []
    for rank, box in enumerate(original, start=1):
        # A proposal can lie entirely on letterbox padding and have no drawable area.
        if (box[2:] > box[:2]).all():
            records.append({"rank": rank, "box": box.tolist(), "score": None})
    return records


def panel(axis, image, records, title, kind):
    axis.imshow(image)
    axis.set_axis_off()
    axis.set_title(title, fontsize=10, loc="left")
    axis.set_xlim(0, image.width)
    axis.set_ylim(image.height, 0)
    for record in records:
        left, top, right, bottom = record["box"]
        if kind == "proposals":
            color, alpha = "#ff9a00", 0.68
        else:
            color, alpha = ("#61e29a", 0.85) if record.get("difficult") else ("#47d9fb", 1)
        axis.add_patch(Rectangle((left, top), right - left, bottom - top, linewidth=1.3,
                                 edgecolor=color, facecolor="none", alpha=alpha))
        if "label" in record:
            name = record["label"]
            if "score" in record:
                name = f"{name} {record['score']:.2f}"
            elif record.get("difficult"):
                name = f"{name} (difficult)"
            axis.text(left, top, name, color="white", fontweight="bold", va="top", fontsize=7,
                      clip_on=True, bbox={"facecolor": "#14273a", "edgecolor": "none", "pad": 1.5, "alpha": 0.92})
    if not records:
        axis.text(0.04, 0.06, "No boxes", transform=axis.transAxes, color="white", fontsize=8,
                  bbox={"facecolor": "#14273a", "edgecolor": "none", "pad": 2})


def rgb_image(path):
    with Image.open(path) as file:
        return file.convert("RGB")


def rpn_figure(model, samples, device, path):
    figure, axes = plt.subplots(4, 3, figsize=(12, 12.5), layout="constrained")
    figure.suptitle("VOC2007 test examples / Faster R-CNN proposals and final detections", fontsize=16)
    evidence = []
    for sample, row in zip(samples, axes):
        image = rgb_image(sample.path)
        pixels, geometry = detector_image(image, model, device)
        ground_truth = [{"box": box.tolist(), "label": CLASSES[int(label)], "difficult": bool(flag)}
                        for box, label, flag in zip(sample.boxes, sample.labels, sample.difficult)]
        rpn = proposals(model, pixels, geometry)
        final = detections(model, pixels, geometry, sample.id)
        panel(row[0], image, ground_truth, f"{sample.id} / Ground truth", "truth")
        panel(row[1], image, rpn, f"Top {PROPOSAL_LIMIT} RPN proposals ({len(rpn)} shown)", "proposals")
        panel(row[2], image, final, f"Final detections / score ≥ {THRESHOLD:.2f}", "predictions")
        evidence.append({"id": sample.id, "threshold": THRESHOLD, "ground_truth": ground_truth,
                         "proposals": rpn, "faster_rcnn": final})
    figure.savefig(path, dpi=150, facecolor="white")
    plt.close(figure)
    return evidence


def comparison_figure(name, image, faster, yolo, device, path):
    records = {}
    figure, axes = plt.subplots(1, 2, figsize=(12, 6.5))
    figure.subplots_adjust(left=0.03, right=0.97, top=0.87, bottom=0.22, wspace=0.05)
    figure.suptitle(f"{name.capitalize()} / confidence cutoff {THRESHOLD:.2f}", fontsize=16)
    for axis, (key, model, label) in zip(axes, (
        ("faster_rcnn", faster, "Faster R-CNN"), ("yolov3", yolo, "YOLOv3"),
    )):
        pixels, geometry = detector_image(image, model, device)
        records[key] = detections(model, pixels, geometry, name)
        panel(axis, image, records[key], label, "predictions")
    title, author, source_file, license_name, license_url = SOURCES[name]
    source_url = COMMONS + quote(source_file.replace(" ", "_"))
    credit = {"title": title, "author": author, "source_url": source_url,
              "original_license": license_name, "adaptation_license": license_name,
              "license_url": license_url, "modifications": MODIFICATIONS}
    figure.text(
        0.04, 0.05,
        f'“{title}” by {author} | Original and annotated image: {license_name}\n'
        f"{source_url}\n{license_url}\n{MODIFICATIONS}",
        fontsize=9, color="#14273a", linespacing=1.5,
    )
    figure.savefig(path, dpi=150, facecolor="white")
    plt.close(figure)
    return {"name": f"{name}.jpg", "threshold": THRESHOLD, "credit": credit, **records}


def resolve_device(value):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faster", type=Path, required=True, help="Validation-best Faster R-CNN checkpoint")
    parser.add_argument("--yolo", type=Path, required=True, help="Validation-best YOLOv3 checkpoint")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "voc2007")
    parser.add_argument("--vendor", type=Path, default=ROOT / "data" / "vendor" / "yolov3", help="Pinned YOLOv3 source checkout")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from experiment import load_checkpoint

    torch.set_num_threads(4)
    device = resolve_device(args.device)
    samples = select_rpn_samples(load_split(args.data, "test"))
    faster, _ = load_checkpoint(args.faster, device, args.vendor, args.data)
    yolo, _ = load_checkpoint(args.yolo, device, args.vendor, args.data)
    if faster.name != "faster_rcnn" or yolo.name != "yolov3":
        raise ValueError("The checkpoints must match --faster and --yolo respectively")
    args.out.mkdir(parents=True, exist_ok=True)
    rpn = rpn_figure(faster, samples, device, args.out / "rpn.png")
    comparisons = [
        comparison_figure(name, rgb_image(EXAMPLES / f"{name}.jpg"), faster, yolo, device,
                          args.out / f"{name}-annotated.jpg")
        for name in SOURCES
    ]
    payload = {
        "threshold": THRESHOLD,
        "coordinate_system": "Original-image pixels, xyxy",
        "rpn_limit": PROPOSAL_LIMIT,
        "rpn_score_note": "RPN proposal scores are not exposed by the detector interface. Rank comes from proposal order.",
        "voc_test_selection": SELECTION,
        "voc_test": rpn,
        "external_examples": comparisons,
    }
    (args.out / "visualizations.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
