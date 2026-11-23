"""Render the VOC2007 detection reproduction from recorded evidence."""

import argparse
import csv
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
from PIL import Image as Raster
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Table, TableStyle

MODELS = {"faster_rcnn": "Faster R-CNN", "yolov3": "YOLOv3"}
PALETTE = {"faster_rcnn": "#146e72", "yolov3": "#9d4939"}
CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)
REPO = "https://github.com/fcxuu/computer-vision-projects"
CODE = f"{REPO}/tree/portfolio/reproductions-2026-11-23/02-object-detection/voc-faster-rcnn-yolov3/reproduction/voc"
RELEASE = f"{REPO}/releases/download/portfolio-voc2007-2026-11-23"
INK = colors.HexColor("#183049")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def close(left, right):
    return math.isclose(left, right, abs_tol=1e-9, rel_tol=0)


def verified_map(metrics, expected_images, path):
    by_class = metrics["per_class"]
    if set(by_class) != set(CLASSES) or metrics["images"] != expected_images:
        raise ValueError(f"Invalid classes or image count in {path}")
    aps = {name: by_class[name]["ap"] for name in CLASSES}
    if any(value is None or not 0 <= value <= 1 for value in aps.values()):
        raise ValueError(f"Missing or invalid category AP in {path}")
    average = math.fsum(aps.values()) / len(aps)
    if not close(average, metrics["map50"]):
        raise ValueError(f"Category APs do not reproduce the recorded mAP in {path}")
    return average, aps


def load_runs(run):
    records = {}
    for name in MODELS:
        folder = run / name
        config = read_json(folder / "train.json")
        validation = read_json(folder / "best-val.json")
        test = read_json(folder / "test.json")
        with (folder / "history.csv").open(newline="") as stream:
            history = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
        if config["variant"] != name or (config["n_train"], config["n_validation"]) != (2501, 2510):
            raise ValueError(f"Unexpected model or official split counts in {folder}")
        if [row["epoch"] for row in history] != list(range(1, config["epochs"] + 1)) or not history:
            raise ValueError(f"Incomplete or unordered epoch history in {folder}")
        if not all(math.isfinite(value) for row in history for value in row.values()):
            raise ValueError(f"Non-finite training history in {folder}")
        selected = max(history, key=lambda row: (row["val_map50"], -row["val_loss"]))
        if any(not close(value, config["best"][key]) for key, value in selected.items()):
            raise ValueError(f"Selected history row disagrees with training metadata in {folder}")
        if (test["variant"], test["epoch"], test["size"]) != (name, selected["epoch"], config["size"]):
            raise ValueError(f"The official test record does not match the frozen checkpoint in {folder}")
        val_map, _ = verified_map(validation, config["n_validation"], folder / "best-val.json")
        test_map, aps = verified_map(test, 4952, folder / "test.json")
        if not close(val_map, selected["val_map50"]) or not close(validation["loss"], selected["val_loss"]):
            raise ValueError(f"Validation metrics do not match the selected epoch in {folder}")
        records[name] = {
            "config": config, "history": history, "selected": selected, "test": test,
            "val_map": val_map, "test_map": test_map, "aps": aps,
        }
    return records


def save_plot(figure, path):
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def plot_curves(records, out):
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.65), layout="constrained")
    for axis, (key, title) in zip(axes[:2], MODELS.items()):
        history, best = records[key]["history"], records[key]["selected"]
        epochs = [row["epoch"] for row in history]
        axis.plot(epochs, [row["train_loss"] for row in history], label="Train", color=PALETTE[key])
        axis.plot(epochs, [row["val_loss"] for row in history], label="Validation", color="#416eae", linestyle="--")
        axis.axvline(best["epoch"], color="#85929c", linestyle=":", linewidth=1)
        axis.set(title=f"{title} objective", ylabel="Model-specific loss")
        axis.legend(frameon=False, fontsize=8)
        axes[2].plot(epochs, [row["val_map50"] for row in history], label=title, color=PALETTE[key])
        axes[2].scatter(best["epoch"], best["val_map50"], color=PALETTE[key], zorder=4)
    axes[2].set(title="Validation mAP50")
    axes[2].yaxis.set_major_formatter(PercentFormatter(1))
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=0.16)
    save_plot(figure, out / "curves.png")


def plot_ap(records, out):
    positions = np.arange(len(CLASSES))
    figure, axis = plt.subplots(figsize=(10.7, 5.1), layout="constrained")
    for offset, (key, title) in zip((-0.18, 0.18), MODELS.items()):
        axis.barh(positions + offset, [records[key]["aps"][name] for name in CLASSES],
                  height=0.34, label=title, color=PALETTE[key])
    axis.set(yticks=positions, yticklabels=CLASSES, xlabel="VOC2007 11-point AP at IoU 0.50", xlim=(0, 1))
    axis.xaxis.set_major_formatter(PercentFormatter(1))
    axis.invert_yaxis()
    axis.grid(axis="x", alpha=0.2)
    axis.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    save_plot(figure, out / "per-class.png")


def preview(path, width, max_height=None):
    with Raster.open(path) as image:
        w, h = image.size
    height = width * h / w
    if max_height is not None and height > max_height:
        width, height = width * max_height / height, max_height
    return Image(str(path), width=width, height=height)


def footer(canvas, doc):
    canvas.setFillColor(INK)
    canvas.setFont("Helvetica", 9)
    canvas.drawString(43, 27, "fcxuu / computer-vision-projects")
    canvas.drawRightString(A4[0] - 43, 27, str(doc.page))


def table(rows, widths):
    widget = Table(rows, colWidths=widths, hAlign="LEFT")
    widget.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#edf5f6"), colors.white]),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
    ]))
    return widget


def link(url, label):
    return f'<link href={quoteattr(url)} color="#146e72"><u>{escape(label)}</u></link>'


def build_pdf(records, visual, out):
    doc = SimpleDocTemplate(str(out / "report.pdf"), pagesize=A4, leftMargin=43, rightMargin=43,
                            topMargin=40, bottomMargin=52, title="Object detection on PASCAL VOC2007",
                            author="fcxuu / computer-vision-projects")
    title = ParagraphStyle("Title", fontName="Helvetica-Bold", fontSize=24, leading=29,
                           textColor=INK, spaceAfter=14)
    heading = ParagraphStyle("Heading", fontName="Helvetica-Bold", fontSize=14, leading=18,
                             textColor=INK, spaceBefore=10, spaceAfter=5)
    text = ParagraphStyle("Body", fontName="Helvetica", fontSize=10, leading=14,
                          textColor=INK, spaceAfter=7)
    compact = ParagraphStyle("Caption", parent=text, fontSize=9, leading=12, spaceAfter=6)
    training = [["Setting", *MODELS.values()]]
    for label, key, fmt in (
        ("Epochs", "epochs", "d"), ("Square input side", "size", "d"), ("Train / validation batch", None, ""),
        ("Optimizer", "optimizer", "s"), ("Momentum", "momentum", "g"),
        ("Weight decay", "weight_decay", "g"), ("Initial LR", "initial_lr", "g"),
        ("Total parameters", "total_parameters", ",d"),
        ("Trainable parameters", "trainable_parameters", ",d"), ("Device", "device", "s"), ("Seed", "seed", "d"),
    ):
        config = [row["config"] for row in records.values()]
        values = [f"{args[key]:{fmt}}" if key else f"{args['batch']} / {args['val_batch']}" for args in config]
        training.append([label, *values])
    results = [["Detector", "Best epoch", "Val mAP50", "Test mAP50"]]
    for name, row in records.items():
        results.append([MODELS[name], f"{row['selected']['epoch']:.0f}",
                        f"{row['val_map']:.2%}", f"{row['test_map']:.2%}"])
    faster, yolo = records.values()
    delta = 100 * (yolo["test_map"] - faster["test_map"])
    differences = {label: yolo["aps"][label] - faster["aps"][label] for label in CLASSES}
    largest = max(differences, key=lambda label: abs(differences[label]))
    threshold = float(visual["threshold"])
    test_examples = visual["voc_test"]
    examples = {example["name"]: example for example in visual["external_examples"]}
    if len(test_examples) != 4 or len({row["id"] for row in test_examples}) != 4 or set(examples) != {"bicycle.jpg", "cat.jpg", "cow.jpg"}:
        raise ValueError("Expected four separate VOC test images and exactly three external examples")
    if not all(close(row["threshold"], threshold) for row in [*test_examples, *examples.values()]):
        raise ValueError("Visualization thresholds are not shared")
    story = [
        Paragraph("Object detection on PASCAL VOC2007", title),
        Paragraph(f"The frozen YOLOv3 checkpoint scored <b>{delta:+.2f} percentage points</b> in official test mAP50 "
                  "relative to the frozen Faster R-CNN checkpoint.", text),
        Paragraph("Data and checkpoint selection", heading),
        Paragraph(f"The official partitions contain {faster['config']['n_train']:,} training images, "
                  f"{faster['config']['n_validation']:,} validation images and {faster['test']['images']:,} test images. "
                  "For each of 20 categories, score-ranked predictions match ground truth at IoU 0.50. Difficult "
                  "objects do not count toward recall, and detections whose best match is difficult are ignored. "
                  "AP uses the VOC2007 11-point interpolated precision rule. mAP is the mean across classes. "
                  "Validation mAP selects the checkpoint, with lower validation loss breaking ties.", text),
        table(results, [146, 104, 105, 105]),
        Paragraph("Training protocol", heading),
        Paragraph("Faster R-CNN uses a MobileNetV3-Large FPN. YOLOv3 uses the pinned Ultralytics anchor-based model. "
                  "Both start from COCO weights with the matching VOC class channels copied into their prediction "
                  "heads, then learn on VOC train. Early backbone blocks are frozen and batch normalization uses "
                  "the pretrained running statistics. Images are letterboxed, and training uses random horizontal flips.", text),
        Paragraph("Both use cosine learning-rate decay and gradient norm "
                  "clipping at 10. Faster R-CNN sums RPN objectness and box losses plus ROI classifier and box losses. "
                  "YOLOv3 uses box regression, objectness and class losses.", text),
        table(training, [176, 142, 142]),
        Paragraph(f'{link("http://host.robots.ox.ac.uk/pascal/VOC/voc2007/", "Official VOC2007")} &nbsp; '
                  f'{link(CODE, "Code and reproduction instructions")} &nbsp; '
                  f'{link(f"{RELEASE}/faster-rcnn.pt", "Faster R-CNN weights")} &nbsp; '
                  f'{link(f"{RELEASE}/yolov3.pt", "YOLOv3 weights")}', compact),
        PageBreak(),
        Paragraph("Learning curves and category AP", title),
        preview(out / "curves.png", 500),
        Paragraph("The curves use the same per-epoch values logged to TensorBoard. Dots and vertical markers identify "
                  "the validation-best epochs. Compare objective values within each model. Validation mAP50 gives "
                  "the shared measure across models.", text),
        preview(out / "per-class.png", 500),
        Paragraph(f"The largest absolute test AP gap was for <b>{escape(largest)}</b>, "
                  f"at {100 * differences[largest]:+.2f} points for YOLOv3 relative to Faster R-CNN. "
                  "The test mAP for each detector was independently reconstructed from its 20 category APs.", text),
        PageBreak(),
        Paragraph("Region proposals on the official test set", title),
        preview(out / "rpn.png", 485, max_height=520),
        Paragraph(f"Four images were selected by ground-truth class coverage before observing predictions. "
                  f"From left to right each row shows ground truth, up to {visual['rpn_limit']} ranked proposals "
                  f"intersecting the image, and final Faster R-CNN predictions at score {threshold:.2f} or higher. "
                  "The confidence threshold controls the displayed detections.", text),
    ]
    for row in test_examples:
        story.append(Paragraph(
            f"Image {escape(row['id'])} has {len(row['ground_truth'])} ground-truth boxes, "
            f"{len(row['proposals'])} proposals shown and {len(row['faster_rcnn'])} final detections shown.", compact))
    story.extend([
        PageBreak(),
        Paragraph("External-image detections", title),
        Paragraph(f"The images use a shared display cutoff of {threshold:.2f}. Each figure credits the original "
                  "photographer and includes its source and reuse terms. The public JSON lists the original-image "
                  "coordinates, labels and scores.", text),
    ])
    for name in ("bicycle", "cat", "cow"):
        example = examples[f"{name}.jpg"]
        notes = []
        for key, label in MODELS.items():
            target = [float(box["score"]) for box in example[key] if box["label"] == name]
            action = f"{max(target):.2f} best {name} score" if target else f"no {name} above the cutoff"
            count = len(example[key])
            noun = "box" if count == 1 else "boxes"
            notes.append(f"{label} shows {count} {noun} ({action})")
        story.extend([
            preview(out / f"{name}-annotated.jpg", 405, max_height=168),
            Paragraph(f"{escape(name.capitalize())}. {escape(notes[0])}. {escape(notes[1])}. "
                      f'{link(example["credit"]["source_url"], "Source photo")} &nbsp; '
                      f'{link(example["credit"]["license_url"], example["credit"]["adaptation_license"])}', compact),
        ])
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Directory with visualization inputs and report outputs")
    args = parser.parse_args()
    records = load_runs(args.run)
    visual = read_json(args.out / "visualizations.json")
    plot_curves(records, args.out)
    plot_ap(records, args.out)
    build_pdf(records, visual, args.out)


if __name__ == "__main__":
    main()
