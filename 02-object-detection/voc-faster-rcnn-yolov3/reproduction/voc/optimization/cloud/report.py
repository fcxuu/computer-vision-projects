"""Build the VOC2007 Faster R-CNN cloud training comparison."""

import argparse
import importlib.util
import math
from pathlib import Path
import re
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate

SPEC = importlib.util.spec_from_file_location("voc_first_round_report", Path(__file__).resolve().parents[2] / "report.py")
FIRST = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIRST)

RELEASE = f"{FIRST.REPO}/releases/download/portfolio-voc2007-cloud-2026-11-25"
TRAIN_IMAGES, VAL_IMAGES, TEST_IMAGES = 2501, 2510, 4952
CLOUD_EPOCHS = 8
CLOUD_SIZE = 320
PER_GPU_BATCH = 8
VARIANT = "faster_rcnn"


def phase_metrics(path, images):
    record = FIRST.read_json(path)
    mean, aps = FIRST.verified_map(record, images, path)
    support = tuple(record["per_class"][label]["positives"] for label in FIRST.CLASSES)
    if any(type(count) is not int or count <= 0 for count in support):
        raise ValueError(f"Invalid class ground-truth support in {path}")
    return {"metadata": record, "map": mean, "aps": aps, "support": support}


def check_checkpoint(metrics, epoch, size, *, require_epoch):
    config = metrics["metadata"]
    if (config.get("variant", VARIANT) != VARIANT or config.get("size", size) != size
            or (require_epoch and config.get("epoch") != epoch)
            or ("epoch" in config and config["epoch"] != epoch)):
        raise ValueError("Evaluation metadata disagrees with the selected Faster R-CNN checkpoint")


def load_baseline(folder):
    config = FIRST.read_json(folder / "train.json")
    if (config["variant"], config["size"], config["epochs"], config["batch"], config["device"],
            config["n_train"], config["n_validation"]) != (VARIANT, 256, 5, 4, "cpu", TRAIN_IMAGES, VAL_IMAGES):
        raise ValueError("Baseline metadata does not identify the first-round CPU Faster R-CNN")
    validation = phase_metrics(folder / "best-val.json", VAL_IMAGES)
    test = phase_metrics(folder / "test.json", TEST_IMAGES)
    best = config["best"]
    check_checkpoint(validation, best["epoch"], config["size"], require_epoch=False)
    check_checkpoint(test, best["epoch"], config["size"], require_epoch=True)
    if (not 1 <= best["epoch"] <= config["epochs"]
            or not FIRST.close(validation["map"], best["val_map50"])):
        raise ValueError("Baseline validation record disagrees with selected epoch metadata")
    return {"config": config, "validation": validation, "test": test, "best_epoch": best["epoch"]}


def load_provenance(folder):
    provenance = FIRST.read_json(folder / "provenance.json")
    names = provenance["gpu_names"]
    peak = provenance["peak_bytes_per_gpu"]
    if (not isinstance(names, list) or len(names) != 2 or any("T4" not in name for name in names)
            or not isinstance(peak, list) or len(peak) != len(names)
            or any(type(size) is not int or size <= 0 for size in peak)):
        raise ValueError("Expected two T4 GPU names with positive peak memory records")
    times = [provenance[key] for key in ("elapsed_to_test_seconds", "duration_epoch_first_to_last_seconds")]
    if not all(math.isfinite(value) and value > 0 for value in times) or times[1] > times[0]:
        raise ValueError("Invalid cloud duration records")
    if not re.fullmatch("[0-9a-f]{7,64}", provenance["project_commit"]):
        raise ValueError("Invalid submitted project commit")
    digest = provenance["submitted_sha256"]
    if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
        raise ValueError("Invalid submitted SHA-256")
    return provenance


def load_cloud(folder):
    payload = FIRST.read_json(folder / "history.json")
    rows = payload["history"] if isinstance(payload, dict) else payload
    if [row["epoch"] for row in rows] != list(range(1, CLOUD_EPOCHS + 1)):
        raise ValueError("Cloud history must contain eight ordered epochs")
    if any(not math.isfinite(row["train_loss"]) or not math.isfinite(row["val_loss"])
           or not 0 <= row["val_map50"] <= 1 for row in rows):
        raise ValueError("Non-finite or out-of-range cloud history")
    best = max(rows, key=lambda row: row["val_map50"])
    validation = phase_metrics(folder / "best-val.json", VAL_IMAGES)
    test = phase_metrics(folder / "test.json", TEST_IMAGES)
    check_checkpoint(validation, best["epoch"], CLOUD_SIZE, require_epoch=False)
    check_checkpoint(test, best["epoch"], CLOUD_SIZE, require_epoch=True)
    if (not FIRST.close(validation["map"], best["val_map50"])
            or not FIRST.close(validation["metadata"]["loss"], best["val_loss"])):
        raise ValueError("Cloud validation record disagrees with the earliest highest-mAP epoch")
    provenance = load_provenance(folder)
    local_path = folder / "local-test.json"
    local = phase_metrics(local_path, TEST_IMAGES) if local_path.is_file() else None
    if local:
        check_checkpoint(local, best["epoch"], CLOUD_SIZE, require_epoch=True)
        if local["support"] != test["support"]:
            raise ValueError("Local reload and cloud test use different class support")
    return {"history": rows, "validation": validation, "test": test, "local": local,
            "best_epoch": best["epoch"], "provenance": provenance}


def curves(baseline, cloud, out):
    fig, (loss, validation) = plt.subplots(1, 2, figsize=(10.6, 3.8), layout="constrained")
    rows = cloud["history"]
    epochs = [row["epoch"] for row in rows]
    loss.plot(epochs, [row["train_loss"] for row in rows], color="#17726c", marker="o", markersize=4)
    validation.plot(epochs, [row["val_map50"] for row in rows], color="#17726c", marker="o",
                    markersize=4, label="Cloud validation")
    validation.axhline(baseline["validation"]["map"], linestyle="--", color="#a65039",
                       label="Baseline selected validation mAP50")
    validation.yaxis.set_major_formatter(PercentFormatter(1))
    validation.legend(frameon=False, fontsize=8)
    for axis, title in ((loss, "Cloud online training objective"), (validation, "Validation mAP50")):
        axis.set(title=title, xlabel="Epoch")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.axvline(cloud["best_epoch"], color="#64788b", linestyle=":", linewidth=1)
        axis.grid(alpha=0.18)
    FIRST.save_plot(fig, out / "learning-curves.png")


def plot_ap(baseline, cloud, out):
    labels = FIRST.CLASSES
    fig, axis = plt.subplots(figsize=(10.2, 5.55), layout="constrained")
    positions = np.arange(len(labels))
    for result, offset, color, name in ((baseline, -0.18, "#a65039", "CPU baseline"),
                                         (cloud, 0.18, "#17726c", "Cloud DDP")):
        axis.barh(positions + offset, [result["test"]["aps"][label] for label in labels],
                  height=0.34, color=color, label=name)
    axis.set(yticks=positions, yticklabels=labels, xlabel="VOC2007 11-point test AP at IoU 0.50", xlim=(0, 1))
    axis.xaxis.set_major_formatter(PercentFormatter(1))
    axis.tick_params(axis="y", labelsize=8)
    axis.grid(axis="x", alpha=0.18)
    axis.invert_yaxis()
    axis.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    FIRST.save_plot(fig, out / "per-class-ap.png")


def build_pdf(baseline, cloud, out):
    title = ParagraphStyle("Title", fontName="Helvetica-Bold", fontSize=24, leading=28, textColor=FIRST.INK, spaceAfter=13)
    head = ParagraphStyle("Heading", fontName="Helvetica-Bold", fontSize=14, leading=18,
                          textColor=FIRST.INK, spaceBefore=10, spaceAfter=5)
    body = ParagraphStyle("Body", fontName="Helvetica", fontSize=10, leading=14, textColor=FIRST.INK, spaceAfter=7)
    doc = SimpleDocTemplate(str(out / "report.pdf"), pagesize=A4, leftMargin=43, rightMargin=43,
                            topMargin=40, bottomMargin=53, title="Faster R-CNN on VOC2007 with dual T4 training",
                            author="fcxuu / computer-vision-projects")
    previous, current = baseline["test"]["map"], cloud["test"]["map"]
    config = baseline["config"]
    provenance = cloud["provenance"]
    setup = [
        ["Setting", "First-round Faster R-CNN", "Kaggle Faster R-CNN"],
        ["Device", config["device"].upper(), "2 x NVIDIA T4 (DDP)"],
        ["Input side", f"{config['size']} px", f"{CLOUD_SIZE} px"],
        ["Epochs", config["epochs"], CLOUD_EPOCHS],
        ["Training batch", config["batch"], f"{PER_GPU_BATCH} per GPU / {PER_GPU_BATCH * len(provenance['gpu_names'])} global"],
        ["Random horizontal flips", "Enabled", "Disabled"],
    ]
    results = [["Checkpoint", "Selected epoch", "Validation mAP50", "Official test mAP50"]]
    for label, run in (("First-round CPU", baseline), ("Cloud DDP", cloud)):
        results.append([label, run["best_epoch"], f"{run['validation']['map']:.2%}", f"{run['test']['map']:.2%}"])
    runtime = f"The cloud run took {provenance['elapsed_to_test_seconds'] / 60:.1f} min through official testing. "               f"The first-to-last recorded epoch span was {provenance['duration_epoch_first_to_last_seconds'] / 60:.1f} min."
    peak = ", ".join(f"{value / 1024**3:.2f} GiB" for value in provenance["peak_bytes_per_gpu"])
    class_gap = {label: cloud["test"]["aps"][label] - baseline["test"]["aps"][label] for label in FIRST.CLASSES}
    largest = max(class_gap, key=lambda label: abs(class_gap[label]))
    story = [
        Paragraph("Faster R-CNN on VOC2007 with dual T4 training", title),
        Paragraph(f"Official test mAP50 moved from <b>{previous:.2%}</b> to <b>{current:.2%}</b> "
                  f"({100 * (current - previous):+.2f} percentage points) in the cloud configuration.", body),
        Paragraph("Data and training protocol", head),
        Paragraph(f"Both runs initialize Faster R-CNN from COCO weights and train on VOC2007. They use the official "
                  f"{TRAIN_IMAGES:,} training, {VAL_IMAGES:,} validation and {TEST_IMAGES:,} test images. "
                  "Resolution, epoch budget, batch size, hardware and augmentation changed together. "
                  "The mAP difference estimates the joint configuration change.", body),
        FIRST.table(setup, [180, 147, 158]),
        Paragraph("Validation selection and held-out results", head),
        Paragraph("The cloud checkpoint is the earliest epoch with the maximum validation mAP50. Both runs score "
                  "detections at IoU 0.50 with VOC2007 11-point interpolation. Difficult objects are excluded "
                  "from recall and detections matching them are ignored. The report recomputes each mAP from the "
                  "20 recorded class APs and checks that test ground-truth support is identical.", body),
        FIRST.table(results, [138, 97, 120, 125]),
    ]
    local = cloud["local"]
    if local:
        story.append(Paragraph(f"A local reload of the cloud checkpoint measured {local['map']:.2%} official test mAP50, "
                               f"{100 * (local['map'] - current):+.4f} percentage points relative to the cloud test.", body))
    story.extend([
        Paragraph("Cloud provenance", head),
        Paragraph(f"{runtime} Reported peak device memory was {peak}. The submitted project commit was "
                  f"<font name='Courier'>{escape(provenance['project_commit'])}</font>.", body),
        Paragraph(f"Submitted SHA-256 <font name='Courier' size='8'>{escape(provenance['submitted_sha256'])}</font>.", body),
        Paragraph(
            " &nbsp; ".join((FIRST.link(f"{FIRST.CODE}/evidence/report.pdf", "First-round report"),
                             FIRST.link(f"{FIRST.CODE}/optimization/cloud", "Cloud optimization code"),
                             FIRST.link(f"{RELEASE}/faster-rcnn-320-t4.pt", "Download 320-pixel T4 checkpoint"))), body),
        PageBreak(),
        Paragraph("Learning curves and category AP", title),
        FIRST.preview(out / "learning-curves.png", 499),
        Paragraph("The vertical marker shows the selected cloud epoch. The dashed reference is the first-round "
                  "selected validation mAP50. The training objective is shown for the cloud run only.", body),
        FIRST.preview(out / "per-class-ap.png", 499),
        Paragraph(f"The largest absolute class AP change was <b>{escape(largest)}</b>, "
                  f"{100 * class_gap[largest]:+.2f} percentage points for the cloud configuration.", body),
    ])
    doc.build(story, onFirstPage=FIRST.footer, onLaterPages=FIRST.footer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--cloud", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    baseline, cloud = load_baseline(args.baseline), load_cloud(args.cloud)
    if baseline["test"]["support"] != cloud["test"]["support"]:
        raise ValueError("The first-round and cloud tests have different ground-truth class support")
    args.out.mkdir(parents=True, exist_ok=True)
    curves(baseline, cloud, args.out)
    plot_ap(baseline, cloud, args.out)
    build_pdf(baseline, cloud, args.out)


if __name__ == "__main__":
    main()
