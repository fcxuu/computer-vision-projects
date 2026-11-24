"""Report the CUB-200-2011 transfer comparison after 224-pixel fine-tuning."""

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate

SPEC = importlib.util.spec_from_file_location("cub_first_round_report", Path(__file__).resolve().parents[1] / "report.py")
FIRST = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIRST)

RELEASE = f"{FIRST.REPO}/releases/download/portfolio-cub224-2026-11-24"
CALTECH = "https://www.vision.caltech.edu/datasets/cub_200_2011/"
TARGETS = (0.65, 0.68)
TRIAL = "transfer"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def close(left, right):
    return math.isclose(left, right, abs_tol=1e-9, rel_tol=0)


def checked_metrics(path, config, *, samples, part, flip, resize):
    row = read_json(path)
    if type(row["samples"]) is not int or type(row["correct"]) is not int or row["samples"] != samples:
        raise ValueError(f"Invalid recorded sample or correct count in {path}")
    if (row.get("split", "test"), row.get("flip_tta", False), row["image_size"], row.get("resize_mode", "square")) != (
        part, flip, config["image_size"], resize
    ) or type(row.get("flip_tta", False)) is not bool:
        raise ValueError(f"Evaluation split, resize or flip selection mismatch in {path}")
    metrics = FIRST.check_test(row, config["best"]["epoch"], config["trial"])
    if "top5_correct" in row and row["top5_correct"] != metrics["top5_correct"]:
        raise ValueError(f"Top-5 count disagrees with the reported rate in {path}")
    if not math.isfinite(metrics["loss"]) or metrics["loss"] < 0:
        raise ValueError(f"Invalid cross-entropy in {path}")
    return metrics


def read_trial(path, *, optimized):
    config = read_json(path / "train.json")
    if (config["trial"], config["initialization"], config["num_classes"]) != (TRIAL, "imagenet", 200):
        raise ValueError(f"Unexpected CUB classification protocol in {path}")
    profile = config["image_size"], config.get("resize_mode", "square"), config.get("label_smoothing", 0.0)
    if optimized and (profile != (224, "letterbox", 0.1) or config["epochs"] != 24):
        raise ValueError("Expected the 224-pixel, letterboxed, smoothed 24-epoch trial")
    with (path / "history.csv").open(newline="") as stream:
        rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    if not rows or [row["epoch"] for row in rows] != list(range(1, config["epochs"] + 1)):
        raise ValueError(f"Incomplete or unordered epoch history in {path}")
    if not all(math.isfinite(value) for row in rows for value in row.values()):
        raise ValueError(f"Non-finite epoch records in {path}")
    best = max(rows, key=FIRST.score)
    if best != config["best"]:
        raise ValueError(f"Validation-best checkpoint disagrees with the history in {path}")
    split_path = path / "split.npz"
    if not optimized and not split_path.is_file():
        split_path = path.parent / "split.npz"
    with np.load(split_path, allow_pickle=False) as archive:
        train, validation = archive["train"], archive["validation"]
    if (train.shape != (config["n_train"],) or validation.shape != (config["n_val"],)
            or len(train) + len(validation) != 5994 or min(len(train), len(validation)) < 1
            or train.dtype.kind not in "iu" or validation.dtype.kind not in "iu"):
        raise ValueError(f"Incorrect saved split sizes or index type in {path}")
    ids = np.r_[train, validation]
    if len(np.unique(ids)) != len(ids) or ids.min() < 1 or ids.max() > 11788:
        raise ValueError(f"Training and validation indices overlap or are out of bounds in {path}")
    methods = {}
    if optimized:
        for method, flip in (("plain", False), ("flip", True)):
            methods[method] = checked_metrics(
                path / f"validation-{method}.json", config, samples=len(validation),
                part="validate", flip=flip, resize=profile[1],
            )
        plain = methods["plain"]
        if (not close(plain["accuracy"], best["val_accuracy"])
                or not close(plain["top5_accuracy"], best["val_top5_accuracy"])
                or not math.isclose(plain["loss"], best["val_loss"], abs_tol=1e-4, rel_tol=0)):
            raise ValueError("Replayed single-view validation does not reproduce the selected epoch")
        if not np.array_equal(plain["confusion"].sum(axis=1), methods["flip"]["confusion"].sum(axis=1)):
            raise ValueError("Validation methods have different class support")
    method = max(methods, key=lambda key: (methods[key]["accuracy"], -methods[key]["loss"])) if methods else "plain"
    test = checked_metrics(path / "test.json", config, samples=5794, part="test", flip=method == "flip", resize=profile[1])
    return {"config": config, "profile": profile, "history": rows, "best": best, "split": (train, validation),
            "methods": methods, "method": method, "test": test}


def load_comparison(baseline, run):
    old = read_trial(baseline / "transfer", optimized=False)
    new = read_trial(run / "transfer-224", optimized=True)
    if any(not np.array_equal(a, b) for a, b in zip(old["split"], new["split"])):
        raise ValueError("First- and second-round training/validation image IDs differ")
    if old["config"]["seed"] != new["config"]["seed"] or old["config"]["n_parameters"] != new["config"]["n_parameters"]:
        raise ValueError("Seed or ResNet-18 parameter count changed")
    if not np.array_equal(old["test"]["confusion"].sum(axis=1), new["test"]["confusion"].sum(axis=1)):
        raise ValueError("Official test class support differs between checkpoints")
    return old, new


def draw_curves(before, after, out):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.55), layout="constrained")
    for result, label, color in ((before, "First round", "#9d4938"), (after, "224-pixel transfer", "#16726c")):
        rows, best = result["history"], result["best"]
        for axis, key in zip(axes, ("val_loss", "val_accuracy", "val_top5_accuracy")):
            axis.plot([row["epoch"] for row in rows], [row[key] for row in rows], color=color, label=label)
            axis.scatter(best["epoch"], best[key], s=20, color=color, zorder=4)
    for axis, title in zip(axes, ("Validation CE", "Validation top-1", "Validation top-5")):
        axis.set(title=title, xlabel="Epoch")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=0.18)
    for axis in axes[1:]:
        axis.yaxis.set_major_formatter(PercentFormatter(1))
    axes[2].legend(frameon=False, fontsize=8)
    FIRST.save_plot(fig, out / "validation-curves.png")


def draw_errors(before, after, names, out):
    orig, new = (trial["test"]["confusion"] for trial in (before, after))
    support = orig.sum(axis=1)
    changes = (new.diagonal() - orig.diagonal()) / support
    chosen = np.argsort(-np.abs(changes), kind="stable")[:10]
    fig, axis = plt.subplots(figsize=(9.6, 4.35), layout="constrained")
    slots = np.arange(len(chosen))
    axis.barh(slots, changes[chosen], color=["#16726c" if changes[i] >= 0 else "#9d4938" for i in chosen])
    axis.set(title="Classes with the largest absolute official test recall changes",
             xlabel="Recall change (percentage points)", yticks=slots,
             yticklabels=[textwrap.fill(names[i], width=31) for i in chosen])
    axis.axvline(0, color="#183049", linewidth=0.7)
    axis.xaxis.set_major_formatter(PercentFormatter(1))
    axis.tick_params(axis="y", labelsize=8)
    axis.invert_yaxis()
    axis.grid(axis="x", alpha=0.18)
    FIRST.save_plot(fig, out / "class-differences.png")
    return names[chosen[0]], float(changes[chosen[0]])


def build_pdf(old, new, class_change, out):
    from xml.sax.saxutils import escape

    title = ParagraphStyle("Title", fontName="Helvetica-Bold", fontSize=24, leading=28, textColor=FIRST.INK, spaceAfter=13)
    head = ParagraphStyle("Heading", fontName="Helvetica-Bold", fontSize=14, leading=18,
                          textColor=FIRST.INK, spaceBefore=10, spaceAfter=6)
    body = ParagraphStyle("Body", fontName="Helvetica", fontSize=10, leading=14, textColor=FIRST.INK, spaceAfter=7)
    doc = SimpleDocTemplate(str(out / "report.pdf"), pagesize=A4, leftMargin=43, rightMargin=43,
                            topMargin=40, bottomMargin=53, title="Higher-resolution bird classification on CUB-200-2011",
                            author="fcxuu / computer-vision-projects")
    before, after = old["test"], new["test"]
    test_rows = [["Frozen checkpoint", "Val best ep.", "Test top-1", "Test top-5", "Macro recall"]]
    setup = [["Setting", "First round", "Second round"]]
    for name, run in (("First-round transfer", old), ("224-pixel transfer", new)):
        test = run["test"]
        test_rows.append([name, int(run["best"]["epoch"]), f"{test['accuracy']:.2%} ({test['correct']:,})",
                          f"{test['top5_accuracy']:.2%} ({test['top5_correct']:,})", f"{test['macro_recall']:.2%}"])
    for label, key in (("Epochs", "epochs"), ("Batch size", "batch"), ("Initial body LR", "body_lr"), ("Initial head LR", "head_lr")):
        setup.append([label, *[f"{run['config'][key]:g}" for run in (old, new)]])
    setup.append(["Input side / resize", *[f"{run['profile'][0]} px / {run['profile'][1]}" for run in (old, new)]])
    setup.append(["Label smoothing", *[f"{run['profile'][2]:g}" for run in (old, new)]])
    setup.append(["Official test inference", "Single view", "Flip average" if new["method"] == "flip" else "Single view"])
    val_rows = [["Validation inference", "Top-1", "Top-5", "Cross-entropy", "Macro recall"]]
    for method, label in (("plain", "Single view"), ("flip", "Average original + flip logits")):
        row = new["methods"][method]
        val_rows.append([label, f"{row['accuracy']:.2%}", f"{row['top5_accuracy']:.2%}",
                         f"{row['loss']:.4f}", f"{row['macro_recall']:.2%}"])
    target_notes = []
    for target in TARGETS:
        difference = 100 * (after["accuracy"] - target)
        target_notes.append(f"{target:.0%} {'target' if target == 0.65 else 'stretch goal'} "
                            f"({'met' if difference >= 0 else 'unmet'}, {difference:+.2f} points)")
    config = new["config"]
    changes = [100 * (after[key] - before[key]) for key in ("accuracy", "top5_accuracy", "macro_recall")]
    leaf, gap = class_change
    doc.build([
        Paragraph("Higher-resolution CUB-200-2011 transfer", title),
        Paragraph(f"The second-round checkpoint scored <b>{after['accuracy']:.2%}</b> official test top-1 "
                  f"({changes[0]:+.2f} points versus first-round transfer). "
                  f"The result is measured against the {' and the '.join(target_notes)}.", body),
        Paragraph("Matched data and combined changes", head),
        Paragraph(f"Both trials use ImageNet-initialized ResNet-18 with {config['n_parameters']:,} parameters. "
                  f"The saved CUB image IDs agree exactly across {config['n_train']:,} training and "
                  f"{config['n_val']:,} validation images from the 5,994-image official training set. "
                  f"Tests use the 5,794-image official test set. The second round combines larger inputs, "
                  "aspect-ratio-preserving padding, label smoothing and a revised training budget. "
                  "This comparison estimates their joint effect with validation-selected inference.", body),
        FIRST.styled_table(setup, [183, 143, 143]),
        Paragraph("Checkpoint and inference selection", head),
        Paragraph("Each checkpoint is selected by validation top-1, then by lower validation cross-entropy. "
                  "On that fixed second-round checkpoint, single-view and flip-averaged logits are compared "
                  "by the same validation rule, with single view winning an exact tie. "
                  "The official test uses the selected mode.", body),
        FIRST.styled_table(val_rows, [168, 70, 70, 83, 77]),
        Paragraph("Official test results", head),
        FIRST.styled_table(test_rows, [122, 65, 100, 100, 86]),
        Paragraph(f"Test top-5 changed by {changes[1]:+.2f} points and macro recall by {changes[2]:+.2f} points. "
                  "Top-1 counts and macro recall were recomputed from the confusion matrices. "
                  "Top-5 counts were checked against the recorded rates and denominator.", body),
        Paragraph(f'<link href="{FIRST.CODE}/evidence/report.pdf" color="#16726c"><u>First-round report</u></link> &nbsp; '
                  f'<link href="{FIRST.CODE}/optimization" color="#16726c"><u>Optimization code</u></link> &nbsp; '
                  f'<link href="{RELEASE}/cub-transfer-224.pt" color="#16726c"><u>224-pixel checkpoint</u></link>', body),
        PageBreak(),
        Paragraph("Validation curves and class differences", title),
        Image(str(out / "validation-curves.png"), width=499, height=499 * 3.55 / 10.5),
        Paragraph("Dots mark selected epochs. Histories show single-view validation; the separate inference "
                  "comparison uses the fixed second-round checkpoint.", body),
        Image(str(out / "class-differences.png"), width=499, height=499 * 4.35 / 9.6),
        Paragraph(f"The largest absolute change in per-class recall was for {escape(leaf)} "
                  f"({100 * gap:+.2f} percentage points). Each recall uses the same official class support in both runs.", body),
        Paragraph("Caltech documents overlap between CUB and ImageNet images. Some official CUB test images may "
                  "therefore have appeared during ImageNet pretraining. Both checkpoints share that starting point. "
                  f'<link href="{CALTECH}" color="#16726c"><u>Official CUB dataset and overlap notice</u></link>', body),
    ], onFirstPage=FIRST.footer, onLaterPages=FIRST.footer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    old, new = load_comparison(args.baseline, args.run)
    args.out.mkdir(parents=True, exist_ok=True)
    names_root = args.run if (args.run / "classes.json").is_file() else args.baseline
    names = FIRST.class_names(names_root)
    draw_curves(old, new, args.out)
    largest = draw_errors(old, new, names, args.out)
    build_pdf(old, new, largest, args.out)


if __name__ == "__main__":
    main()
