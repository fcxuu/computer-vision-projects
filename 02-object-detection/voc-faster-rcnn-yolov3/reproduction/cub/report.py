"""Build the CUB-200-2011 report from frozen trial evidence."""

import argparse
import csv
import json
import math
from pathlib import Path
import textwrap
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Table, TableStyle

REPO = "https://github.com/fcxuu/computer-vision-projects"
CODE = f"{REPO}/tree/portfolio/reproductions-2026-11-23/02-object-detection/voc-faster-rcnn-yolov3/reproduction/cub"
RELEASE = f"{REPO}/releases/download/portfolio-cub200-2026-11-23"
DOI = "https://doi.org/10.22002/D1.20098"
TRIALS = ("scratch", "transfer", "transfer_low", "transfer_high")
PALETTE = ("#a84d36", "#16706b", "#516baa", "#8b589d")
INK = colors.HexColor("#183049")


def score(record):
    return float(record["val_accuracy"]), -float(record["val_loss"])


def check_test(test, expected_epoch, trial):
    matrix = np.asarray(test["confusion"])
    if matrix.shape != (200, 200) or matrix.dtype.kind not in "iu" or (matrix < 0).any():
        raise ValueError(f"Expected integer 200 x 200 confusion counts for {trial}")
    row_counts = matrix.sum(axis=1)
    samples, correct = int(row_counts.sum()), int(matrix.trace())
    if (row_counts <= 0).any() or samples != test["samples"] or correct != test["correct"]:
        raise ValueError(f"Confusion totals disagree with the reported sample/correct counts for {trial}")
    accuracy = correct / samples
    macro = float((matrix.diagonal() / row_counts).mean())
    for actual, expected in ((accuracy, test["accuracy"]), (macro, test["macro_recall"])):
        if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"Confusion-derived metrics disagree with reported metrics for {trial}")
    top5_correct = round(test["top5_accuracy"] * samples)
    if not correct <= top5_correct <= samples or not math.isclose(
        top5_correct / samples, test["top5_accuracy"], rel_tol=0, abs_tol=1e-9
    ):
        raise ValueError(f"Top-5 rate is incompatible with the sample count for {trial}")
    if test["epoch"] != expected_epoch or test["trial"] != trial:
        raise ValueError(f"Test metadata disagrees with the frozen validation checkpoint for {trial}")
    return {**test, "correct": correct, "accuracy": accuracy, "macro_recall": macro,
            "top5_correct": top5_correct, "confusion": matrix}


def load_trials(run):
    results = {}
    for name in TRIALS:
        trial = run / name
        train = json.loads((trial / "train.json").read_text())
        with (trial / "history.csv").open(newline="") as stream:
            history = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
        if len(history) != train["epochs"] or [row["epoch"] for row in history] != list(range(1, train["epochs"] + 1)):
            raise ValueError(f"Incomplete or unordered epoch history for {name}")
        if not all(math.isfinite(value) for row in history for value in row.values()):
            raise ValueError(f"Non-finite epoch metrics for {name}")
        best = max(history, key=score)
        if train["trial"] != name or best["epoch"] != train["best"]["epoch"] or score(best) != score(train["best"]):
            raise ValueError(f"Validation-best record disagrees with the training history for {name}")
        test = check_test(json.loads((trial / "test.json").read_text()), best["epoch"], name)
        results[name] = {"train": train, "history": history, "test": test}
    protocol = ("batch", "seed", "image_size", "n_train", "n_val", "num_classes")
    if any(tuple(result["train"][key] for key in protocol) != tuple(results["scratch"]["train"][key] for key in protocol)
           for result in results.values()):
        raise ValueError("Trial metadata describes different dataset or preprocessing budgets")
    support = results["scratch"]["test"]["confusion"].sum(axis=1)
    if any(not np.array_equal(result["test"]["confusion"].sum(axis=1), support) for result in results.values()):
        raise ValueError("Trials disagree on official test support by class")
    return results


def save_plot(figure, output):
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


def plot_curves(results, winner, output):
    figure, axes = plt.subplots(1, 3, figsize=(10.6, 3.7), layout="constrained")
    for (name, result), color in zip(results.items(), PALETTE):
        history = result["history"]
        epochs = [row["epoch"] for row in history]
        for axis, metric in zip(axes, ("train_loss", "val_loss", "val_accuracy")):
            axis.plot(epochs, [row[metric] for row in history], label=name.replace("_", " "), color=color)
            if name in ("scratch", winner):
                best = result["train"]["best"]
                axis.scatter(best["epoch"], best[metric], color=color, edgecolors="white", zorder=4)
    for axis, title in zip(axes, ("Online training CE", "Validation CE", "Validation top-1")):
        axis.set(xlabel="Epoch", title=title)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=0.22)
    axes[2].yaxis.set_major_formatter(PercentFormatter(1))
    axes[2].legend(frameon=False, loc="lower right", fontsize=8)
    save_plot(figure, output)


def class_names(run):
    path = run / "classes.json"
    if not path.is_file():
        return [f"Class {index + 1}" for index in range(200)]
    names = json.loads(path.read_text())
    if not isinstance(names, list) or len(names) != 200 or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("classes.json must contain 200 nonempty class names in label order")
    return [name.replace("_", " ") for name in names]


def plot_errors(scratch, transfer, names, output):
    matrices = [scratch["confusion"].copy(), transfer["confusion"].copy()]
    for matrix in matrices:
        np.fill_diagonal(matrix, 0)
    ranking = matrices[0] + matrices[1]
    pairs = [np.unravel_index(flat, ranking.shape)
             for flat in np.argsort(-ranking.ravel(), kind="stable")[:8] if ranking.flat[flat] > 0]
    figure, axis = plt.subplots(figsize=(9.5, 6.2), layout="constrained")
    axis.set(title="Most frequent ordered error pairs across both checkpoints", xlabel="Official test images")
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    if pairs:
        positions = np.arange(len(pairs))
        for matrix, color, offset, label in zip(matrices, PALETTE, (-0.18, 0.18), ("Scratch", "Selected transfer")):
            values = [int(matrix[row, col]) for row, col in pairs]
            bars = axis.barh(positions + offset, values, height=0.33, color=color, label=label)
            axis.bar_label(bars, fontsize=8, padding=2)
        labels = [textwrap.fill(f"{names[row]} predicted as {names[col]}", width=38) for row, col in pairs]
        axis.set(yticks=positions, yticklabels=labels)
        axis.tick_params(axis="y", labelsize=8)
        axis.invert_yaxis()
        axis.margins(x=0.15)
        axis.legend(frameon=False)
    else:
        axis.text(0.5, 0.5, "No off-diagonal errors", transform=axis.transAxes, ha="center")
    axis.grid(axis="x", alpha=0.18)
    save_plot(figure, output)
    return [(names[row], names[col], int(matrices[0][row, col]), int(matrices[1][row, col])) for row, col in pairs]


def footer(canvas, document):
    canvas.setFillColor(INK)
    canvas.setFont("Helvetica", 9)
    canvas.drawString(43, 27, "fcxuu / computer-vision-projects")
    canvas.drawRightString(A4[0] - 43, 27, str(document.page))


def styled_table(values, widths):
    result = Table(values, colWidths=widths, hAlign="LEFT")
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#edf4f5"), colors.white]),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
    ]))
    return result


def build_report(results, winner, errors, out):
    doc = SimpleDocTemplate(str(out / "report.pdf"), pagesize=A4, leftMargin=43, rightMargin=43,
                            topMargin=40, bottomMargin=53, title="Bird species classification on CUB-200-2011",
                            author="fcxuu / computer-vision-projects")
    title_style = ParagraphStyle("Title", fontName="Helvetica-Bold", fontSize=24, leading=28,
                                 textColor=INK, spaceAfter=16)
    head_style = ParagraphStyle("Heading", fontName="Helvetica-Bold", fontSize=14, leading=18,
                                textColor=INK, spaceBefore=12, spaceAfter=5)
    body_style = ParagraphStyle("Body", fontName="Helvetica", fontSize=10, leading=14,
                                textColor=INK, spaceBefore=4, spaceAfter=6)

    def body(value):
        return Paragraph(value, body_style)

    def heading(value):
        return Paragraph(value, head_style)

    def link(url, label):
        return f'<link href="{url}" color="#16706b"><u>{label}</u></link>'

    first, selected = results["scratch"]["test"], results[winner]["test"]
    chosen_epoch = results[winner]["train"]["best"]
    protocol = results["scratch"]["train"]
    difference = 100 * (selected["accuracy"] - first["accuracy"])
    training = [["Trial", "Body LR", "Head LR", "Frozen", "Epochs", "Steps", "Best", "Val top-1"]]
    for name, result in results.items():
        config = result["train"]
        steps = config["epochs"] * math.ceil(config["n_train"] / config["batch"])
        training.append([name, f"{config['body_lr']:g}", f"{config['head_lr']:g}", config["warmup"],
                         config["epochs"], f"{steps:,}", config["best"]["epoch"], f"{config['best']['val_accuracy']:.2%}"])
    test_rows = [["Frozen checkpoint", "Top-1", "Top-5", "Macro recall"]]
    for name, metrics in (("Scratch", first), ("Selected transfer", selected)):
        samples = metrics["samples"]
        test_rows.append([name, f"{metrics['accuracy']:.2%} ({metrics['correct']:,}/{samples:,})",
                          f"{metrics['top5_accuracy']:.2%} ({metrics['top5_correct']:,}/{samples:,})",
                          f"{metrics['macro_recall']:.2%}"])
    note = "Both checkpoints classified every test example correctly."
    if errors:
        source, prediction, before, after = errors[0]
        note = (f"The most frequent ordered error across the two checkpoints was true {escape(source)} "
                f"predicted as {escape(prediction)}. It occurred {before:,} times with scratch "
                f"and {after:,} times with selected transfer.")
    doc.build([
        Paragraph("Bird species classification on CUB-200-2011", title_style),
        body(f"Selected ImageNet transfer changed official test top-1 by <b>{difference:+.2f} percentage points</b> "
             f"relative to scratch. The selected transfer trial was <b>{winner.replace('_', ' ')}</b>."),
        heading("Data and training protocol"),
        body(f"ResNet-18 with a 200-class linear head was trained on {protocol['n_train']:,} images and selected "
             f"using {protocol['n_val']:,} validation images drawn from the official training split. The held-out official "
             f"test set contains {first['samples']:,} images. Each trial used a batch size of {protocol['batch']}, "
             f"a seed of {protocol['seed']}, and {protocol['image_size']} x {protocol['image_size']} inputs. "
             f"The model has {protocol['n_parameters']:,} parameters and trained on {protocol['device'].upper()}."),
        body("Scratch starts from random weights. The transfer trials start from ImageNet weights and train the "
             "new head during the frozen-body period before fine-tuning the full network. All trials use "
             f"cross-entropy, AdamW with weight decay {protocol['weight_decay']:g}, "
             "and cosine learning-rate decay."),
        heading("Trial budgets and validation selection"),
        styled_table(training, [102, 58, 57, 55, 51, 60, 49, 72]),
        body("LRs are initial optimizer rates. Frozen and Best are epoch counts and epoch number, respectively. "
             "Steps count minibatch optimizer updates. Checkpoints and the transfer variant are selected by highest "
             "validation top-1, then lowest validation cross-entropy."),
        heading("Official test results"),
        styled_table(test_rows, [134, 126, 126, 118]),
        body("Macro recall is the unweighted mean of the 200 class recalls. Top-1 counts and macro recall "
             "were independently recomputed from the confusion matrix."),
        body(f'{link(DOI, "Caltech dataset (DOI 10.22002/D1.20098)")} &nbsp; {link(CODE, "Code and reproduction instructions")}'),
        body(f'{link(f"{RELEASE}/scratch.pt", "Download scratch.pt")} &nbsp; '
             f'{link(f"{RELEASE}/best-transfer.pt", "Download best-transfer.pt")}'),
        PageBreak(),
        Paragraph("Training and validation curves", title_style),
        Image(str(out / "curves.png"), width=494, height=494 * 3.7 / 10.6),
        body("Training CE aggregates the predictions made before each optimizer step. Validation CE and top-1 "
             "are evaluated after each epoch. Dots mark the selected scratch and transfer checkpoints."),
        body(f"At the selected transfer epoch, online training CE was {chosen_epoch['train_loss']:.3f} "
             f"and validation CE was {chosen_epoch['val_loss']:.3f}. The validation loss measures "
             "generalization to images outside the supervised updates."),
        heading("Official test errors"),
        Image(str(out / "test-errors.png"), width=470, height=470 * 6.2 / 9.5),
        body("Pairs are ranked by the total number of errors across the two frozen checkpoints. "
             "The direction of each pair is true class to predicted class."),
        body(note),
    ], onFirstPage=footer, onLaterPages=footer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results = load_trials(args.run)
    winner = max(TRIALS[1:], key=lambda name: score(results[name]["train"]["best"]))
    args.out.mkdir(parents=True, exist_ok=True)
    plot_curves(results, winner, args.out / "curves.png")
    errors = plot_errors(results["scratch"]["test"], results[winner]["test"], class_names(args.run),
                         args.out / "test-errors.png")
    build_report(results, winner, errors, args.out)


if __name__ == "__main__":
    main()
