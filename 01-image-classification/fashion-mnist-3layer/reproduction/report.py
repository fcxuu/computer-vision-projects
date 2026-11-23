"""Build the Fashion-MNIST report from a completed search and test run."""
import argparse
import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from model import load_checkpoint

REPO = "https://github.com/fcxuu/computer-vision-projects"
BRANCH = "portfolio/reproductions-2026-11-23"
PROJECT = "01-image-classification/fashion-mnist-3layer"
CODE_URL = f"{REPO}/tree/{BRANCH}/{PROJECT}/reproduction"
WEIGHTS_URL = f"{REPO}/raw/refs/heads/{BRANCH}/{PROJECT}/reproduction/evidence/best.npz"
NAVY = "#172d46"
TEAL = "#146c68"
RUST = "#b74c36"
GRAY = "#4d6073"


def save_plot(figure, path):
    figure.savefig(path, dpi=190, facecolor="white")
    plt.close(figure)


def plot_curves(history, best_epoch, path):
    figure, (loss, accuracy) = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    epochs = [int(row["epoch"]) for row in history]
    for field, label, color in [("train_ce", "Training", NAVY), ("val_ce", "Validation", TEAL)]:
        loss.plot(epochs, [float(row[field]) for row in history], label=label, color=color)
    accuracy.plot(epochs, [float(row["val_accuracy"]) for row in history], color=TEAL)
    for axis in (loss, accuracy):
        axis.axvline(best_epoch, color=RUST, linestyle="--", linewidth=1, label="Selected epoch")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.set_xlabel("Epoch")
        axis.legend(frameon=False)
        axis.grid(alpha=0.18)
    loss.set_title("Training and validation cross-entropy")
    loss.set_ylabel("Cross-entropy")
    accuracy.set_title("Validation accuracy")
    accuracy.yaxis.set_major_formatter(PercentFormatter(1))
    save_plot(figure, path)


def plot_parameters(weights, classes, path):
    figure = plt.figure(figsize=(10, 8), layout="constrained")
    grid = figure.add_gridspec(4, 4, height_ratios=[1, 1, 1.7, 1.1])
    w1 = weights["W1"]
    strongest = np.argsort(-np.linalg.norm(w1, axis=0), kind="stable")[:8]
    inputs = [figure.add_subplot(grid[index // 4, index % 4]) for index in range(8)]
    bound = float(np.abs(w1[:, strongest]).max()) or 1.0
    for axis, index in zip(inputs, strongest):
        pixels = axis.imshow(w1[:, index].reshape(28, 28), cmap="RdBu_r", vmin=-bound, vmax=bound)
        axis.set_title(f"Unit {index + 1}", fontsize=9)
        axis.axis("off")
    for axis in inputs[len(strongest):]:
        axis.axis("off")
    figure.colorbar(pixels, ax=inputs, shrink=0.75, label="W1 weight")
    output = figure.add_subplot(grid[2, :])
    w2 = weights["W2"]
    bound = float(np.abs(w2).max()) or 1.0
    links = output.imshow(w2.T, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto",
                           extent=(0.5, w2.shape[0] + 0.5, len(classes) - 0.5, -0.5))
    output.set(xlim=(0.5, w2.shape[0] + 0.5), yticks=range(len(classes)), yticklabels=classes, xlabel="Hidden unit", title="W2 output weights")
    output.tick_params(axis="y", labelsize=8)
    figure.colorbar(links, ax=output, label="W2 weight")
    for cell, key, title in [(grid[3, :2], "b1", "b1 hidden biases"), (grid[3, 2:], "b2", "b2 class biases")]:
        axis = figure.add_subplot(cell)
        axis.bar(np.arange(weights[key].size) + (key == "b1"), weights[key], color=TEAL)
        axis.axhline(0, color=NAVY, linewidth=0.7)
        axis.set(title=title, xlabel="Hidden unit" if key == "b1" else "Class index (0-9)", ylabel="Bias")
    save_plot(figure, path)
    pixels = w1[:, strongest[0]]
    strongest_weights = np.abs(pixels)
    center = strongest_weights.reshape(28, 28)[7:21, 7:21]
    mean = strongest_weights.mean()
    share = float((pixels > 0).mean())
    unit, category = np.unravel_index(np.abs(w2).argmax(), w2.shape)
    return (
        f"Unit {strongest[0] + 1} has the largest W1 column norm. Its mean absolute weight is "
        f"{center.mean():.4f} in the central 14 x 14 patch and {mean:.4f} across the whole filter. "
        f"{share:.1%} of its weights are positive. The largest absolute W2 connection links hidden unit "
        f"{unit + 1} to {escape(classes[category])} ({w2[unit, category]:+.3f}). Across all class biases, "
        f"values range from {weights['b2'].min():.3f} to {weights['b2'].max():.3f}."
    )


def plot_confusion(matrix, classes, path):
    figure, axis = plt.subplots(figsize=(8.4, 6.0), layout="constrained")
    plotted = axis.imshow(matrix, cmap="Blues")
    axis.set(xticks=range(len(classes)), xticklabels=classes, xlabel="Predicted class",
             yticks=range(len(classes)), yticklabels=classes, ylabel="True class")
    axis.tick_params(axis="x", rotation=34)
    for (row, column), count in np.ndenumerate(matrix):
        axis.text(column, row, str(count), ha="center", va="center",
                  fontsize=8, color="white" if count > matrix.max() * 0.55 else NAVY)
    figure.colorbar(plotted, ax=axis, label="Test images")
    save_plot(figure, path)
    errors = matrix.copy()
    np.fill_diagonal(errors, 0)
    if not errors.any():
        return "The matrix contains no off-diagonal entries."
    row, column = np.unravel_index(errors.argmax(), errors.shape)
    return (
        f"The largest off-diagonal count is {errors[row, column]:,}. "
        f"{errors[row, column]:,} of {matrix[row].sum():,} true {escape(classes[row])} images "
        f"were predicted as {escape(classes[column])}. The reverse error occurred "
        f"{errors[column, row]:,} times."
    )


def footer(canvas, document):
    canvas.setStrokeColor(colors.HexColor("#d7e0e9"))
    canvas.line(44, 41, A4[0] - 44, 41)
    canvas.setFillColor(colors.HexColor(GRAY))
    canvas.setFont("Helvetica", 9)
    canvas.drawString(44, 27, "Fashion-MNIST  /  NumPy experiment report")
    canvas.drawRightString(A4[0] - 44, 27, str(document.page))


def build_pdf(out, search, winner, test, correct, parameter_notes, confusion_notes, weights_url):
    doc = SimpleDocTemplate(str(out / "report.pdf"), pagesize=A4, leftMargin=44, rightMargin=44,
                            topMargin=42, bottomMargin=61, title="Fashion-MNIST with a three-layer NumPy network")
    title = ParagraphStyle("Title", fontName="Helvetica-Bold", fontSize=27, leading=31,
                           textColor=colors.HexColor(NAVY), spaceAfter=16)
    heading = ParagraphStyle("Heading", fontName="Helvetica-Bold", fontSize=15, leading=19,
                             textColor=colors.HexColor(NAVY), spaceBefore=12, spaceAfter=6)
    body = ParagraphStyle("Body", fontName="Helvetica", fontSize=10.5, leading=15,
                          textColor=colors.HexColor(NAVY), spaceAfter=7)
    caption = ParagraphStyle("Caption", parent=body, fontSize=9, leading=12, textColor=colors.HexColor(GRAY))
    best = winner["best"]
    rows = [["Trial", "Width", "Activation", "Initial LR", "L2", "Epoch", "Val acc.", "Val CE"]]
    for trial in search["trials"]:
        cfg, epoch = trial["configuration"], trial["best"]
        rows.append([trial["name"], str(cfg["hidden"]), cfg["activation"], f"{cfg['lr']:g}", f"{cfg['l2']:g}",
                     str(epoch["epoch"]), f"{epoch['val_accuracy']:.2%}", f"{epoch['val_ce']:.4f}"])
    table = Table(rows, colWidths=[88, 43, 66, 63, 53, 47, 64, 66], repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f1f5f8"), colors.white]),
        ("ALIGN", (3, 0), (-1, -1), "RIGHT"),
    ]))
    chart_width = doc.width - 45
    story = [
        Paragraph("Fashion-MNIST classification with NumPy", title),
        Paragraph(
            f"<b>Held-out test accuracy {test['accuracy']:.2%}</b> "
            f"({correct:,} of {test['samples']:,} images). "
            f"Test cross-entropy {test['cross_entropy']:.4f}.", body),
        Paragraph("Data and method", heading),
        Paragraph(
            f"Fashion-MNIST contains 28 x 28 grayscale images in ten apparel classes. Pixel values are "
            f"scaled by 255. The official training set is split by class into {search['train_size']:,} "
            f"training and {search['validation_size']:,} validation images with seed {search['seed']}. "
            f"The official {test['samples']:,}-image test set is evaluated after model selection.", body),
        Paragraph(
            "The model has 784 inputs, one configurable hidden layer and ten output logits. NumPy implements "
            "the forward pass and backpropagation. The objective is mean softmax cross-entropy plus "
            "L2/2 times the sum of squared W1 and W2 weights. The bias terms are excluded from the penalty. "
            f"Shuffled minibatch SGD uses batch size {search['batch']} and multiplies the learning rate by "
            f"{search['lr_decay']:g} after each epoch.", body),
        Paragraph("Validation search", heading),
        Paragraph(
            f"Every trial runs for {search['epochs']} epochs on the same split. Validation accuracy chooses "
            "each checkpoint and the winning trial, with lower validation cross-entropy breaking ties. "
            f"The winner is <b>{escape(winner['name'])}</b> at epoch {best['epoch']}, "
            f"with {best['val_accuracy']:.2%} validation accuracy and {best['val_ce']:.4f} validation cross-entropy.", body),
        table,
        Spacer(1, 15),
        Paragraph(
            f'<link href={quoteattr(CODE_URL)} color="{TEAL}"><u>Code and reproduction instructions on GitHub</u></link>'
            f' &nbsp;&nbsp; <link href={quoteattr(weights_url)} color="{TEAL}"><u>Download the selected NPZ checkpoint</u></link>', body),
        PageBreak(),
        Paragraph("Learning curves and parameters", title),
        Image(str(out / "learning-curves.png"), width=chart_width, height=chart_width * 0.35),
        Paragraph("Training and validation curves for the selected trial. Both loss curves show cross-entropy "
                  "without the L2 penalty. The dashed line marks the selected epoch.", caption),
        Image(str(out / "parameters.png"), width=chart_width, height=chart_width * 0.8),
        Paragraph("The W1 panels show the eight strongest input-weight vectors by L2 norm, or all units "
                  "when the hidden layer is smaller. Blue and red show opposite signs on a shared scale. "
                  "W2 shows the connections to the ten classes, followed by both bias vectors.", caption),
        Paragraph(parameter_notes, body),
        PageBreak(),
        Paragraph("Errors on the official test set", title),
        Image(str(out / "confusion.png"), width=doc.width, height=doc.width * 6 / 8.4),
        Paragraph("Each cell gives an image count. Rows are true classes and columns are predicted classes. "
                  "The diagonal contains the correct predictions.", caption),
        Paragraph(confusion_notes, body),
    ]
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Completed experiment directory")
    parser.add_argument("--out", type=Path, required=True, help="Report output directory")
    parser.add_argument("--weights-url", default=WEIGHTS_URL, help="Public download URL for this checkpoint")
    args = parser.parse_args()
    search = json.loads((args.run / "search.json").read_text())
    test = json.loads((args.run / "test.json").read_text())
    winner = next(trial for trial in search["trials"] if trial["name"] == search["selected"])
    with (args.run / f"{winner['name']}.csv").open(newline="") as stream:
        history = list(csv.DictReader(stream))
    weights, activation = load_checkpoint(args.run / "best.npz")
    if (activation, weights["b1"].size) != (winner["configuration"]["activation"], winner["configuration"]["hidden"]):
        raise ValueError("Checkpoint architecture disagrees with the selected trial")
    matrix = np.asarray(test["confusion"])
    correct = int(matrix.trace())
    if matrix.sum() != test["samples"] or not np.isclose(correct / test["samples"], test["accuracy"]):
        raise ValueError("Test confusion counts disagree with the reported accuracy or sample count")
    args.out.mkdir(parents=True, exist_ok=True)
    plot_curves(history, winner["best"]["epoch"], args.out / "learning-curves.png")
    parameter_notes = plot_parameters(weights, test["classes"], args.out / "parameters.png")
    confusion_notes = plot_confusion(matrix, test["classes"], args.out / "confusion.png")
    build_pdf(args.out, search, winner, test, correct, parameter_notes, confusion_notes, args.weights_url)


if __name__ == "__main__":
    main()
