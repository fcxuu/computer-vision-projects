"""Render the recorded MNIST experiment and selected checkpoints to a PDF."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from dataset import load
from experiment import ROOT, score
from networks import Network

URL = (
    "https://github.com/fcxuu/computer-vision-projects/tree/portfolio/reproductions-2026-11-23"
    "/01-image-classification/nndl-mnist-mlp-cnn/reproduction"
)
INK = "#172d46"
ACCENT = "#146c68"


def save(figure, path):
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def plot_curves(run, names, output):
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.5), layout="constrained")
    for name in names:
        with (run / f"{name}.csv").open(newline="") as stream:
            history = list(csv.DictReader(stream))
        epochs = [int(row["epoch"]) for row in history]
        for axis, metric in zip(axes, ("train_ce", "val_ce", "val_accuracy")):
            axis.plot(epochs, [float(row[metric]) for row in history], label=name)
    for axis, title in zip(axes, ("Online train CE", "Clean validation CE", "Validation accuracy")):
        axis.set(title=title, xlabel="Epoch")
        axis.grid(alpha=0.2)
    axes[-1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[-1].legend(fontsize=7, loc="lower right", frameon=False)
    save(figure, output / "curves.png")


def plot_parameters(run, winners, output):
    figure, axes = plt.subplots(4, 8, figsize=(9, 4.9), layout="constrained")
    mlp = Network.load(run / f"{winners['mlp']}.npz")
    cnn = Network.load(run / f"{winners['cnn']}.npz")
    weights = mlp.layers[0].W
    indices = np.argsort(-np.linalg.norm(weights, axis=0), kind="stable")[:16]
    groups = [(weights[:, indices].T.reshape(-1, 28, 28), "Unit", indices + 1),
              (cnn.layers[0].W[:, 0], "Filter", np.arange(1, 17))]
    for panels, (images, label, indices) in zip((axes[:2], axes[2:]), groups):
        limit = float(np.abs(images).max())
        for panel, image, index in zip(panels.flat, images, indices):
            panel.imshow(image, cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
            panel.set_title(f"{label} {index}", fontsize=8)
            panel.axis("off")
    save(figure, output / "parameters.png")


def plot_test(run, best_cnn, test, root, output):
    images, targets = load(root, "t10k")
    model = Network.load(run / f"{best_cnn}.npz")
    predictions = np.concatenate([model(batch).argmax(axis=1) for batch in np.array_split(images, 80)])
    matrix = np.zeros((10, 10), dtype=np.int64)
    np.add.at(matrix, (targets, predictions), 1)
    np.testing.assert_array_equal(matrix, test["confusion"])
    figure, axis = plt.subplots(figsize=(5.5, 4.5), layout="constrained")
    colored = axis.imshow(matrix, cmap="Blues")
    axis.set(xticks=range(10), yticks=range(10), xlabel="Predicted digit", ylabel="True digit")
    for (row, col), count in np.ndenumerate(matrix):
        axis.text(col, row, str(count), ha="center", va="center", fontsize=7,
                  color="white" if count > matrix.max() * 0.5 else INK)
    figure.colorbar(colored, ax=axis, label="Images")
    save(figure, output / "confusion.png")
    figure, axes = plt.subplots(2, 8, figsize=(9, 2.6), layout="constrained")
    examples = np.flatnonzero(predictions != targets)[:16]
    for axis in axes.flat:
        axis.axis("off")
    for axis, index in zip(axes.flat, examples):
        axis.imshow(images[index].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        axis.set_title(f"True {targets[index]} / pred {predictions[index]}", fontsize=8)
    save(figure, output / "errors.png")
    np.fill_diagonal(matrix, 0)
    source, predicted = np.unravel_index(matrix.argmax(), matrix.shape)
    return source, predicted, matrix[source, predicted]


def page_footer(canvas, document):
    canvas.setFillColor(colors.HexColor(INK))
    canvas.setFont("Helvetica", 9)
    canvas.drawString(43, 28, "MNIST / NumPy reproduction")
    canvas.drawRightString(A4[0] - 43, 28, str(document.page))


def build_report(output, train, test, winners, error_pair):
    styles = getSampleStyleSheet()
    for style in styles.byName.values():
        style.textColor = colors.HexColor(INK)
    styles["BodyText"].spaceAfter = 7
    styles["BodyText"].leading = 14
    styles["Title"].fontSize = 23
    styles["Title"].leading = 28
    styles["Title"].alignment = 0
    title = lambda text: Paragraph(text, styles["Title"])
    p = lambda text: Paragraph(text, styles["BodyText"])
    rows = [["Trial", "Epochs", "Best epoch", "Val accuracy", "Test accuracy", "Test CE"]]
    trials = train["trials"]
    for name, trial in trials.items():
        rows.append([name, trial["epochs_run"], trial["best"]["epoch"], f"{trial['best']['val_accuracy']:.2%}",
                     f"{test[name]['accuracy']:.2%}", f"{test[name]['cross_entropy']:.4f}"])
    table = Table(rows, colWidths=[106, 48, 61, 93, 98, 78], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(INK)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#edf4f6"), colors.white]),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    mlp, cnn = winners["mlp"], winners["cnn"]
    zoom = "0.9 to 1.1"
    mlp_width = trials[mlp].get("mlp_width", 600)
    activation = trials[mlp]["activation"]
    summary = " ".join(
        f"{name.upper()} {test[trial]['correct']:,}/{test[trial]['samples']:,} "
        f"({test[trial]['accuracy']:.2%})."
        for name, trial in winners.items()
    )
    source, predicted, count = error_pair
    def delta(with_aug, plain):
        return 100 * (test[with_aug]["accuracy"] - test[plain]["accuracy"])

    story = [
        title("Hand-written MLP and CNN on MNIST"),
        p(f"<b>Held-out test accuracy.</b> {summary}"),
        p(f"<b>Protocol.</b> Stratified fixed split of {train['train_samples']:,} training images "
          f"and {train['validation_samples']:,} validation images from the official training set. "
          f"Seed {train['seed']}, batch size {train['batch']}, up to {train['epochs']} epochs per trial. "
          f"The official test set was evaluated after every validation checkpoint was frozen."),
        p(f"<b>Models.</b> MLP: 784 to {mlp_width} to 10. CNN: two 5 x 5 convolutions with "
          "16 and 32 outputs, stride two and padding two, then flattening and a 100-unit hidden layer. "
          f"Hidden activation: {activation}. Optimization uses softmax cross-entropy. NumPy handles "
          "forward and backward passes. SciPy performs "
          "bilinear image resampling only. L2 regularizes the weights."),
        p(f"<b>Augmentation.</b> With probability 0.5 per image during training, apply a rotation "
          f"up to 15 degrees, horizontal/vertical translation up to two pixels and scale {zoom}. "
          f"Augmentation and sampling have separate deterministic random streams."),
        p("<b>Optimization.</b> The default MLP uses momentum 0.9, initial LR 0.04, L2=1e-4 and "
          "exponential decay. The CNN uses momentum 0.9, LR 0.02, L2=1e-5 and two LR drops. "
          "The CSVs record the actual learning-rate schedules. Validation accuracy chooses each checkpoint, with cross-entropy breaking ties."),
        table, Spacer(1, 12),
        p(f"With the optimizer settings held fixed, affine augmentation changed test accuracy by "
          f"{delta('mlp_affine','mlp_plain'):+.2f} percentage points for the MLP and "
          f"{delta('cnn_affine','cnn_plain'):+.2f} for the CNN. The comparison uses a single seed."),
        p(f'<link href="{URL}" color="{ACCENT}"><u>GitHub code, checkpoints and raw metrics</u></link>'),
        PageBreak(),
        title("Training diagnostics and weights"),
        Image(str(output / "curves.png"), width=493, height=157),
        p("The training curves aggregate batches before each update. Augmented trials include "
          "transformed training images. Validation uses the original images and an end-of-epoch checkpoint."),
        Image(str(output / "parameters.png"), width=480, height=261),
        p(f"Top rows: the 16 highest-norm input vectors from {mlp}, each reshaped to 28 x 28. "
          f"Bottom rows: the first convolutional filters from {cnn}. The two groups use separate "
          "symmetric color ranges. Red and blue mark positive and negative coefficients."),
        p("The MLP learns position-specific patterns over the full grid. A CNN kernel instead "
          "assigns signed weights to a 5 x 5 neighborhood reused across spatial locations."),
        PageBreak(),
        title("Test errors of the selected CNN"),
        Image(str(output / "confusion.png"), width=410, height=335),
        p(f"The most frequent off-diagonal pair is true {source} predicted as {predicted} ({count} images)."),
        Image(str(output / "errors.png"), width=480, height=139),
        p("First sixteen errors by official test index. Each label gives the target and the model prediction."),
    ]
    doc = SimpleDocTemplate(str(output / "report.pdf"), pagesize=A4,
                            leftMargin=44, rightMargin=44, topMargin=40, bottomMargin=52,
                            title="MNIST MLP and CNN from scratch")
    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "mnist")
    args = parser.parse_args()
    train = json.loads((args.run / "train.json").read_text())
    trials = train["trials"]
    test = json.loads((args.run / "test.json").read_text())
    if trials.keys() != test.keys() or not {"mlp_plain", "mlp_affine", "cnn_plain", "cnn_affine"} <= trials.keys():
        raise ValueError("Report requires tested matched MLP/CNN ablations")
    winners = {
        name: max((key for key, trial in trials.items() if trial["config"]["network"] == name),
                  key=lambda key: score(trials[key]["best"]))
        for name in ("mlp", "cnn")
    }
    args.out.mkdir(parents=True, exist_ok=True)
    plot_curves(args.run, trials, args.out)
    plot_parameters(args.run, winners, args.out)
    errors = plot_test(args.run, winners["cnn"], test[winners["cnn"]], args.data, args.out)
    build_report(args.out, train, test, winners, errors)


if __name__ == "__main__":
    main()
