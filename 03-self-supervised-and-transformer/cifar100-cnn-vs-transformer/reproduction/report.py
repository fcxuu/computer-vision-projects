"""Render the parameter-matched CIFAR-100 architecture comparison."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Table, TableStyle

MODELS = {"cnn": "Residual CNN", "vit": "Vision Transformer"}
SETTINGS = {"adamw_1e-3": (0.001, 0.03), "adamw_5e-4": (0.0005, 0.01)}
COLORS = {"cnn": "#176f69", "vit": "#9c4935"}
REPO = "https://github.com/fcxuu/computer-vision-projects"
CODE = f"{REPO}/tree/portfolio/reproductions-2026-11-23/03-self-supervised-and-transformer/cifar100-cnn-vs-transformer/reproduction"
RELEASE = f"{REPO}/releases/tag/portfolio-cifar100-architectures-2026-11-23"
INK = colors.HexColor("#183049")


def score(row):
    return row["val_accuracy"], -row["val_loss"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verified_test(test, config):
    expected = (config["variant"], config["setting"], config["best"]["epoch"])
    if tuple(test[key] for key in ("variant", "setting", "epoch")) != expected:
        raise ValueError(f"Test record is not for the frozen validation checkpoint {expected}")
    n, top1, top5 = (test[key] for key in ("samples", "correct", "top5_correct"))
    if any(type(value) is not int for value in (n, top1, top5)) or not (n == 10_000 and 0 <= top1 <= top5 <= n):
        raise ValueError(f"Invalid official test counts for {expected}")
    for count, key in ((top1, "accuracy"), (top5, "top5")):
        if not math.isclose(count / n, test[key], abs_tol=1e-10, rel_tol=0):
            raise ValueError(f"Test counts disagree with {key} for {expected}")
    return {**test, "accuracy": top1 / n, "top5": top5 / n}


def load_runs(run):
    records = {}
    references = None
    for setting, hyperparameters in SETTINGS.items():
        for variant in MODELS:
            path = run / f"{variant}-{setting}"
            config = read_json(path / "train.json")
            if (config["variant"], config["setting"]) != (variant, setting):
                raise ValueError(f"Incorrect metadata identity for {path}")
            if (config["initial_lr"], config["weight_decay"]) != hyperparameters:
                raise ValueError(f"Unexpected LR/weight-decay combination for {path}")
            if (config["n_train"], config["n_val"], config["size"]) != (45_000, 5000, 32):
                raise ValueError(f"Unexpected dataset partition or input size in {path}")
            with np.load(path / "split.npz", allow_pickle=False) as split:
                indices = split["train"], split["validation"]
            train, val = indices
            if (train.shape != (45_000,) or val.shape != (5000,)
                    or any(index.dtype.kind not in "iu" for index in indices)
                    or not np.array_equal(np.sort(np.concatenate(indices)), np.arange(50_000))):
                raise ValueError(f"Split is not disjoint and exhaustive in {path}")
            if references is None:
                references = indices
            elif any(not np.array_equal(left, right) for left, right in zip(references, indices)):
                raise ValueError(f"Saved partitions differ in {path}")
            with (path / "history.csv").open(newline="") as stream:
                history = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
            if not history or [row["epoch"] for row in history] != list(range(1, config["epochs"] + 1)):
                raise ValueError(f"Incomplete epoch history in {path}")
            if not all(math.isfinite(value) for row in history for value in row.values()):
                raise ValueError(f"Non-finite training history in {path}")
            if max(history, key=score) != config["best"]:
                raise ValueError(f"Validation-best checkpoint disagrees with history in {path}")
            per_epoch = math.ceil(config["n_train"] / config["batch"])
            for row in history:
                mixed, area = row["cutmix_batches"], row["mean_replaced_area"]
                if not (mixed.is_integer() and 0 <= mixed <= per_epoch and 0 <= area <= mixed / per_epoch + 1e-12):
                    raise ValueError(f"Invalid recorded CutMix statistics in {path}")
            records[variant, setting] = {
                "config": config, "history": history, "test": verified_test(read_json(path / "test.json"), config),
                "updates": per_epoch * config["epochs"],
                "mixed": sum(int(row["cutmix_batches"]) for row in history),
                "mean_area": sum(row["mean_replaced_area"] for row in history) / len(history),
            }
    shared = ("batch", "epochs", "seed", "optimizer", "warmup_epochs", "schedule", "clip_norm", "cutmix_alpha",
              "cutmix_probability", "device")
    if len({tuple(record["config"][key] for key in shared) for record in records.values()}) != 1:
        raise ValueError("The four runs must use the same budget and optimization protocol outside LR and decay")
    for setting in SETTINGS:
        traces = [[row["lr"] for row in records[variant, setting]["history"]] for variant in MODELS]
        if traces[0] != traces[1]:
            raise ValueError(f"Learning-rate traces differ between architectures for {setting}")
        mixing = [[(row["cutmix_batches"], row["mean_replaced_area"])
                   for row in records[variant, setting]["history"]] for variant in MODELS]
        if mixing[0] != mixing[1]:
            raise ValueError(f"CutMix exposure differs between architectures for {setting}")
    params = {}
    for variant in MODELS:
        candidates = {records[variant, setting]["config"]["parameters"] for setting in SETTINGS}
        if len(candidates) != 1 or min(candidates) <= 0:
            raise ValueError(f"Parameter counts differ for {variant}")
        params[variant] = candidates.pop()
    gap = (max(params.values()) - min(params.values())) / min(params.values())
    if any(not math.isclose(gap, row["config"]["parameter_gap"], abs_tol=1e-10, rel_tol=0)
           for row in records.values()):
        raise ValueError("Recorded relative parameter gap disagrees with counts")
    return records, params, gap


def plot_curves(records, out):
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.7), layout="constrained")
    metrics = (("train_loss", "Online train CE"), ("val_loss", "Validation CE"),
               ("val_accuracy", "Validation top-1"), ("train_weighted_accuracy", "Online train top-1 (weighted)"),
               ("val_top5", "Validation top-5"))
    for (variant, setting), result in records.items():
        rows, best = result["history"], result["config"]["best"]
        epochs = [row["epoch"] for row in rows]
        color = COLORS[variant]
        line = "-" if setting == next(iter(SETTINGS)) else "--"
        rate, wd = SETTINGS[setting]
        for axis, (metric, title) in zip(axes.flat, metrics):
            axis.plot(epochs, [row[metric] for row in rows], color=color, linestyle=line,
                      label=f"{MODELS[variant]} / {rate:g}, {wd:g}")
            axis.scatter(best["epoch"], best[metric], s=18, color=color, zorder=4)
            axis.set(title=title, xlabel="Epoch")
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.grid(alpha=0.18)
            if metric.endswith(("accuracy", "top5")):
                axis.yaxis.set_major_formatter(PercentFormatter(1))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[1, 2].axis("off")
    axes[1, 2].legend(handles, labels, title="Architecture / initial LR, weight decay",
                      loc="center", frameon=False, fontsize=8, title_fontsize=8)
    fig.savefig(out / "curves.png", dpi=180, facecolor="white")
    plt.close(fig)


def table(rows, widths):
    result = Table(rows, colWidths=widths, hAlign="LEFT")
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#edf4f5"), colors.white]),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return result


def footer(canvas, doc):
    canvas.setFillColor(INK)
    canvas.setFont("Helvetica", 9)
    canvas.drawString(43, 27, "fcxuu / computer-vision-projects")
    canvas.drawRightString(A4[0] - 43, 27, str(doc.page))


def build_pdf(records, params, gap, out):
    title = ParagraphStyle("Title", fontName="Helvetica-Bold", fontSize=24, leading=28, textColor=INK, spaceAfter=13)
    head = ParagraphStyle("Heading", fontName="Helvetica-Bold", fontSize=14, leading=18, textColor=INK,
                          spaceBefore=10, spaceAfter=5)
    body = ParagraphStyle("Body", fontName="Helvetica", fontSize=10, leading=14, textColor=INK, spaceAfter=7)
    first = next(iter(records.values()))
    common = first["config"]
    winners = {variant: max(SETTINGS, key=lambda setting: score(records[variant, setting]["config"]["best"]))
               for variant in MODELS}
    cnn, vit = (records[name, winners[name]] for name in MODELS)
    delta = 100 * (vit["test"]["accuracy"] - cnn["test"]["accuracy"])
    rows = [["Architecture", "Initial LR", "WD", "Best ep.", "Val top-1", "Val top-5", "Test top-1", "Test top-5"]]
    cutmix_rows = [["Architecture", "Initial LR", "CutMix updates", "Share", "Area / update", "Area / mix"]]
    for (variant, setting), data in records.items():
        config, test = data["config"], data["test"]
        best = config["best"]
        lr = f"{config['initial_lr']:g}"
        rows.append([MODELS[variant], lr, f"{config['weight_decay']:g}", best["epoch"],
                     f"{best['val_accuracy']:.2%}", f"{best['val_top5']:.2%}", f"{test['accuracy']:.2%}", f"{test['top5']:.2%}"])
        share = data["mixed"] / data["updates"]
        conditional = f"{data['mean_area'] / share:.2%}" if share else "None"
        cutmix_rows.append([MODELS[variant], lr, f"{data['mixed']:,}", f"{share:.2%}",
                            f"{data['mean_area']:.2%}", conditional])
    paired = []
    for setting, (lr, decay) in SETTINGS.items():
        one, two = (records[variant, setting] for variant in MODELS)
        val_difference = 100 * (two["config"]["best"]["val_accuracy"] - one["config"]["best"]["val_accuracy"])
        test_difference = 100 * (two["test"]["accuracy"] - one["test"]["accuracy"])
        paired.append(f"LR {lr:g} and WD {decay:g} gave a ViT minus CNN difference of {val_difference:+.2f} "
                      f"points in validation top-1 and {test_difference:+.2f} points in test top-1.")
    opt = common["optimizer"]
    doc = SimpleDocTemplate(str(out / "report.pdf"), pagesize=A4, leftMargin=43, rightMargin=43,
                            topMargin=40, bottomMargin=53, title="CNN versus Vision Transformer on CIFAR-100",
                            author="fcxuu / computer-vision-projects")
    doc.build([
        Paragraph("CNN versus Vision Transformer on CIFAR-100", title),
        Paragraph(f"After validation selected one setting per architecture, the Vision Transformer scored "
                  f"{delta:+.2f} percentage points in official test top-1 relative to the residual CNN.", body),
        Paragraph("Architectures and common budget", head),
        Paragraph(f"The residual CNN has {params['cnn']:,} trainable parameters. A 48-channel stem feeds four "
                  "residual blocks of widths 64, 96, 128 and 144, then global average pooling and the classifier.", body),
        Paragraph(f"The Vision Transformer has {params['vit']:,} trainable parameters. It projects 4 x 4 patches "
                  "into 64 tokens of width 160, appends a learned class token, and uses learned positions, "
                  "four encoder blocks with four attention heads each, and a class-token classifier. "
                  f"The parameter difference is {gap:.2%} of the smaller model.", body),
        Paragraph(f"Identical saved splits use {common['n_train']:,} training and {common['n_val']:,} "
                  f"validation images from the official training partition. The official {vit['test']['samples']:,}-image "
                  f"test split evaluates frozen checkpoints. All trials run {common['epochs']} epochs, batch size "
                  f"{common['batch']}, and {first['updates']:,} optimizer updates with seed {common['seed']} "
                  f"on {common['device']}. The shared protocol uses {opt}, cosine decay with "
                  f"{common['warmup_epochs']} warmup epochs, and gradient clipping at norm {common['clip_norm']:g}.", body),
        Paragraph("Matched settings and frozen test results", head),
        table(rows, [111, 58, 43, 50, 58, 58, 59, 59]),
        Paragraph("Each run selects the checkpoint with highest validation top-1 and uses lower validation "
                  "cross-entropy to break ties. The same ranking selects a setting within each architecture.", body),
        Paragraph(" ".join(paired), body),
        Paragraph(
            f"The selected CNN ({winners['cnn']}) gets {cnn['test']['correct']:,} top-1 and "
            f"{cnn['test']['top5_correct']:,} top-5 hits. The selected ViT ({winners['vit']}) gets "
            f"{vit['test']['correct']:,} top-1 and {vit['test']['top5_correct']:,} top-5 hits. "
            "The run uses one seed, and the tested settings change LR and weight decay together.", body),
        Paragraph(f'<link href="https://www.cs.toronto.edu/~kriz/cifar.html" color="#176f69"><u>Official CIFAR dataset</u></link> '
                  f'&nbsp; <link href="{CODE}" color="#176f69"><u>Code and instructions</u></link> '
                  f'&nbsp; <link href="{RELEASE}" color="#176f69"><u>Weights release</u></link>', body),
        PageBreak(),
        Paragraph("Learning curves and CutMix exposure", title),
        Image(str(out / "curves.png"), width=500, height=500 * 5.7 / 11),
        Paragraph("Dots mark the best validation epoch in each run. Online training cross-entropy mixes the "
                  "two label losses using actual rectangle area. Training top-1 uses those same label weights. "
                  "Validation cross-entropy uses original labels and unaugmented evaluation inputs.", body),
        Paragraph(f"CutMix runs with probability {common['cutmix_probability']:.0%} per eligible batch and Beta "
                  f"shape {common['cutmix_alpha']:g} for both parameters. The rectangle is clipped to the "
                  "image boundary. The donor-label weight is the replaced area fraction, and the original label "
                  "receives the remaining mass. The table reports logged counts and area means across all "
                  "updates and among updates with CutMix.", body),
        table(cutmix_rows, [111, 58, 99, 63, 82, 82]),
        Paragraph("The same epoch series are available in TensorBoard. Set the log directory to the suite root "
                  "with <font name='Courier'>tensorboard --logdir &quot;$RUN&quot;</font>. Each trial stores events "
                  "under its <font name='Courier'>tensorboard/</font> subdirectory.", body),
    ], onFirstPage=footer, onLaterPages=footer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records, parameters, gap = load_runs(args.run)
    args.out.mkdir(parents=True, exist_ok=True)
    plot_curves(records, args.out)
    build_pdf(records, parameters, gap, args.out)


if __name__ == "__main__":
    main()
