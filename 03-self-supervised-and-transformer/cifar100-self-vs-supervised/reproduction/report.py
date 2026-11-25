"""Render the CIFAR-100 SimCLR comparison from completed trial records."""

import argparse
import csv
import json
import math
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Table, TableStyle

PRETRAIN = {"5k": (5000, 0.2), "20k": (20000, 0.2), "45k": (45000, 0.2), "20k_t05": (20000, 0.5)}
SOURCES = ("scratch", "imagenet", *(f"simclr_{key}" for key in PRETRAIN))
LABELS = dict(zip(SOURCES, ("Scratch", "ImageNet", "SimCLR 5k", "SimCLR 20k", "SimCLR 45k", "SimCLR 20k (T=0.5)")))
PALETTE = ("#4d6379", "#9c4a36", "#4777ae", "#16716c", "#855ba4", "#996716")
REPO = "https://github.com/fcxuu/computer-vision-projects"
CODE = f"{REPO}/tree/portfolio/reproductions-2026-11-23/03-self-supervised-and-transformer/cifar100-self-vs-supervised/reproduction"
RELEASE = f"{REPO}/releases/tag/portfolio-cifar100-simclr-2026-11-23"
INK = colors.HexColor("#183049")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def history(path):
    with path.open(newline="") as stream:
        rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    if not rows or not all(math.isfinite(value) for row in rows for value in row.values()):
        raise ValueError(f"Missing or non-finite history in {path}")
    return rows


def score(row):
    return row["val_accuracy"], -row["val_loss"]


def verified_test(test, source, config, best):
    if (test["source"], test["epoch"], test["size"]) != (source, best["epoch"], config["size"]):
        raise ValueError(f"Test metadata disagrees with the frozen {source} checkpoint")
    n, top1, top5 = (test[key] for key in ("samples", "correct", "top5_correct"))
    if any(type(value) is not int for value in (n, top1, top5)) or not (n == 10_000 and 0 <= top1 <= top5 <= n):
        raise ValueError(f"Invalid official test counts for {source}")
    for count, key in ((top1, "accuracy"), (top5, "top5")):
        if not math.isclose(count / n, test[key], rel_tol=0, abs_tol=1e-10):
            raise ValueError(f"Official test count and rate disagree for {source}: {key}")
    return {**test, "accuracy": top1 / n, "top5": top5 / n}


def load_runs(run):
    pretrain, classifiers = {}, {}
    for name, expected in PRETRAIN.items():
        folder = run / f"pretrain-{name}"
        config = read_json(folder / "run.json")
        rows = history(folder / "history.csv")
        recorded_steps = [row["step"] for row in rows]
        expected_steps = [*range(50, config["steps"], 50), config["steps"]]
        if (config["pretrain_samples"], config["temperature"]) != expected or config["variant"] != name:
            raise ValueError(f"Invalid pretraining configuration for {name}")
        if config["steps"] < 1 or recorded_steps != expected_steps:
            raise ValueError(f"Incomplete contrastive history for {name}")
        pretrain[name] = {"config": config, "history": rows}
    for source in SOURCES:
        folder = run / source
        config = read_json(folder / "train.json")
        rows = history(folder / "history.csv")
        if config["source"] != source or (config["n_train"], config["n_validation"]) != (45_000, 5000):
            raise ValueError(f"Invalid classifier split or metadata for {source}")
        if [row["epoch"] for row in rows] != list(range(1, config["epochs"] + 1)):
            raise ValueError(f"Incomplete classifier history for {source}")
        best = max(rows, key=score)
        if best != config["best"]:
            raise ValueError(f"Validation-best epoch disagrees with history for {source}")
        before, after = (config[f"frozen_fingerprint_{time}"] for time in ("before", "after"))
        if source == "scratch":
            if config["protocol"] != "end_to_end" or before is not None or after is not None:
                raise ValueError("Scratch trial must train the whole model")
        else:
            if config["protocol"] != "linear" or not isinstance(before, str) or not re.fullmatch("[0-9a-f]{64}", before) or before != after:
                raise ValueError(f"Frozen encoder fingerprint changed or is missing for {source}")
            if source.startswith("simclr_"):
                variant = source.removeprefix("simclr_")
                if before != pretrain[variant]["config"]["backbone_digest"]:
                    raise ValueError(f"Linear probe encoder disagrees with pretraining digest for {source}")
        classifiers[source] = {"config": config, "history": rows, "best": best,
                               "test": verified_test(read_json(folder / "test.json"), source, config, best)}
    for records, keys in (
        (pretrain, ("steps", "batch", "seed", "size", "lr", "optimizer", "weight_decay")),
        (classifiers, ("epochs", "batch", "seed", "size", "optimizer", "weight_decay")),
        ({name: classifiers[name] for name in SOURCES[1:]}, ("lr",)),
    ):
        signatures = {tuple(row["config"][key] for key in keys) for row in records.values()}
        if len(signatures) != 1:
            raise ValueError("Comparison requires a shared optimization and data budget within each stage")
    return pretrain, classifiers


def verify_indices(run):
    folders = [*(run / f"pretrain-{name}" for name in PRETRAIN), *(run / name for name in SOURCES)]
    split_paths = [folder / "split.npz" for folder in folders]
    subset_paths = {name: run / f"pretrain-{name}" / "pretrain-indices.npy" for name in PRETRAIN}
    all_paths = [*split_paths, *subset_paths.values()]
    if not any(path.is_file() for path in all_paths):
        return False
    if not all(path.is_file() for path in all_paths):
        raise ValueError("Incomplete saved split or subset indices")
    with np.load(split_paths[0], allow_pickle=False) as saved:
        train, validation = saved["train"], saved["validation"]
    if train.shape != (45_000,) or validation.shape != (5000,) or not np.array_equal(
        np.sort(np.concatenate((train, validation))), np.arange(50_000)
    ):
        raise ValueError("Expected a disjoint and exhaustive official training partition")
    for path in split_paths[1:]:
        with np.load(path, allow_pickle=False) as saved:
            if not np.array_equal(train, saved["train"]) or not np.array_equal(validation, saved["validation"]):
                raise ValueError(f"Training partitions differ: {path}")
    subsets = {}
    for name, path in subset_paths.items():
        subset = np.load(path, allow_pickle=False)
        if subset.shape != (PRETRAIN[name][0],) or len(np.unique(subset)) != len(subset) or not np.isin(subset, train).all():
            raise ValueError(f"Invalid pretraining subset: {path}")
        subsets[name] = subset
    if (not np.array_equal(subsets["20k"], subsets["20k_t05"])
            or not np.isin(subsets["5k"], subsets["20k"]).all()
            or not np.isin(subsets["20k"], subsets["45k"]).all()):
        raise ValueError("Expected nested scale subsets and an identical temperature-control subset")
    return True


def save_plot(fig, path):
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def plot_pretraining(records, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    for (name, record), color in zip(records.items(), PALETTE[2:]):
        rows = record["history"]
        for axis, key in zip(axes, ("ntxent", "pair_top1")):
            axis.plot([row["step"] for row in rows], [row[key] for row in rows],
                      color=color, label=LABELS[f"simclr_{name}"].removeprefix("SimCLR "))
    for axis, title in zip(axes, ("NT-Xent loss", "Paired-view retrieval top-1")):
        axis.set(title=title, xlabel="Optimizer step")
        axis.grid(alpha=0.18)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].legend(frameon=False, fontsize=8)
    save_plot(fig, out / "pretrain-curves.png")


def plot_classification(records, out):
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.7), layout="constrained")
    panels = (("train_loss", "Online train CE"), ("val_loss", "Validation CE"),
              ("val_accuracy", "Validation top-1"), ("train_accuracy", "Online train top-1"),
              ("val_top5", "Validation top-5"))
    for (source, record), color in zip(records.items(), PALETTE):
        rows, best = record["history"], record["best"]
        epochs = [row["epoch"] for row in rows]
        for axis, (key, title) in zip(axes.flat, panels):
            axis.plot(epochs, [row[key] for row in rows], color=color, label=LABELS[source])
            axis.scatter(best["epoch"], best[key], s=15, color=color, zorder=4)
            axis.set(title=title, xlabel="Epoch")
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.grid(alpha=0.18)
            if key.endswith(("accuracy", "top5")):
                axis.yaxis.set_major_formatter(PercentFormatter(1))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[1, 2].axis("off")
    axes[1, 2].legend(handles, labels, frameon=False, loc="center left", fontsize=9)
    save_plot(fig, out / "classification-curves.png")


def table(rows, widths):
    result = Table(rows, colWidths=widths, hAlign="LEFT")
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#edf4f6"), colors.white]),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return result


def footer(canvas, document):
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(INK)
    canvas.drawString(43, 27, "fcxuu / computer-vision-projects")
    canvas.drawRightString(A4[0] - 43, 27, str(document.page))


def build_pdf(pretrain, classifiers, split_verified, out):
    title = ParagraphStyle("Title", fontName="Helvetica-Bold", fontSize=24, leading=28, textColor=INK, spaceAfter=13)
    heading = ParagraphStyle("Heading", fontName="Helvetica-Bold", fontSize=14, leading=18, textColor=INK,
                             spaceBefore=10, spaceAfter=5)
    body = ParagraphStyle("Body", fontName="Helvetica", fontSize=10, leading=14, textColor=INK, spaceAfter=7)
    doc = SimpleDocTemplate(str(out / "report.pdf"), pagesize=A4, leftMargin=43, rightMargin=43,
                            topMargin=40, bottomMargin=53, title="SimCLR and supervised learning on CIFAR-100",
                            author="fcxuu / computer-vision-projects")
    config = classifiers["scratch"]["config"]
    pre = pretrain["5k"]["config"]
    pre_rows = [["Trial", "Images", "Temperature", "Steps", "Draws / image", "Last NT-Xent", "Last retrieval"]]
    for key, result in pretrain.items():
        settings, last = result["config"], result["history"][-1]
        n, steps = settings["pretrain_samples"], settings["steps"]
        pre_rows.append([LABELS[f"simclr_{key}"], f"{n:,}", f"{settings['temperature']:g}", f"{steps:,}",
                         f"{steps * settings['batch'] / n:.2f}", f"{last['ntxent']:.4f}", f"{last['pair_top1']:.2%}"])
    clf_rows = [["Run", "Best ep.", "Train top-1", "Val top-1", "Val top-5", "Test top-1", "Test top-5"]]
    for source, result in classifiers.items():
        best, test = result["best"], result["test"]
        clf_rows.append([LABELS[source], f"{best['epoch']:.0f}", f"{best['train_accuracy']:.2%}", f"{best['val_accuracy']:.2%}",
                         f"{best['val_top5']:.2%}", f"{test['accuracy']:.2%}", f"{test['top5']:.2%}"])
    scale = [classifiers[f"simclr_{key}"]["test"]["accuracy"] for key in ("5k", "20k", "45k")]
    temperature = classifiers["simclr_20k_t05"]["test"]["accuracy"] - scale[1]
    winner = max(SOURCES[2:], key=lambda source: score(classifiers[source]["best"]))
    test = classifiers[winner]["test"]
    image_diff = 100 * (test["accuracy"] - classifiers["imagenet"]["test"]["accuracy"])
    scratch = classifiers["scratch"]["test"]
    imagenet = classifiers["imagenet"]["test"]
    split_sentence = ("Saved indices verify that all splits match, all pretraining samples belong to the training "
                      "partition, the scale subsets are nested and the two temperature runs use identical images.")
    if not split_verified:
        split_sentence = "The comparison uses the split sizes recorded by the training runs."
    doc.build([
        Paragraph("SimCLR and supervised learning on CIFAR-100", title),
        Paragraph(f"The validation-selected SimCLR probe, <b>{LABELS[winner]}</b>, scored {test['accuracy']:.2%} official "
                  f"test top-1 ({test['correct']:,}/{test['samples']:,}), a {image_diff:+.2f}-point difference from "
                  "the frozen ImageNet probe.", body),
        Paragraph("Partitions and downstream protocol", heading),
        Paragraph(f"The official training split supplies {config['n_train']:,} training images and "
                  f"{config['n_validation']:,} validation images. The separate official test set contains "
                  f"{test['samples']:,} images. {split_sentence}", body),
        Paragraph("All classifiers use ResNet-18 and a 100-way linear head. ImageNet and SimCLR use frozen encoders "
                  "in evaluation mode and train the same linear head on features from the same training partition. "
                  "Encoder parameter and buffer fingerprints match before and after every probe. The random "
                  "initialization baseline trains the full network end to end with image augmentation.", body),
        Paragraph(f"Classification uses {config['optimizer']} (scratch LR {classifiers['scratch']['config']['lr']:g}, "
                  f"linear-probe LR {classifiers['imagenet']['config']['lr']:g}, weight decay "
                  f"{config['weight_decay']:g}), cosine decay, {config['epochs']} epochs, batch size {config['batch']}, "
                  f"{config['size']} x {config['size']} inputs and seed {config['seed']}. Highest validation top-1 "
                  "selects each checkpoint, with lower validation cross-entropy breaking ties.", body),
        table(clf_rows, [105, 52, 71, 70, 70, 70, 70]),
        Paragraph(f"Official top-1 / top-5 counts. Scratch {scratch['correct']:,} / {scratch['top5_correct']:,}, "
                  f"ImageNet {imagenet['correct']:,} / {imagenet['top5_correct']:,}, selected SimCLR "
                  f"{test['correct']:,} / {test['top5_correct']:,}. Each count uses the same {test['samples']:,}-image "
                  "denominator. Reported rates are checked against integer counts.", body),
        Paragraph(f'<link href="https://www.cs.toronto.edu/~kriz/cifar.html" color="#16716c"><u>Official CIFAR dataset</u></link> '
                  f'&nbsp; <link href="{CODE}" color="#16716c"><u>Code and instructions</u></link> '
                  f'&nbsp; <link href="{RELEASE}" color="#16716c"><u>Published weights and evidence</u></link>', body),
        PageBreak(),
        Paragraph("Contrastive pretraining and data scale", title),
        Paragraph("Each input image yields two independently augmented views. A ResNet-18 encoder and projection "
                  "MLP produce normalized embeddings. NT-Xent uses the paired view as each anchor's positive, "
                  "with the other images' views as negatives and cosine similarities scaled by temperature.", body),
        Paragraph(f"All runs use {pre['optimizer']} (initial LR {pre['lr']:g}, weight decay {pre['weight_decay']:g}), "
                  f"batch size {pre['batch']}, cosine decay and {pre['size']} x {pre['size']} inputs. "
                  "Each pair uses two views of a single sampled image.", body),
        table(pre_rows, [102, 49, 65, 52, 75, 69, 78]),
        Image(str(out / "pretrain-curves.png"), width=500, height=175),
        Paragraph("Curves show successive logging-window means. Retrieval top-1 measures whether each projected view "
                  "ranks its paired view first. Temperature changes the NT-Xent scaling, so downstream accuracy is "
                  "the shared measure for the temperature comparison.", body),
        Paragraph(f"At temperature 0.2, test top-1 for 5k, 20k and 45k was {scale[0]:.2%}, {scale[1]:.2%} and "
                  f"{scale[2]:.2%}, respectively. The 45k versus 5k difference was {100 * (scale[2] - scale[0]):+.2f} "
                  "points. The optimizer step count and batch size fix the total number of image draws. "
                  "A larger subset offers more distinct images and fewer draws per available image on average.", body),
        Paragraph(f"On the 20k subset, raising temperature from 0.2 to 0.5 changed test top-1 by {100 * temperature:+.2f} "
                  "points. These are measured differences for one seed and the recorded training budget.", body),
        PageBreak(),
        Paragraph("Classifier learning curves", title),
        Image(str(out / "classification-curves.png"), width=500, height=500 * 5.7 / 11),
        Paragraph("Dots mark validation-best checkpoints. Training CE and top-1 are online minibatch aggregates. "
                  "Validation metrics use unaugmented examples after each epoch. The training log does not "
                  "measure training top-5.", body),
    ], onFirstPage=footer, onLaterPages=footer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    pretrain, classifiers = load_runs(args.run)
    split_verified = verify_indices(args.run)
    args.out.mkdir(parents=True, exist_ok=True)
    plot_pretraining(pretrain, args.out)
    plot_classification(classifiers, args.out)
    build_pdf(pretrain, classifiers, split_verified, args.out)


if __name__ == "__main__":
    main()
