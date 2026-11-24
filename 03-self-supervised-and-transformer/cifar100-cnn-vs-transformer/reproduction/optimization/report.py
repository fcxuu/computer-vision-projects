"""Report matched training budgets for the CIFAR-100 CNN and ViT."""

import argparse
import csv
import importlib.util
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate

SPEC = importlib.util.spec_from_file_location("architecture_baseline_report", Path(__file__).resolve().parents[1] / "report.py")
FIRST = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIRST)

MODELS = {"cnn": "Residual CNN", "vit": "Vision Transformer"}
BUDGETS = (20, 40)
SETTING = "adamw_1e-3"
TARGET = 0.52
RELEASE = f"{FIRST.REPO}/releases/download/portfolio-cifar100-arch-extended-2026-11-24"
PROTOCOL = ("batch", "size", "seed", "optimizer", "initial_lr", "weight_decay",
            "warmup_epochs", "schedule", "clip_norm", "cutmix_alpha", "cutmix_probability")
TRACE = ("lr", "cutmix_batches", "mean_replaced_area")


def expected_rate(config, epoch):
    warmup, epochs, lr = (config[key] for key in ("warmup_epochs", "epochs", "initial_lr"))
    if epoch <= warmup:
        return lr * epoch / warmup
    progress = (epoch - warmup) / (epochs - warmup)
    return lr * (0.02 + 0.98 * (1 + math.cos(math.pi * progress)) / 2)


def read_trial(path, variant, budget):
    config = FIRST.read_json(path / "train.json")
    if (config["variant"], config["setting"], config["epochs"]) != (variant, SETTING, budget):
        raise ValueError(f"Unexpected architecture, setting or budget in {path}")
    if (config["n_train"], config["n_val"], config["size"]) != (45_000, 5000, 32):
        raise ValueError(f"Unexpected image partition or size in {path}")
    if ((config["initial_lr"], config["weight_decay"]) != FIRST.SETTINGS[SETTING]
            or config["optimizer"] != "AdamW" or config["schedule"] != "cosine"
            or not 0 < config["warmup_epochs"] < budget or config["batch"] < 1):
        raise ValueError(f"Unexpected optimization configuration in {path}")
    with np.load(path / "split.npz", allow_pickle=False) as archive:
        split = archive["train"], archive["validation"]
    if (tuple(len(ids) for ids in split) != (45_000, 5000)
            or any(ids.ndim != 1 or ids.dtype.kind not in "iu" for ids in split)
            or not np.array_equal(np.sort(np.concatenate(split)), np.arange(50_000))):
        raise ValueError(f"Split is not disjoint and exhaustive in {path}")
    with (path / "history.csv").open(newline="") as stream:
        rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    if [row["epoch"] for row in rows] != list(range(1, budget + 1)):
        raise ValueError(f"Incomplete or unordered history in {path}")
    batches = math.ceil(config["n_train"] / config["batch"])
    for row in rows:
        if not all(math.isfinite(value) for value in row.values()):
            raise ValueError(f"Non-finite history in {path}")
        if not math.isclose(row["lr"], expected_rate(config, row["epoch"]), abs_tol=1e-13, rel_tol=1e-10):
            raise ValueError(f"Learning-rate trace contradicts cosine protocol in {path}")
        count, area = row["cutmix_batches"], row["mean_replaced_area"]
        if not (count.is_integer() and 0 <= count <= batches and 0 <= area <= count / batches + 1e-12):
            raise ValueError(f"Impossible CutMix count or area in {path}")
    best = max(rows, key=FIRST.score)
    if best != config["best"]:
        raise ValueError(f"Validation selection disagrees with epoch history in {path}")
    test = FIRST.verified_test(FIRST.read_json(path / "test.json"), config)
    return {"config": config, "split": split, "history": rows, "best": best, "test": test,
            "updates": batches * budget, "mixed": sum(int(row["cutmix_batches"]) for row in rows),
            "area": math.fsum(row["mean_replaced_area"] for row in rows) / budget}


def read_comparison(baseline, run):
    trials = {(variant, budget): read_trial(root / f"{variant}-{SETTING}", variant, budget)
              for root, budget in zip((baseline, run), BUDGETS) for variant in MODELS}
    reference = trials["cnn", 20]
    for trial in trials.values():
        if (any(trial["config"][key] != reference["config"][key] for key in PROTOCOL)
                or any(not np.array_equal(a, b) for a, b in zip(trial["split"], reference["split"]))):
            raise ValueError("The four runs must share the same partition and optimization protocol")
    for budget in BUDGETS:
        cnn, vit = (trials[model, budget]["history"] for model in MODELS)
        if any(one[key] != other[key] for one, other in zip(cnn, vit) for key in TRACE):
            raise ValueError(f"LR or CutMix trajectory differs between architectures at {budget} epochs")
    counts = {}
    for variant in MODELS:
        before, after = (trials[variant, budget]["config"]["parameters"] for budget in BUDGETS)
        if type(before) is not int or before < 1 or before != after:
            raise ValueError(f"Invalid or inconsistent parameter count for {variant}")
        counts[variant] = before
    gap = (max(counts.values()) - min(counts.values())) / min(counts.values())
    if any(not math.isclose(gap, trial["config"]["parameter_gap"], abs_tol=1e-12, rel_tol=0)
           for trial in trials.values()):
        raise ValueError("Parameter gap disagrees with the recorded counts")
    return trials, counts, gap


def plot(trials, out):
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.1), layout="constrained")
    panels = (("train_loss", "Online train CE"), ("val_loss", "Validation CE"),
              ("val_accuracy", "Validation top-1"), ("lr", "Learning rate"))
    for (variant, budget), trial in trials.items():
        rows, best = trial["history"], trial["best"]
        for axis, (field, title) in zip(axes.flat, panels):
            axis.plot([row["epoch"] for row in rows], [row[field] for row in rows],
                      label=f"{MODELS[variant]} / {budget} epochs", color=FIRST.COLORS[variant],
                      linestyle="--" if budget == 20 else "-")
            if field == "val_accuracy":
                axis.scatter(best["epoch"], best[field], color=FIRST.COLORS[variant], s=18)
            axis.set(title=title, xlabel="Epoch")
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.grid(alpha=0.2)
    axes[1, 0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0, 1].legend(frameon=False, fontsize=8)
    fig.savefig(out / "budget-curves.png", dpi=175, facecolor="white")
    plt.close(fig)


def delta(left, right):
    return [f"{100 * (right[key] - left[key]):+.2f}" for key in ("accuracy", "top5")]


def report(trials, counts, gap, out):
    doc = SimpleDocTemplate(str(out / "report.pdf"), pagesize=A4, leftMargin=43, rightMargin=43,
                            topMargin=40, bottomMargin=53, title="Extending CNN and ViT training on CIFAR-100",
                            author="fcxuu / computer-vision-projects")
    title = ParagraphStyle("Title", fontName="Helvetica-Bold", fontSize=24, leading=28, textColor=FIRST.INK, spaceAfter=13)
    head = ParagraphStyle("Heading", fontName="Helvetica-Bold", fontSize=14, leading=18,
                          textColor=FIRST.INK, spaceBefore=10, spaceAfter=5)
    body = ParagraphStyle("Body", fontName="Helvetica", fontSize=10, leading=14, textColor=FIRST.INK, spaceAfter=7)
    common = trials["cnn", 20]["config"]
    vit_test = trials["vit", 40]["test"]
    margin = 100 * (vit_test["accuracy"] - TARGET)
    goal = (f"{abs(margin):.2f} percentage points {'above' if margin > 0 else 'below'} the {TARGET:.0%} project target"
            if margin else f"meeting the {TARGET:.0%} project target")
    results = [["Architecture", "Epochs", "Best ep.", "Val top-1", "Val top-5", "Test top-1", "Test top-5"]]
    for (variant, budget), trial in trials.items():
        best, test = trial["best"], trial["test"]
        results.append([MODELS[variant], budget, int(best["epoch"]), f"{best['val_accuracy']:.2%}", f"{best['val_top5']:.2%}",
                        f"{test['accuracy']:.2%} ({test['correct']:,})", f"{test['top5']:.2%} ({test['top5_correct']:,})"])
    differences = [["Official test change", "Top-1 (pp)", "Top-5 (pp)"]]
    for variant in MODELS:
        differences.append([f"{MODELS[variant]}, 40 minus 20 epochs",
                            *delta(trials[variant, 20]["test"], trials[variant, 40]["test"])])
    for budget in BUDGETS:
        differences.append([f"ViT minus CNN at {budget} epochs", *delta(trials["cnn", budget]["test"], trials["vit", budget]["test"])])
    exposures = [["Epochs", "Optimizer updates", "CutMix updates", "CutMix share", "Mean replaced area"]]
    for budget in BUDGETS:
        trial = trials["cnn", budget]
        exposures.append([budget, f"{trial['updates']:,}", f"{trial['mixed']:,}",
                          f"{trial['mixed'] / trial['updates']:.2%}", f"{trial['area']:.2%}"])
    doc.build([
        Paragraph("Extending CNN and ViT training on CIFAR-100", title),
        Paragraph(f"The 40-epoch Vision Transformer scored <b>{vit_test['accuracy']:.2%}</b> on official test top-1, {goal}.", body),
        Paragraph("Matched architectures and optimization", head),
        Paragraph(f"The residual CNN has {counts['cnn']:,} parameters and the Vision Transformer has {counts['vit']:,}, "
                  f"a {gap:.2%} difference relative to the smaller model. The four runs share identical saved "
                  f"{common['n_train']:,}-image training and {common['n_val']:,}-image validation partitions. "
                  f"The official test set contains {vit_test['samples']:,} images.", body),
        Paragraph(f"Runs use AdamW at initial LR {common['initial_lr']:g} and weight decay {common['weight_decay']:g}, "
                  f"batch size {common['batch']}, seed {common['seed']} and {common['warmup_epochs']} linear warmup epochs. "
                  "The 40-epoch run stretches the cosine decay over a longer interval. Both budgets start from seeded initialization.", body),
        Paragraph("Validation selection and official test", head),
        FIRST.table(results, [103, 45, 50, 58, 58, 88, 88]),
        Paragraph("Highest validation top-1 selects each checkpoint, with lower validation CE breaking ties. "
                  "Parentheses give the test counts for those selected checkpoints.", body),
        FIRST.table(differences, [250, 115, 115]),
        Paragraph("These comparisons cover one seed and one matched optimizer setting.", body),
        Paragraph(f'<link href="{FIRST.CODE}/evidence/report.pdf" color="#176f69"><u>First-round report</u></link> &nbsp; '
                  f'<link href="{FIRST.CODE}/optimization" color="#176f69"><u>Optimization code</u></link> &nbsp; '
                  f'<link href="{RELEASE}/cnn-40ep.pt" color="#176f69"><u>CNN weights</u></link> &nbsp; '
                  f'<link href="{RELEASE}/vit-40ep.pt" color="#176f69"><u>ViT weights</u></link>', body),
        PageBreak(),
        Paragraph("Learning curves and CutMix exposure", title),
        Image(str(out / "budget-curves.png"), width=493, height=493 * 6.1 / 10.4),
        Paragraph("Dots mark the validation-selected epochs. Online train cross-entropy uses the actual "
                  "replaced area to weight CutMix labels. Validation cross-entropy uses original labels.", body),
        Paragraph(f"CutMix uses probability {common['cutmix_probability']:.0%} per eligible batch with "
                  f"Beta shape parameters {common['cutmix_alpha']:g}. Learning rates, CutMix batch counts and "
                  "replaced-area traces match across architectures within each budget. Mean area includes unmixed updates as zero.", body),
        FIRST.table(exposures, [65, 106, 101, 96, 117]),
    ], onFirstPage=FIRST.footer, onLaterPages=FIRST.footer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    trials, counts, gap = read_comparison(args.baseline, args.run)
    args.out.mkdir(parents=True, exist_ok=True)
    plot(trials, args.out)
    report(trials, counts, gap, args.out)


if __name__ == "__main__":
    main()
