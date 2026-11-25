"""Report paired SimCLR pretraining budgets and fixed linear probes."""

import argparse
import importlib.util
import math
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate

SPEC = importlib.util.spec_from_file_location("simclr_first_round_report", Path(__file__).resolve().parents[1] / "report.py")
FIRST = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIRST)

REPO = "https://github.com/fcxuu/computer-vision-projects"
DIRECTORY = "03-self-supervised-and-transformer/cifar100-self-vs-supervised/reproduction"
CODE = f"{REPO}/tree/portfolio/reproductions-2026-11-23/{DIRECTORY}"
RELEASE = f"{REPO}/releases/tag/portfolio-simclr-extended-2026-11-24"
ASSETS = RELEASE.replace("/tag/", "/download/")
BUDGETS = (1200, 4800)
GOAL = 0.40


def assert_schedule(rows, start, steps, progress_key, rate_key, offset):
    """Check the logged learning rate against the cosine schedule actually used."""
    for row in rows:
        progress = row[progress_key] - offset
        expected = start * (0.02 + 0.98 * (1 + math.cos(math.pi * progress / steps)) / 2)
        if not math.isclose(expected, row[rate_key], rel_tol=1e-8, abs_tol=1e-12):
            raise ValueError("Learning-rate trace disagrees with the cosine schedule")


def split(path):
    with np.load(path / "split.npz", allow_pickle=False) as archive:
        train, val = (archive[key] for key in ("train", "validation"))
    if (train.shape != (45_000,) or val.shape != (5000,) or train.dtype.kind not in "iu"
            or val.dtype.kind not in "iu" or not np.array_equal(np.sort(np.r_[train, val]), np.arange(50_000))):
        raise ValueError(f"Invalid official training/validation partition in {path}")
    return train, val


def matching_split(left, right):
    return all(np.array_equal(one, two) for one, two in zip(left, right))


def load_probe(probe, config, variant, epochs, standardized=False):
    probe_config = FIRST.read_json(probe / "train.json")
    if (probe_config["source"], probe_config["protocol"], probe_config["n_train"], probe_config["n_validation"]) != (
        f"simclr_{variant}", "linear", 45_000, 5000
    ):
        raise ValueError(f"Unexpected linear evaluation protocol in {probe}")
    if probe_config.get("standardized_probe", False) is not standardized:
        raise ValueError(f"Unexpected feature standardization in {probe}")
    if probe_config["epochs"] != epochs:
        raise ValueError(f"Expected {epochs} linear-probe epochs in {probe}")
    if any(config[key] != probe_config[key] for key in ("batch", "seed", "size")):
        raise ValueError(f"Linear probe does not match source pretraining data settings in {probe}")
    probe_rows = FIRST.history(probe / "history.csv")
    if not probe_rows or [row["epoch"] for row in probe_rows] != list(range(1, epochs + 1)):
        raise ValueError(f"Incomplete linear-probe history in {probe}")
    assert_schedule(probe_rows, probe_config["lr"], epochs, "epoch", "learning_rate", 1)
    best = max(probe_rows, key=FIRST.score)
    if best != probe_config["best"]:
        raise ValueError(f"Validation-best checkpoint disagrees with history in {probe}")
    digest = config["backbone_digest"]
    if (not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest)
            or any(probe_config[f"frozen_fingerprint_{when}"] != digest for when in ("before", "after"))):
        raise ValueError(f"Frozen encoder fingerprint changed or differs from the pretrained source in {probe}")
    test = FIRST.verified_test(FIRST.read_json(probe / "test.json"), f"simclr_{variant}", probe_config, best)
    return probe_config, probe_rows, best, test


def load_variant(root, variant, steps):
    pre = root / f"pretrain-{variant}"
    probe = root / f"simclr_{variant}"
    config = FIRST.read_json(pre / "run.json")
    n = int(variant.removesuffix("k")) * 1000
    if (config["variant"], config["pretrain_samples"], config["steps"], config["temperature"]) != (variant, n, steps, 0.2):
        raise ValueError(f"Unexpected contrastive config in {pre}")
    pre_rows = FIRST.history(pre / "history.csv")
    if [row["step"] for row in pre_rows] != list(range(50, steps + 1, 50)):
        raise ValueError(f"Incomplete pretraining trace in {pre}")
    assert_schedule(pre_rows, config["lr"], steps, "step", "learning_rate", 1)
    probe_config, probe_rows, best, test = load_probe(probe, config, variant, 18)
    partition = split(pre)
    if not matching_split(partition, split(probe)):
        raise ValueError(f"Pretraining and classification partitions differ in {root}")
    subset = np.load(pre / "pretrain-indices.npy", allow_pickle=False)
    if (subset.shape != (n,) or subset.dtype.kind not in "iu" or len(np.unique(subset)) != n
            or not np.isin(subset, partition[0]).all()):
        raise ValueError(f"Invalid pretraining subset in {pre}")
    return {
        "pre": config, "probe": probe_config, "pre_rows": pre_rows, "probe_rows": probe_rows,
        "best": best, "test": test, "partition": partition, "subset": subset,
    }


def load_paired(baseline, run):
    records = {}
    pre_keys = ("temperature", "batch", "seed", "size", "lr", "optimizer", "weight_decay")
    probe_keys = ("batch", "seed", "size", "epochs", "lr", "optimizer", "weight_decay", "trainable_parameters")
    for variant in ("45k", "5k"):
        if variant == "5k" and not any((run / folder).exists() for folder in ("pretrain-5k", "simclr_5k")):
            continue
        first, later = (load_variant(root, variant, budget) for root, budget in zip((baseline, run), BUDGETS))
        if not matching_split(first["partition"], later["partition"]) or not np.array_equal(first["subset"], later["subset"]):
            raise ValueError(f"First-round and extended splits or image indices differ for {variant}")
        for key, fields in (("pre", pre_keys), ("probe", probe_keys)):
            if any(first[key][field] != later[key][field] for field in fields):
                raise ValueError(f"First-round and extended {key} settings differ for {variant}")
        records[variant] = dict(zip(BUDGETS, (first, later)))
    if "5k" in records:
        for budget in BUDGETS:
            small, big = (records[variant][budget] for variant in ("5k", "45k"))
            if (not matching_split(small["partition"], big["partition"])
                    or not np.isin(small["subset"], big["subset"]).all()
                    or any(small["pre"][key] != big["pre"][key] for key in pre_keys)
                    or any(small["probe"][key] != big["probe"][key] for key in probe_keys)):
                raise ValueError("The two image-pool sizes do not share the same controlled protocol")
    return records


def load_extended_probes(run, records, *, standardized=False):
    name = "standardized-60" if standardized else "probe-60"
    folders = {steps: run / name / str(steps) for steps in BUDGETS}
    if not any(path.exists() for path in folders.values()):
        return {}
    if not all(path.is_dir() for path in folders.values()):
        raise ValueError(f"The {name} comparison requires both pretraining budgets")
    results = {}
    matching = ("batch", "seed", "size", "lr", "optimizer", "weight_decay", "trainable_parameters")
    for steps, path in folders.items():
        original = records["45k"][steps]
        config, rows, best, test = load_probe(path, original["pre"], "45k", 60, standardized=standardized)
        if (not matching_split(original["partition"], split(path))
                or any(config[key] != original["probe"][key] for key in matching)):
            raise ValueError(f"The 60-epoch linear probe changed the fixed split or optimization settings in {path}")
        results[steps] = {"probe": config, "probe_rows": rows, "best": best, "test": test}
    return results


def load_continuation(run, records, extended):
    pre, probe = (run / name for name in ("continuation-45k", "continuation-probe"))
    if not pre.exists() and not probe.exists():
        return None
    if not pre.is_dir() or not probe.is_dir() or 4800 not in extended:
        raise ValueError("Continuation comparison requires its encoder, probe and the original raw 60-epoch control")
    original = records["45k"][4800]
    config = FIRST.read_json(pre / "run.json")
    if (config["initial_steps"], config["steps"], config["total_pretrain_steps"], config["lr"]) != (
        4800, 3600, 8400, 0.0004
    ):
        raise ValueError(f"Invalid continuation budget or initial LR in {pre}")
    keys = ("variant", "pretrain_samples", "temperature", "batch", "seed", "size", "optimizer", "weight_decay")
    if any(config[key] != original["pre"][key] for key in keys):
        raise ValueError(f"Continuation changed the fixed pretraining protocol in {pre}")
    subset = np.load(pre / "pretrain-indices.npy", allow_pickle=False)
    if (subset.dtype.kind not in "iu" or not np.array_equal(subset, original["subset"])
            or not matching_split(split(pre), original["partition"])
            or not matching_split(split(probe), original["partition"])):
        raise ValueError("Continuation changed the pretraining pool or evaluation partition")
    pre_rows = FIRST.history(pre / "history.csv")
    if [row["step"] for row in pre_rows] != list(range(50, config["steps"] + 1, 50)):
        raise ValueError(f"Incomplete continuation trace in {pre}")
    assert_schedule(pre_rows, config["lr"], config["steps"], "step", "learning_rate", 1)
    probe_config, probe_rows, best, test = load_probe(probe, config, "45k", 60)
    keys = ("lr", "optimizer", "weight_decay", "trainable_parameters")
    if any(probe_config[key] != extended[4800]["probe"][key] for key in keys):
        raise ValueError(f"Continuation changed the fixed raw linear-probe settings in {probe}")
    return {"pre": config, "pre_rows": pre_rows, "probe": probe_config,
            "probe_rows": probe_rows, "best": best, "test": test}


def plot_continuation(original, continued, out):
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), layout="constrained")
    initial = continued["pre"]["initial_steps"]
    for axis, name, label in zip(axes, ("ntxent", "pair_top1"), ("NT-Xent", "Paired-view retrieval top-1")):
        for run, offset, color, series in (
            (original, 0, "#14736c", "Original pretraining"),
            (continued, initial, "#a35039", "Continuation"),
        ):
            rows = run["pre_rows"]
            axis.plot([row["step"] + offset for row in rows], [row[name] for row in rows], color=color, label=series)
        axis.axvline(initial, color="#607483", linestyle=":", linewidth=1)
        axis.set(xlabel="Cumulative pretraining step", title=label)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=0.18)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].legend(frameon=False, fontsize=8)
    FIRST.save_plot(figure, out / "continuation-curves.png")


def plot_extended_probes(original, extended, out):
    fig, (validation, test_axis) = plt.subplots(1, 2, figsize=(10.5, 3.8), layout="constrained")
    for index, steps in enumerate(BUDGETS):
        color = ("#146f69", "#a35039")[index]
        scores = []
        for data, style in ((original[steps], "--"), (extended[steps], "-")):
            label = f"{steps:,} pretrain / {data['probe']['epochs']} probe"
            rows, best = data["probe_rows"], data["best"]
            validation.plot([row["epoch"] for row in rows], [row["val_accuracy"] for row in rows],
                            color=color, linestyle=style, label=label)
            validation.scatter(best["epoch"], best["val_accuracy"], s=18, color=color, zorder=4)
            scores.append(data["test"]["accuracy"])
        slots = np.arange(2) + (index - 0.5) * 0.32
        bars = test_axis.bar(slots, scores, width=0.30, color=color, label=f"{steps:,} pretrain steps")
        test_axis.bar_label(bars, labels=[f"{value:.2%}" for value in scores], padding=3, fontsize=8)
    validation.set(xlabel="Linear-probe epoch", title="Validation top-1")
    validation.xaxis.set_major_locator(MaxNLocator(integer=True))
    validation.legend(frameon=False, fontsize=7)
    test_axis.set(title="Official test top-1 at selected epoch", xlabel="Linear-probe budget",
                  xticks=[0, 1], xticklabels=["18 epochs", "60 epochs"])
    test_axis.margins(y=0.22)
    test_axis.legend(frameon=False, fontsize=7, loc="lower right")
    for axis in (validation, test_axis):
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(axis="y", alpha=0.18)
    FIRST.save_plot(fig, out / "probe-curves.png")


def plot_curves(records, out):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.4), layout="constrained")
    for variant, budgets in records.items():
        color = "#14736c" if variant == "45k" else "#a35039"
        for budget, result in budgets.items():
            style = "--" if budget == BUDGETS[0] else "-"
            label = f"{variant} / {budget:,} steps"
            for axis, rows, x, y in (
                (axes[0, 0], result["pre_rows"], "step", "ntxent"),
                (axes[0, 1], result["pre_rows"], "step", "pair_top1"),
                (axes[1, 0], result["probe_rows"], "epoch", "val_loss"),
                (axes[1, 1], result["probe_rows"], "epoch", "val_accuracy"),
            ):
                axis.plot([row[x] for row in rows], [row[y] for row in rows],
                          color=color, linestyle=style, label=label)
                if x == "epoch":
                    axis.scatter(result["best"][x], result["best"][y], color=color, s=16)
    for axis, title in zip(axes.flat, ("Pretraining NT-Xent", "Pretraining paired-view retrieval top-1",
                                     "Linear-probe validation CE", "Linear-probe validation top-1")):
        axis.set(title=title, xlabel="Optimizer step" if title.startswith("Pretraining") else "Probe epoch")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=0.18)
    for axis in axes[:, 1]:
        axis.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0, 1].legend(frameon=False, fontsize=8)
    FIRST.save_plot(fig, out / "budget-curves.png")


def select_release(records, extended, standardized, continuation=None):
    candidates = [(f"{pool} / 4,800 pretraining steps / 18 probe epochs", data[4800])
                  for pool, data in records.items()]
    for results, label in ((extended, "raw features"), (standardized, "standardized features")):
        candidates.extend((f"45k / {steps:,} pretraining steps / 60 probe epochs / {label}", result)
                          for steps, result in results.items())
    if continuation:
        candidates.append(("45k / 8,400 cumulative pretraining steps / 60 probe epochs / raw features", continuation))
    return max(candidates, key=lambda candidate: FIRST.score(candidate[1]["best"]))


def build_pdf(records, extended, standardized, continuation, out):
    title = ParagraphStyle("Title", fontName="Helvetica-Bold", fontSize=24, leading=28, textColor=FIRST.INK, spaceAfter=13)
    heading = ParagraphStyle("Heading", fontName="Helvetica-Bold", fontSize=14, leading=18,
                             textColor=FIRST.INK, spaceBefore=10, spaceAfter=5)
    body = ParagraphStyle("Body", fontName="Helvetica", fontSize=10, leading=14,
                          textColor=FIRST.INK, spaceAfter=7)
    doc = SimpleDocTemplate(str(out / "report.pdf"), pagesize=A4, rightMargin=43, leftMargin=43,
                            topMargin=40, bottomMargin=53, title="Extending SimCLR pretraining on CIFAR-100",
                            author="fcxuu / computer-vision-projects")
    initial, extended_pretrain = (records["45k"][steps] for steps in BUDGETS)
    before, after = initial["test"], extended_pretrain["test"]
    pre, probe = extended_pretrain["pre"], extended_pretrain["probe"]
    diff = 100 * (after["accuracy"] - before["accuracy"])
    published_name, published = select_release(records, extended, standardized, continuation)
    public_test = published["test"]
    gap = 100 * (public_test["accuracy"] - GOAL)
    target_note = (f"It exceeded the {GOAL:.0%} project target by {gap:.2f} points." if gap > 0
                   else f"It matched the {GOAL:.0%} project target." if gap == 0
                   else f"It was {abs(gap):.2f} points below the {GOAL:.0%} project target.")
    rows = [["Pool", "Steps", "Best ep.", "Val top-1", "Test top-1", "Test top-5"]]
    comparisons = []
    for variant, budgets in records.items():
        for steps, result in budgets.items():
            test, best = result["test"], result["best"]
            rows.append([variant, f"{steps:,}", f"{best['epoch']:.0f}", f"{best['val_accuracy']:.2%}",
                         f"{test['accuracy']:.2%} ({test['correct']:,})", f"{test['top5']:.2%} ({test['top5_correct']:,})"])
        short, long = (budgets[steps]["test"] for steps in BUDGETS)
        comparisons.append(f"The {variant} pool changed by {100 * (long['accuracy'] - short['accuracy']):+.2f} points "
                           f"in top-1 and {100 * (long['top5'] - short['top5']):+.2f} points in top-5.")
    if "5k" in records:
        size_notes = []
        for steps in BUDGETS:
            large, small = (records[variant][steps]["test"] for variant in ("45k", "5k"))
            size_notes.append(f"At {steps:,} steps, 45k minus 5k was "
                              f"{100 * (large['accuracy'] - small['accuracy']):+.2f} top-1 points and "
                              f"{100 * (large['top5'] - small['top5']):+.2f} top-5 points.")
        comparisons.extend(size_notes)
    assets = " &nbsp; ".join(
        f'<link href="{ASSETS}/{kind}-{variant}-4800.pt" color="#14736c"><u>{variant} {label}</u></link>'
        for variant in records for kind, label in (("pretrained", "encoder"), ("simclr", "classifier"))
    )
    story = [
        Paragraph("Extending SimCLR pretraining on CIFAR-100", title),
        Paragraph(f"The validation-selected release result is <b>{published_name}</b>. At probe epoch "
                  f"{published['best']['epoch']:.0f}, its validation top-1 was {published['best']['val_accuracy']:.2%}. "
                  f"Official test top-1 was <b>{public_test['accuracy']:.2%}</b> "
                  f"({public_test['correct']:,}/{public_test['samples']:,}) and top-5 was {public_test['top5']:.2%}. "
                  f"{target_note}", body),
        Paragraph(f"With the 18-epoch probe fixed, the 45k run moved from {before['accuracy']:.2%} to "
                  f"{after['accuracy']:.2%} official test top-1 ({diff:+.2f} percentage points).", body),
        Paragraph("Paired training and evaluation", heading),
        Paragraph(f"Each pool reuses identical saved image indices across budgets. Each run starts from scratch "
                  f"with ResNet-18, paired augmented views, NT-Xent at temperature {pre['temperature']:g}, "
                  f"batch size {pre['batch']}, input side {pre['size']}, seed {pre['seed']} and "
                  f"{pre['optimizer']} at initial LR {pre['lr']:g} with weight decay {pre['weight_decay']:g}. "
                  "Cosine decay spans each run's entire budget, so learning rates after the first update "
                  "follow different schedules in absolute step time.", body),
        Paragraph(f"The pretraining-budget comparison uses frozen linear probes on the same {probe['n_train']:,} training and "
                  f"{probe['n_validation']:,} validation images, {probe['epochs']} epochs, "
                  f"{probe['optimizer']} at initial LR {probe['lr']:g}, and the same cosine schedule. "
                  "The encoder's SHA-256 fingerprint matches its pretrained source and stays unchanged "
                  "during the probe. Highest validation top-1, then lowest validation CE, selects each checkpoint and "
                  "the release candidate across the completed second-round runs. Test scores never enter that ranking. "
                  "Official testing began after "
                  f"validation selection for the evaluated checkpoints and uses a separate {after['samples']:,}-image split.", body),
        Paragraph("Fixed-checkpoint results", heading),
        FIRST.table(rows, [56, 65, 62, 85, 114, 114]),
        Paragraph("Parentheses are counts of correct predictions, with the same official test denominator for all runs.", body),
        Paragraph(" ".join(comparisons), body),
        Paragraph(
            f'<link href="{CODE}/evidence/report.pdf" color="#14736c"><u>First-round report</u></link> &nbsp; '
            f'<link href="{CODE}/optimization" color="#14736c"><u>Second-round code</u></link> &nbsp; '
            f'<link href="{RELEASE}" color="#14736c"><u>Second-round weights release</u></link>', body),
        Paragraph(assets, body),
        PageBreak(),
        Paragraph("Contrastive and downstream curves", title),
        Image(str(out / "budget-curves.png"), width=499, height=499 * 6.4 / 10.5),
        Paragraph("Pretraining retrieval top-1 tracks whether an augmented view ranks its matching view first "
                  "within the contrastive batch. The downstream validation top-1 measures CIFAR-100 class "
                  "predictions. Dots mark the selected linear-probe checkpoints.", body),
        Paragraph(f"The fixed batch size and step counts give {BUDGETS[0] * pre['batch']:,} and "
                  f"{BUDGETS[1] * pre['batch']:,} image draws before the two-view transformation. "
                  "The results describe a single seed and this optimization schedule.", body),
    ]
    if extended:
        probe_table = [["Pretrain steps", "Probe epochs", "Selected epoch", "Val top-1", "Test top-1"]]
        for steps in BUDGETS:
            for result in (records["45k"][steps], extended[steps]):
                best, test = result["best"], result["test"]
                probe_table.append([f"{steps:,}", result["probe"]["epochs"], int(best["epoch"]),
                                    f"{best['val_accuracy']:.2%}", f"{test['accuracy']:.2%} ({test['correct']:,})"])
        short, long = BUDGETS
        original = records["45k"]
        budget_effect = {
            data[short]["probe"]["epochs"]: 100 * (data[long]["test"]["accuracy"] - data[short]["test"]["accuracy"])
            for data in (original, extended)
        }
        probe_effect = {
            steps: 100 * (extended[steps]["test"]["accuracy"] - original[steps]["test"]["accuracy"])
            for steps in BUDGETS
        }
        story.extend([
            PageBreak(),
            Paragraph("Separating pretraining and probe budgets", title),
            Paragraph("The 45k experiment crosses two encoder pretraining budgets with two linear-probe budgets. "
                      "Each encoder is reused for both probe lengths. Identical train and validation partitions, "
                      "matching encoder fingerprints and a fixed probe optimizer and initial learning rate control "
                      "the comparison. Cosine decay is stretched to the specified probe length.", body),
            FIRST.table(probe_table, [102, 96, 106, 78, 110]),
            Image(str(out / "probe-curves.png"), width=499, height=499 * 3.8 / 10.5),
            Paragraph("Dots mark each run's highest validation top-1, with lower validation CE breaking ties. "
                      "The official test bars use those checkpoints and the common 10,000-image denominator. "
                      "Parentheses show the number of correct classifications.", body),
            Paragraph(f"Increasing pretraining from {short:,} to {long:,} steps changed official top-1 by "
                      f"{budget_effect[18]:+.2f} points at 18 probe epochs and {budget_effect[60]:+.2f} points "
                      f"at 60 probe epochs. Increasing the probe from 18 to 60 epochs changed official top-1 by "
                      f"{probe_effect[short]:+.2f} points with {short:,} pretraining steps and {probe_effect[long]:+.2f} "
                      f"points with {long:,} pretraining steps.", body),
        ])
    if standardized:
        normalization_rows = [["Pretrain steps", "Probe features", "Selected epoch", "Val top-1", "Test top-1", "Test top-5"]]
        changes = []
        for steps in BUDGETS:
            raw, normalized = extended[steps], standardized[steps]
            for label, result in (("Raw", raw), ("Standardized", normalized)):
                best, test = result["best"], result["test"]
                normalization_rows.append([f"{steps:,}", label, int(best["epoch"]), f"{best['val_accuracy']:.2%}",
                                           f"{test['accuracy']:.2%}", f"{test['top5']:.2%}"])
            validation_change = 100 * (normalized["best"]["val_accuracy"] - raw["best"]["val_accuracy"])
            test_change = 100 * (normalized["test"]["accuracy"] - raw["test"]["accuracy"])
            changes.append(f"At {steps:,} steps, standardization changed validation top-1 by "
                           f"{validation_change:+.2f} points and official test top-1 by {test_change:+.2f} points.")
        story.extend([
            PageBreak(),
            Paragraph("Standardizing frozen encoder features", title),
            Paragraph("Each pair keeps the 45k encoder, data split, initialization seed and 60-epoch optimization "
                      "budget fixed. Feature means and population variances are fitted on the training features "
                      "only. Each feature is centered and divided by its standard deviation, with a floor of "
                      "0.0001. The fitted statistics also transform validation features.", body),
            Paragraph("Checkpoint export folds centering and scaling into the linear layer. The published model "
                      "accepts raw encoder features on reload, with the same logits as the standardized probe "
                      "up to numerical precision.", body),
            FIRST.table(normalization_rows, [91, 91, 84, 71, 76, 76]),
            Paragraph(" ".join(changes), body),
        ])
    if continuation:
        previous = extended[4800]
        config = continuation["pre"]
        compared = [["Pretrain steps", "Selected probe epoch", "Val top-1", "Test top-1", "Test top-5"]]
        for steps, result in ((config["initial_steps"], previous), (config["total_pretrain_steps"], continuation)):
            best, test = result["best"], result["test"]
            compared.append([f"{steps:,}", int(best["epoch"]), f"{best['val_accuracy']:.2%}",
                             f"{test['accuracy']:.2%} ({test['correct']:,})",
                             f"{test['top5']:.2%} ({test['top5_correct']:,})"])
        validation_gain = 100 * (continuation["best"]["val_accuracy"] - previous["best"]["val_accuracy"])
        top1_gain = 100 * (continuation["test"]["accuracy"] - previous["test"]["accuracy"])
        top5_gain = 100 * (continuation["test"]["top5"] - previous["test"]["top5"])
        story.extend([
            PageBreak(),
            Paragraph("Continuing the 45k encoder", title),
            Paragraph(f"The continuation starts from the {config['initial_steps']:,}-step encoder and adds "
                      f"{config['steps']:,} updates on the same ordered image pool. It restarts {config['optimizer']} "
                      f"at learning rate {config['lr']:g} with a new cosine segment and independent shuffling "
                      "and augmentation streams. Both encoders are evaluated with the same raw-feature "
                      "60-epoch linear-probe protocol.", body),
            FIRST.table(compared, [90, 116, 75, 108, 108]),
            Image(str(out / "continuation-curves.png"), width=499, height=499 * 3.8 / 10.5),
            Paragraph("The vertical line separates the two learning-rate segments. Retrieval top-1 measures "
                      "paired-view matching during pretraining.", body),
            Paragraph(f"The continuation changed validation top-1 by {validation_gain:+.2f} points, official "
                      f"test top-1 by {top1_gain:+.2f} points and official test top-5 by {top5_gain:+.2f} points. "
                      "These differences estimate the combined continuation procedure, including its optimizer "
                      "restart and new learning-rate segment.", body),
        ])
    doc.build(story, onFirstPage=FIRST.footer, onLaterPages=FIRST.footer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = load_paired(args.baseline, args.run)
    extended = load_extended_probes(args.run, records)
    standardized = load_extended_probes(args.run, records, standardized=True)
    if standardized and not extended:
        raise ValueError("The standardized comparison requires the paired raw 60-epoch probes")
    continuation = load_continuation(args.run, records, extended)
    args.out.mkdir(parents=True, exist_ok=True)
    plot_curves(records, args.out)
    if extended:
        plot_extended_probes(records["45k"], extended, args.out)
    if continuation:
        plot_continuation(records["45k"][4800], continuation, args.out)
    build_pdf(records, extended, standardized, continuation, args.out)


if __name__ == "__main__":
    main()
