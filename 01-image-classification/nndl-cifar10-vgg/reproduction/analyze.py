"""Analyze a completed CIFAR-10 suite without retraining or reading test data."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import PercentFormatter
import numpy as np
import torch
from torch.nn.functional import cross_entropy

from data import CIFAR10, Partition, ROOT, device_for
from landscape import gradient_probe, surface
from model import VARIANTS, checkpoint
from suite import SWEEP

PAIRED = VARIANTS[:2]
NAMES = dict(zip(VARIANTS, ("VGG-A", "VGG-A + BN", "VGG-A + dropout")))
COLORS = dict(zip(VARIANTS, ("#9b452e", "#19747a", "#6a5795")))
SUBSET_SEED = 41010


def table(path, fields):
    with path.open(newline="") as stream:
        rows = np.array([[float(row[field]) for field in fields] for row in csv.DictReader(stream)])
    if rows.ndim != 2 or not len(rows) or not np.isfinite(rows).all():
        raise ValueError(f"Empty or non-finite metric series in {path}")
    return rows


def contiguous_axis(values, path):
    if not np.array_equal(values, np.arange(1, len(values) + 1)):
        raise ValueError(f"Expected contiguous, ordered indices in {path}")
    return values.astype(int)


def save_figure(figure, path):
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)


def plot_curves(run, out):
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.7), layout="constrained")
    fields = ("epoch", "train_ce", "val_ce", "val_accuracy")
    for variant in VARIANTS:
        path = run / f"full-{variant}" / "epochs.csv"
        rows = table(path, fields)
        epochs = contiguous_axis(rows[:, 0], path)
        for index, axis in enumerate(axes, start=1):
            axis.plot(epochs, rows[:, index], label=NAMES[variant], color=COLORS[variant])
    for axis, label in zip(axes, ("Online train CE", "Validation CE", "Validation accuracy")):
        axis.set(xlabel="Epoch", title=label)
        axis.grid(alpha=0.2)
    axes[2].yaxis.set_major_formatter(PercentFormatter(1))
    axes[2].legend(frameon=False, fontsize=8)
    save_figure(figure, out / "curves.png")


def summarize_gradient(values):
    return {name: float(value) for name, value in zip(
        ("minimum", "p25", "median", "p75", "maximum"), np.quantile(values, [0, 0.25, 0.5, 0.75, 1])
    )}


def plot_sweep(run, out):
    figure, (loss, first, last) = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    fields = ("step", "epoch", "loss", "first_conv_grad", "output_grad")
    summaries = {}
    expected = None
    for offset, variant in enumerate(PAIRED):
        per_lr = {}
        for lr in SWEEP:
            path = run / f"sweep-{variant}-{lr}" / "batches.csv"
            rows = table(path, fields)
            steps = contiguous_axis(rows[:, 0], path)
            axes = rows[:, :2]
            if expected is None:
                expected = axes
            elif not np.array_equal(axes, expected):
                raise ValueError(f"Sweep step or epoch alignment differs in {path}")
            per_lr[lr] = rows[:, 2:]
        losses = np.stack([values[:, 0] for values in per_lr.values()])
        lower, upper = losses.min(axis=0), losses.max(axis=0)
        color = COLORS[variant]
        loss.fill_between(steps, lower, upper, color=color, alpha=0.20, label=NAMES[variant])
        loss.plot(steps, lower, color=color, linewidth=0.7)
        loss.plot(steps, upper, color=color, linewidth=0.7)
        summaries[variant] = {
            "steps": len(steps), "step": steps.tolist(), "min_ce": lower.tolist(), "max_ce": upper.tolist(),
            "mean_width": float((upper - lower).mean()), "max_width": float((upper - lower).max()),
            "gradient_norms": {
                lr: {key: summarize_gradient(values[:, index])
                     for index, key in enumerate(fields[3:], start=1)}
                for lr, values in per_lr.items()
            },
        }
        positions = np.arange(len(SWEEP)) + (-0.19 if offset == 0 else 0.19)
        for index, axis in enumerate((first, last), start=1):
            boxes = axis.boxplot([values[:, index] for values in per_lr.values()], positions=positions,
                                 widths=0.28, showfliers=False, patch_artist=True,
                                 medianprops={"color": color, "linewidth": 1.3})
            for box in boxes["boxes"]:
                box.set(facecolor=color, alpha=0.45)
    loss.set(title="Min/max CE across four learning rates", xlabel="Optimizer step", ylabel="Batch CE")
    loss.legend(frameon=False, fontsize=8)
    for axis, title in zip((first, last), ("First convolution", "Output layer")):
        axis.set(title=f"{title} weight gradients", xlabel="Constant learning rate", ylabel="L2 norm",
                 xticks=np.arange(len(SWEEP)), xticklabels=SWEEP)
        axis.grid(axis="y", alpha=0.2)
    save_figure(figure, out / "lr-sweep.png")
    return summaries


def read_split(path):
    with np.load(path, allow_pickle=False) as archive:
        train, validation = archive["train"], archive["validation"]
    return train, validation


def matched_indices(run, labels):
    train, validation = read_split(run / f"full-{PAIRED[0]}" / "split.npz")
    other_train, other_validation = read_split(run / f"full-{PAIRED[1]}" / "split.npz")
    if not np.array_equal(train, other_train) or not np.array_equal(validation, other_validation):
        raise ValueError("The two full VGG checkpoints used different train/validation partitions")
    members = np.concatenate((train, validation))
    if not np.array_equal(np.sort(members), np.arange(len(labels))):
        raise ValueError("The saved train/validation split is not disjoint and exhaustive")
    rng = np.random.default_rng(SUBSET_SEED)
    selected = []
    for category in range(10):
        pool = validation[labels[validation] == category]
        if len(pool) < 50:
            raise ValueError("The validation split needs at least 50 examples per class")
        selected.append(rng.permutation(pool)[:50])
    return np.sort(np.concatenate(selected)), np.sort(np.concatenate([rows[:15] for rows in selected]))


def load_best(run, variant, device):
    directory = run / f"full-{variant}"
    meta = json.loads((directory / "run.json").read_text())
    model, state = checkpoint(directory / "best.pt", device)
    if state["variant"] != variant or state["epoch"] != meta["best"]["epoch"]:
        raise ValueError(f"Checkpoint disagrees with the validation-best record in {directory}")
    return model.eval(), state["epoch"]


def plot_filters(run, out):
    figure, axes = plt.subplots(3, 8, figsize=(10, 4.2), layout="constrained")
    for variant, panels in zip(VARIANTS, axes):
        model, epoch = load_best(run, variant, torch.device("cpu"))
        weights = model.features[0].weight.detach().numpy()
        strongest = np.argsort(-np.linalg.norm(weights.reshape(len(weights), -1), axis=1), kind="stable")[:8]
        limit = float(np.abs(weights[strongest]).max())
        for panel in panels:
            panel.axis("off")
        for panel, unit in zip(panels, strongest):
            rgb = np.clip(weights[unit].transpose(1, 2, 0) / (2 * limit) + 0.5, 0, 1) if limit else np.full((3, 3, 3), 0.5)
            panel.imshow(rgb, interpolation="nearest")
            panel.set_title(f"Filter {unit + 1}", fontsize=8)
        panels[0].set_title(f"{NAMES[variant]} / epoch {epoch}\nFilter {strongest[0] + 1}", fontsize=8)
    figure.suptitle("First-layer filters with the largest L2 norms (zero = mid-gray)")
    save_figure(figure, out / "filters.png")


@torch.no_grad()
def independent_ce(model, batches):
    return math.fsum(cross_entropy(model(images), labels, reduction="sum").item()
                     for images, labels in batches) / sum(len(labels) for _, labels in batches)


def plot_landscapes(surfaces, out):
    differences = {variant: np.asarray(result["losses"]) - result["origin"] for variant, result in surfaces.items()}
    limit = max(float(np.abs(grid).max()) for grid in differences.values()) or 1.0
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.1), layout="constrained")
    for variant, axis in zip(PAIRED, axes):
        result = surfaces[variant]
        mesh = axis.pcolormesh(result["axis_x"], result["axis_y"], differences[variant],
                               shading="nearest", cmap="RdBu_r", norm=norm)
        axis.scatter(0, 0, marker="+", color="black", s=70)
        axis.set(title=NAMES[variant], xlabel="Filter-normalized direction 1", ylabel="Direction 2", aspect="equal")
    figure.colorbar(mesh, ax=axes, label="Validation CE minus checkpoint CE")
    save_figure(figure, out / "surface.png")


def plot_probes(probes, out):
    figure, (loss, cosine, difference) = plt.subplots(1, 3, figsize=(11, 4), layout="constrained")
    for variant in PAIRED:
        rows = probes[variant]["points"]
        radius = [row["relative_radius"] for row in rows]
        kwargs = {"color": COLORS[variant], "marker": "o", "markersize": 3}
        center = probes[variant]["center_loss"]
        loss.plot(radius, [row["actual_loss"] - center for row in rows], label=NAMES[variant], **kwargs)
        loss.plot(radius, [row["predicted_loss"] - center for row in rows], linestyle="--", **kwargs)
        cosine.plot(radius, [row["gradient_cosine"] for row in rows], **kwargs)
        difference.plot(radius, [row["gradient_difference"] for row in rows], **kwargs)
    for axis, title, ylabel in (
        (loss, "Loss prediction", "CE change from center"),
        (cosine, "Gradient alignment", "Cosine with center gradient"),
        (difference, "Gradient change", "L2 norm of gradient difference"),
    ):
        axis.set(title=title, xlabel="Relative perturbation radius", ylabel=ylabel)
        axis.grid(alpha=0.2)
    loss.legend(frameon=False, fontsize=8, title="Solid actual / dashed predicted", title_fontsize=8)
    save_figure(figure, out / "gradient-probe.png")


def analyze(run, out, data, device):
    dataset = CIFAR10(data, train=True, download=False)
    labels = np.asarray(dataset.targets)
    surface_ids, gradient_ids = matched_indices(run, labels)
    surface_batches, gradient_batches = (
        tuple(Partition.from_arrays(dataset.data[indices], labels[indices], device).batches(64))
        for indices in (surface_ids, gradient_ids)
    )
    out.mkdir(parents=True, exist_ok=True)
    sweep = plot_sweep(run, out)
    plot_curves(run, out)
    plot_filters(run, out)
    surfaces, probes, epochs, origins = {}, {}, {}, {}
    for variant in PAIRED:
        print(f"Analyzing {NAMES[variant]} on {device}", flush=True)
        model, epoch = load_best(run, variant, device)
        epochs[variant] = epoch
        origins[variant] = {"surface_ce": independent_ce(model, surface_batches),
                            "gradient_ce": independent_ce(model, gradient_batches)}
        surfaces[variant] = surface(model, surface_batches, points=7)
        probes[variant] = gradient_probe(model, gradient_batches)
        for actual, reference in (
            (surfaces[variant]["origin"], origins[variant]["surface_ce"]),
            (probes[variant]["center_loss"], origins[variant]["gradient_ce"]),
        ):
            if not math.isclose(actual, reference, rel_tol=1e-5, abs_tol=1e-6):
                raise RuntimeError(f"Origin does not match the independent checkpoint evaluation for {variant}")
    plot_landscapes(surfaces, out)
    plot_probes(probes, out)
    payload = {
        "device": str(device), "subset_seed": SUBSET_SEED,
        "surface_indices": surface_ids.tolist(), "gradient_indices": gradient_ids.tolist(),
        "checkpoint_epochs": epochs, "origin_checks": origins,
        "surfaces": surfaces, "gradient_probes": probes, "lr_sweep": sweep,
    }
    partial = out / "analysis.json.tmp"
    partial.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    partial.replace(out / "analysis.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "cifar10")
    args = parser.parse_args()
    torch.set_num_threads(4)
    analyze(args.run, args.out, args.data, device_for(args.device))


if __name__ == "__main__":
    main()
