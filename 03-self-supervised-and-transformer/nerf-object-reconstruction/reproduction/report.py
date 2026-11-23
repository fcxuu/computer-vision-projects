"""Build the Buddha NeRF report from registered cameras and recorded metrics."""

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from PIL import Image as Raster
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Table, TableStyle

REPO = "https://github.com/fcxuu/computer-vision-projects"
CODE = f"{REPO}/tree/portfolio/reproductions-2026-11-23/03-self-supervised-and-transformer/nerf-object-reconstruction/reproduction"
RELEASE = f"{REPO}/releases/download/portfolio-nerf-buddha-2026-11-24"
FRAMEWORK = "https://github.com/sadeeqgandalf/Nerf-3D-Reconstruction"
DOI = "https://doi.org/10.5281/zenodo.1203283"
INK = colors.HexColor("#19334b")
TEAL = "#14736c"
ORANGE = "#b35036"
METRICS = ("mse", "psnr", "ssim")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_series(path):
    with path.open(newline="") as stream:
        rows = [{key: float(value) if value else None for key, value in row.items()}
                for row in csv.DictReader(stream)]
    if not rows or any(not math.isfinite(value) for row in rows for value in row.values() if value is not None):
        raise ValueError(f"Empty or non-finite history in {path}")
    steps = [row["step"] for row in rows]
    if any(step is None or step < 1 or not step.is_integer() for step in steps) or steps != sorted(set(steps)):
        raise ValueError(f"Expected unique, increasing optimizer steps in {path}")
    return rows


def validate_metrics(row):
    if (not all(math.isfinite(row[key]) for key in METRICS)
            or not 0 < row["mse"] <= 1 or not -1 <= row["ssim"] <= 1):
        raise ValueError("Invalid normalized RGB metric record")


def frame_key(value):
    return Path(str(value)).stem


def verify(scene, run, out):
    registration = scene["registration"]
    counts = Counter(frame["split"] for frame in scene["frames"])
    if (set(counts) != {"train", "val", "test"} or counts != registration["split_counts"]
            or sum(counts.values()) != registration["registered_frames"]
            or registration["registered_frames"] > registration["source_frames"]
            or len({frame_key(row["name"]) for row in scene["frames"]}) != len(scene["frames"])):
        raise ValueError("Scene registration and split counts disagree")
    if not 0 < scene["near"] < scene["far"] or min(scene["width"], scene["height"]) < 1:
        raise ValueError("Invalid scene resolution or ray bounds")
    config = read_json(run / "train.json")
    history = read_series(run / "history.csv")
    if history[-1]["step"] != config["config"]["steps"]:
        raise ValueError("Training history does not reach the configured step budget")
    validation = [row for row in history if row["val_psnr"] is not None]
    best = max(validation, key=lambda row: row["val_psnr"])
    if best["step"] != config["best_step"] or not math.isclose(best["val_psnr"], config["best_val_psnr"], abs_tol=1e-6):
        raise ValueError("Best validation PSNR does not reproduce the recorded selection")
    test = read_json(out / "test-metrics.json")
    if test["step"] != config["best_step"]:
        raise ValueError("Test metrics are for a different checkpoint")
    names = {frame_key(frame["name"]) for frame in scene["frames"] if frame["split"] == "test"}
    observed = [frame_key(row["id"]) for row in test["frames"]]
    if len(observed) != len(names) or set(observed) != names:
        raise ValueError("Test metrics do not cover the fixed held-out split exactly once")
    validate_metrics(test["mean"])
    for row in test["frames"]:
        validate_metrics(row)
    for key in METRICS:
        mean = math.fsum(row[key] for row in test["frames"]) / len(names)
        if not math.isclose(mean, test["mean"][key], abs_tol=1e-6, rel_tol=1e-6):
            raise ValueError(f"Reported test {key} disagrees with its per-image average")
    track_path = out / "trajectory.csv"
    track = read_series(track_path) if track_path.is_file() else []
    if track:
        for row in track:
            validate_metrics(row)
        if not {row["step"] for row in track} <= {row["step"] for row in history}:
            raise ValueError("Held-out trajectory references unrecorded optimizer steps")
        matching = next((row for row in track if row["step"] == config["best_step"]), None)
        if matching and any(not math.isclose(matching[key], test["mean"][key], abs_tol=1e-6, rel_tol=1e-6)
                            for key in METRICS):
            raise ValueError("The posterior test rerun disagrees at the selected checkpoint")
    orbit_path = out / "orbit-trajectory.json"
    orbit = read_json(orbit_path) if orbit_path.is_file() else None
    if orbit is not None:
        if not orbit["frames"] or min(orbit["width"], orbit["height"], orbit["fps"]) <= 0:
            raise ValueError("Invalid orbit metadata")
        probe_path = out / "orbit-ffprobe.json"
        if probe_path.is_file():
            stream = next(row for row in read_json(probe_path)["streams"] if row["codec_type"] == "video")
            if (stream["width"], stream["height"]) != (orbit["width"], orbit["height"]):
                raise ValueError("Encoded orbit resolution disagrees with metadata")
            count = stream.get("nb_frames")
            if count not in (None, "N/A") and int(count) != len(orbit["frames"]):
                raise ValueError("Encoded orbit frame count disagrees with metadata")
    return config, history, validation, test, track, orbit


def plot_curves(history, val, test, trajectory, best_step, out):
    fig, (mse, psnr) = plt.subplots(1, 2, figsize=(10, 3.65), layout="constrained")
    for axis, name, ylabel in ((mse, "mse", "RGB mean squared error"), (psnr, "psnr", "PSNR (dB)")):
        axis.plot([row["step"] for row in history], [row[f"train_{name}"] for row in history],
                  color="#526b84", linewidth=1, label="Training")
        axis.plot([row["step"] for row in val], [row[f"val_{name}"] for row in val],
                  color=TEAL, marker="o", markersize=4, label="Validation")
        if trajectory:
            axis.plot([row["step"] for row in trajectory], [row[name] for row in trajectory],
                      color=ORANGE, linestyle="--", marker="o", markersize=3, label="Test after training")
        axis.scatter(best_step, test["mean"][name], color=ORANGE, marker="D", s=25, label="Selected checkpoint test")
        axis.axvline(best_step, color="#a5afb7", linestyle=":", linewidth=1)
        axis.set(xlabel="Optimizer step", ylabel=ylabel)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=0.18)
    psnr.legend(frameon=False, fontsize=7)
    fig.savefig(out / "training-curves.png", dpi=180, facecolor="white")
    plt.close(fig)


def figure(path, width, max_height):
    with Raster.open(path) as raster:
        w, h = raster.size
    scale = min(width / w, max_height / h)
    return Image(str(path), width=w * scale, height=h * scale)


def footer(canvas, document):
    canvas.setFillColor(INK)
    canvas.setFont("Helvetica", 9)
    canvas.drawString(43, 27, "fcxuu / computer-vision-projects")
    canvas.drawRightString(A4[0] - 43, 27, str(document.page))


def build_pdf(scene, train, test, trajectory, orbit, out, other=None):
    title = ParagraphStyle("Title", fontName="Helvetica-Bold", fontSize=24, leading=28, textColor=INK, spaceAfter=13)
    head = ParagraphStyle("Heading", fontName="Helvetica-Bold", fontSize=14, leading=18, textColor=INK,
                          spaceBefore=10, spaceAfter=6)
    body = ParagraphStyle("Body", fontName="Helvetica", fontSize=10, leading=14, textColor=INK, spaceAfter=7)
    doc = SimpleDocTemplate(str(out / "report.pdf"), pagesize=A4, leftMargin=43, rightMargin=43,
                            topMargin=40, bottomMargin=52, title="NeRF reconstruction of the Buddha scene",
                            author="fcxuu / computer-vision-projects")
    meta, settings, mean = scene["registration"], train["config"], test["mean"]
    counts = meta["split_counts"]
    ordered = sorted(test["frames"], key=lambda row: row["psnr"])
    rows = [["Held-out view", "MSE", "PSNR (dB)", "SSIM"]]
    for row in (*ordered, {"id": "Mean", **mean}):
        rows.append([str(row["id"]), f"{row['mse']:.6f}", f"{row['psnr']:.2f}", f"{row['ssim']:.4f}"])
    metrics = Table(rows, colWidths=[168, 102, 102, 102], hAlign="LEFT", repeatRows=1)
    metrics.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#edf4f6"), colors.white]),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
    ]))
    track_note = (
        f"Test trajectories were recomputed for {len(trajectory)} saved checkpoints after training and validation "
        "selection finished. The selected checkpoint is marked separately."
        if trajectory else "The held-out set was evaluated on the validation-selected checkpoint after training."
    )
    if settings.get("focus_share", 0):
        sampler = (f"Fine-tuning started from the uniform run's step {settings['initialization_step']:,}. "
                   f"It draws {settings['focus_share']:.0%} of each batch from rays intersecting a sphere of "
                   f"radius {settings['focus_radius']:g} centered on the calibrated object. "
                   "The remaining draws are uniform across training pixels.")
    else:
        sampler = "Training rays are drawn uniformly across training pixels."
    if other:
        names = {"uniform": "uniform sampling", "focus": "object-focused fine-tuning"}
        mode = "focus" if settings.get("focus_share", 0) else "uniform"
        alternative = "uniform" if mode == "focus" else "focus"
        sampler += (f" The highest full-view validation PSNR was {train['best_val_psnr']:.2f} dB with {names[mode]} "
                    f"and {other['best_val_psnr']:.2f} dB with {names[alternative]}.")
        if "test_mean" in other:
            sampler += (f" The other checkpoint reached {other['test_mean']['psnr']:.2f} dB and "
                        f"{other['test_mean']['ssim']:.4f} SSIM on the same test views.")
    story = [
        Paragraph("NeRF reconstruction of the Buddha scene", title),
        Paragraph(f"The validation-selected model reached <b>{mean['psnr']:.2f} dB PSNR</b> and "
                  f"<b>{mean['ssim']:.4f} SSIM</b> averaged over {len(ordered)} held-out photographs.", body),
        Paragraph("Source images and camera estimation", head),
        Paragraph(f"The Buddha Dataset contains real views of a tabletop sculpture. COLMAP estimated intrinsics and "
                  f"poses independently from the photographs. It registered {meta['registered_frames']} of "
                  f"{meta['source_frames']} images and reconstructed {meta['points3D']:,} sparse points with mean "
                  f"reprojection error {meta['mean_reprojection_error']:.3f} pixels. The images were undistorted "
                  f"and resized to {scene['width']} x {scene['height']}; camera intrinsics were rescaled to match.", body),
        Paragraph(f"Sorted registered filenames determine the fixed photometric split. Every eighth image starting "
                  f"at index zero is test; index four in each block is validation. The result is {counts['train']} train, "
                  f"{counts['val']} validation and {counts['test']} test views. Camera registration uses all photos. "
                  "Only training views supply the RGB targets used to fit the radiance field.", body),
        Paragraph("Radiance field and validation selection", head),
        Paragraph(f"The field maps positional encodings of 3D locations and viewing directions to density and color. "
                  f"Volume rendering combines samples along each ray. This run used {settings['layers']} layers, "
                  f"width {settings['width']}, encoding frequencies {settings['frequencies_xyz']} for position and "
                  f"{settings['frequencies_dir']} for direction, with {settings['n_coarse']} coarse and "
                  f"{settings['n_fine']} fine samples per ray. Near and far bounds are "
                  f"{scene['near']:.3f} and {scene['far']:.3f} in normalized scene coordinates.", body),
        Paragraph(f"The run used {settings['steps']:,} optimizer steps with {settings['rays']:,} rays per step, "
                  f"initial learning rate {settings['lr']:g}, seed {settings['seed']} and device "
                  f"{escape(str(settings['device']))}. Training took {train['seconds'] / 60:.1f} minutes. "
                  f"The highest validation PSNR was {train['best_val_psnr']:.2f} dB at step {train['best_step']:,}. "
                  f"{track_note}", body),
        Paragraph(sampler, body),
        Paragraph(f'Gasparini, S., Castan, F., and Lanthony, Y. (2017). Buddha Dataset, version 1.0, Zenodo. '
                  f'<link href="{DOI}" color="{TEAL}"><u>DOI 10.5281/zenodo.1203283</u></link>. '
                  f'Source photographs shown here are undistorted and resized and remain attributed under '
                  f'<link href="https://creativecommons.org/licenses/by/4.0/" color="{TEAL}"><u>CC BY 4.0</u></link>.',
                  body),
        Paragraph(f'<link href="{CODE}" color="{TEAL}"><u>Reproduction code</u></link> &nbsp; '
                  f'<link href="{FRAMEWORK}" color="{TEAL}"><u>Third-party NeRF implementation</u></link> &nbsp; '
                  f'<link href="{RELEASE}/buddha-nerf.pt" color="{TEAL}"><u>Model checkpoint</u></link> &nbsp; '
                  f'<link href="{RELEASE}/buddha-orbit.mp4" color="{TEAL}"><u>Orbit video</u></link> &nbsp; '
                  f'<link href="{RELEASE}/buddha-scene.zip" color="{TEAL}"><u>Prepared scene</u></link>', body),
        PageBreak(),
        Paragraph("Learning curves and held-out metrics", title),
        figure(out / "training-curves.png", 500, 194),
        Paragraph("Training traces use sampled rays. Validation and test traces use full images from their fixed splits. "
                  "The vertical line marks validation selection. Test values at the selected checkpoint are "
                  "arithmetic means of per-image metrics.", body),
        metrics,
        PageBreak(),
        Paragraph("Rendered views and camera orbit", title),
        figure(out / "test-comparison.png", 500, 316),
        Paragraph("The held-out comparison pairs renders with the source photographs. "
                  f"The lowest test PSNR was recorded for {escape(str(ordered[0]['id']))} "
                  f"({ordered[0]['psnr']:.2f} dB). The large-scale shape is visible. Close-ups blur small "
                  "surface relief and some background regions show streaks.", body),
        figure(out / "orbit-preview.png", 500, 255),
    ]
    if orbit:
        frames = len(orbit["frames"])
        story.append(Paragraph(f"The rendered orbit contains {frames} views at {orbit['width']} x {orbit['height']} "
                               f"pixels, encoded at {orbit['fps']:g} fps for {frames / orbit['fps']:.1f} seconds.", body))
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="Directory containing test and orbit render evidence")
    parser.add_argument("--other-run", type=Path, help="Optional comparison training directory")
    args = parser.parse_args()
    scene = read_json(args.scene)
    train, history, val, test, trajectory, orbit = verify(scene, args.run, args.out)
    plot_curves(history, val, test, trajectory, train["best_step"], args.out)
    other = read_json(args.other_run / "train.json") if args.other_run else None
    if other and (
        other["scene_digest"] != train["scene_digest"]
        or bool(other["config"].get("focus_share")) == bool(train["config"].get("focus_share"))
        or other["best_val_psnr"] > train["best_val_psnr"]
    ):
        raise ValueError("The comparison must use the same scene and agree with validation selection")
    comparison = args.out / "comparison" / "test-metrics.json"
    if other and comparison.exists():
        scored = read_json(comparison)
        if scored["step"] != other["best_step"] or {row["id"] for row in scored["frames"]} != {row["id"] for row in test["frames"]}:
            raise ValueError("Comparison evaluation does not match its checkpoint or test frames")
        other["test_mean"] = scored["mean"]
    build_pdf(scene, train, test, trajectory, orbit, args.out, other)


if __name__ == "__main__":
    main()
