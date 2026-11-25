"""Evaluate registered held-out views and render a closed NeRF orbit."""

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import structural_similarity
from torch.utils.tensorboard import SummaryWriter

from field import Scene, camera_rays, device_for, load_field

ATTRIBUTION = "Source: Buddha Dataset (CC BY 4.0)"
METRICS = ("mse", "psnr", "ssim")


def _write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _render(field, scene, frame, device):
    origins, directions = camera_rays(
        scene.meta["height"], scene.meta["width"],
        frame["intrinsics"], frame["camera_to_world"], device,
    )
    with torch.inference_mode():
        pixels, _ = field.render(origins, directions, chunk=1024)
    pixels = pixels.detach().float().cpu().numpy()
    if not np.isfinite(pixels).all():
        raise ValueError("Rendered image contains non-finite RGB values")
    return pixels.clip(0.0, 1.0)


def _test_checkpoint(checkpoint, vendor, device, scene, keep_examples=False):
    renderer, state = load_field(checkpoint, vendor, device, scene)
    test_frames = scene.frames("test")
    if not test_frames:
        raise ValueError("No registered held-out test views")
    examples, rows = [], []
    example_indices = set(np.linspace(0, len(test_frames) - 1, min(4, len(test_frames)), dtype=int))
    for index, frame in enumerate(test_frames):
        photo = scene.rgb(frame)
        pixels = _render(renderer, scene, frame, device)
        mse = float(np.mean((photo - pixels) ** 2, dtype=np.float64))
        # Floor MSE only for PSNR so mathematically perfect matches remain JSON-safe.
        psnr = float(-10 * np.log10(max(mse, 1e-12)))
        ssim = float(structural_similarity(photo, pixels, data_range=1.0, channel_axis=-1))
        frame_id = str(frame.get("id", frame.get("name", frame.get("file_path", index))))
        rows.append({"id": frame_id, "mse": mse, "psnr": psnr, "ssim": ssim})
        if keep_examples and index in example_indices:
            examples.append((frame_id, photo, pixels, psnr))
    result = {
        "checkpoint": str(checkpoint),
        "step": int(state["step"]),
        "mean": {name: float(np.mean([row[name] for row in rows])) for name in METRICS},
        "frames": rows,
    }
    return result, examples


def _test(args, scene, device):
    result, examples = _test_checkpoint(args.checkpoint, args.vendor, device, scene, keep_examples=True)
    _write_json(args.out / "test-metrics.json", result)
    figure, axes = plt.subplots(2, len(examples), squeeze=False, figsize=(3 * len(examples), 6))
    for index, (frame_id, photo, pixels, psnr) in enumerate(examples):
        axes[0, index].imshow(photo)
        axes[0, index].set_title(f"Photo {frame_id}")
        axes[1, index].imshow(pixels)
        axes[1, index].set_title(f"Render · {psnr:.2f} dB")
    for axis in axes.flat:
        axis.axis("off")
    figure.supxlabel(ATTRIBUTION, fontsize=9)
    figure.tight_layout()
    figure.savefig(args.out / "test-comparison.png", dpi=160, bbox_inches="tight")
    plt.close(figure)

    history = {result["step"]: result["mean"]}
    if args.trajectory:
        checkpoints = {path.resolve(): path for path in args.trajectory}
        checkpoints.pop(args.checkpoint.resolve(), None)
        for checkpoint in checkpoints.values():
            previous, _ = _test_checkpoint(checkpoint, args.vendor, device, scene)
            history.setdefault(previous["step"], previous["mean"])
        with (args.out / "trajectory.csv").open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=("step", *METRICS), lineterminator="\n")
            writer.writeheader()
            for step, values in sorted(history.items()):
                writer.writerow({"step": step, **values})
    with SummaryWriter(str(args.out / "tensorboard")) as writer:
        for step, values in sorted(history.items()):
            for name, value in values.items():
                writer.add_scalar(f"test/{name}", value, step)
    print(json.dumps(result["mean"]))


def _unit(vector):
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        raise ValueError("Degenerate orbit camera geometry")
    return vector / norm


def _orbit_up(poses):
    positions = poses[:, :3, 3]
    centroid = positions.mean(axis=0)
    _, _, directions = np.linalg.svd(positions - centroid, full_matrices=False)
    up = directions[-1]
    if up @ centroid < 0:
        up = -up

    # The image-up axis depends on whether the source photo was turned by 90 degrees.
    # Pick the best quarter-turn for each camera against the current world-up estimate.
    x, y = poses[:, :3, 0], poses[:, :3, 1]
    image_axes = np.stack((-y, x, y, -x), axis=1)
    for _ in range(4):
        nearest = np.argmax(image_axes @ up, axis=1)
        up = _unit(image_axes[np.arange(len(poses)), nearest].mean(axis=0))
    return up


def _orbit_frames(scene, count):
    frames = scene.frames("train")
    if len(frames) < 3:
        raise ValueError("At least three registered training views are required for the orbit")
    poses = np.array([frame["camera_to_world"] for frame in frames], dtype=np.float64)
    positions = poses[:, :3, 3]
    up = _orbit_up(poses)
    heights = positions @ up
    horizontal = positions - heights[:, None] * up
    radii = np.linalg.norm(horizontal, axis=1)
    radius = float(np.median(radii))
    height = float(np.median(heights))
    if radius < 1e-8:
        raise ValueError("Orbit cameras have no horizontal baseline")
    start = int(np.argmin(np.abs(radii - radius) + np.abs(heights - height)))
    axis_x = _unit(horizontal[start])
    axis_y = _unit(np.cross(up, axis_x))
    fx, fy, _, _ = frames[start]["intrinsics"]
    # A wider, centered lens leaves space around the sculpture throughout the orbit.
    intrinsics = [0.80 * fx, 0.80 * fy, scene.meta["width"] / 2, scene.meta["height"] / 2]
    result = []
    for angle in np.linspace(0, 2 * np.pi, count, endpoint=False):
        origin = height * up + radius * (np.cos(angle) * axis_x + np.sin(angle) * axis_y)
        optical = _unit(-origin)
        right = _unit(np.cross(optical, up))
        down = _unit(np.cross(optical, right))
        matrix = np.eye(4)
        matrix[:3, :3] = np.column_stack((right, down, optical))
        matrix[:3, 3] = origin
        result.append({"camera_to_world": matrix.tolist(), "intrinsics": intrinsics})
    return result


def _verify_video(probe, path, frames, fps):
    info = json.loads(subprocess.check_output([
        probe, "-v", "error", "-count_frames", "-show_streams", "-show_format",
        "-of", "json", str(path),
    ], text=True))
    streams = [stream for stream in info.get("streams", []) if stream["codec_type"] == "video"]
    if len(streams) != 1:
        raise RuntimeError("Expected exactly one video stream")
    stream = streams[0]
    numerator, denominator = map(int, stream["avg_frame_rate"].split("/"))
    if (
        stream["codec_name"] != "h264" or stream["pix_fmt"] != "yuv420p"
        or int(stream["nb_read_frames"]) != frames or numerator != fps * denominator
    ):
        raise RuntimeError(f"Unexpected ffprobe result: {stream}")
    return info


def _orbit(args, scene, device):
    if args.frames < 3 or args.fps < 1:
        raise ValueError("Orbit requires at least three frames and positive fps")
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("ffmpeg and ffprobe must be available on PATH")
    renderer, _ = load_field(args.checkpoint, args.vendor, device, scene)
    frames = _orbit_frames(scene, args.frames)
    height, width = scene.meta["height"], scene.meta["width"]
    preview = []
    preview_indices = set(np.linspace(0, len(frames), min(4, len(frames)), endpoint=False, dtype=int))
    with TemporaryDirectory(prefix=".orbit-", dir=args.out) as folder, TemporaryFile() as errors:
        staged = Path(folder) / "orbit.mp4"
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
            "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
            "-framerate", str(args.fps), "-i", "pipe:0",
            "-an", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-metadata", f"comment={ATTRIBUTION}",
            str(staged),
        ]
        with subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors) as encoder:
            try:
                for index, frame in enumerate(frames):
                    rgb = np.rint(_render(renderer, scene, frame, device) * 255).astype(np.uint8)
                    encoder.stdin.write(rgb.tobytes())
                    if index in preview_indices:
                        preview.append(rgb)
                encoder.stdin.close()
            except BrokenPipeError as error:
                encoder.wait()
                errors.seek(0)
                raise RuntimeError(errors.read().decode(errors="replace")) from error
            except BaseException:
                encoder.kill()
                raise
            if encoder.wait():
                errors.seek(0)
                raise RuntimeError(errors.read().decode(errors="replace"))
        info = _verify_video(ffprobe, staged, args.frames, args.fps)
        final = args.out / "orbit.mp4"
        staged.replace(final)
    info["format"]["filename"] = str(final)
    _write_json(args.out / "orbit-ffprobe.json", info)
    _write_json(args.out / "orbit-trajectory.json", {
        "width": width, "height": height, "fps": args.fps, "frames": frames,
    })
    figure, axes = plt.subplots(1, len(preview), squeeze=False, figsize=(3 * len(preview), 3))
    for axis, image in zip(axes.flat, preview):
        axis.imshow(image)
        axis.axis("off")
    figure.supxlabel(ATTRIBUTION, fontsize=9)
    figure.tight_layout()
    figure.savefig(args.out / "orbit-preview.png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(final)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    commands = parser.add_subparsers(dest="command", required=True)
    test = commands.add_parser("test")
    orbit = commands.add_parser("orbit")
    for command in (test, orbit):
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
    test.add_argument("--trajectory", nargs="+", type=Path, help="Replay checkpoints on the held-out set for reporting")
    orbit.add_argument("--frames", type=int, default=90)
    orbit.add_argument("--fps", type=int, default=18)
    args = parser.parse_args()
    device = device_for(args.device)
    scene = Scene(args.scene, device)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "test":
        _test(args, scene, device)
    else:
        _orbit(args, scene, device)


if __name__ == "__main__":
    main()
