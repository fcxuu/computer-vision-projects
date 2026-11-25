"""Calibrated COLMAP rays and a pinned third-party NeRF field."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
VENDOR = ROOT / "data" / "vendor" / "nerf"
VENDOR_COMMIT = "9c47e83c5a2c9736c9e770cc993237ce3e2667ed"


def device_for(name):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


class Scene:
    def __init__(self, scene_json, device):
        path = Path(scene_json).resolve()
        raw = path.read_bytes()
        self.digest = hashlib.sha256(raw).hexdigest()
        self.root = path.parent
        self.meta = json.loads(raw)
        self.device = device
        self.height, self.width = (int(self.meta[key]) for key in ("height", "width"))
        self.near, self.far = (float(self.meta[key]) for key in ("near", "far"))
        if min(self.height, self.width) < 8 or not (0 < self.near < self.far < float("inf")):
            raise ValueError("Invalid scene image dimensions or metric ray bounds")
        names = set()
        for frame in self.meta["frames"]:
            if frame["split"] not in ("train", "val", "test") or frame["name"] in names:
                raise ValueError(f"Duplicate frame or unknown split: {frame['name']}")
            names.add(frame["name"])
            pose = np.asarray(frame["camera_to_world"], dtype=np.float64)
            intrinsics = np.asarray(frame["intrinsics"], dtype=np.float64)
            if (
                pose.shape != (4, 4) or not np.isfinite(pose).all()
                or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6)
                or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4)
                or np.linalg.det(pose[:3, :3]) < 0.99
                or intrinsics.shape != (4,) or not np.isfinite(intrinsics).all()
                or (intrinsics[:2] <= 0).any()
            ):
                raise ValueError(f"Invalid camera pose or intrinsics: {frame['name']}")
            image_path = (self.root / frame["file_path"]).resolve()
            if not image_path.is_relative_to(self.root) or not image_path.is_file():
                raise ValueError(f"Missing scene image: {frame['file_path']}")
        if any(not self.frames(split) for split in ("train", "val", "test")):
            raise ValueError("Scene must have disjoint nonempty train, validation, and test view sets")

    def frames(self, split):
        return [frame for frame in self.meta["frames"] if frame["split"] == split]

    def rgb(self, frame):
        with Image.open(self.root / frame["file_path"]) as image:
            if image.size != (self.width, self.height):
                raise ValueError(f"Image dimensions disagree with camera intrinsics: {frame['name']}")
            return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0

    def rays(self, frame):
        return camera_rays(self.height, self.width, frame["intrinsics"], frame["camera_to_world"], self.device)


def camera_rays(height, width, intrinsics, camera_to_world, device):
    """Generate metric world rays using COLMAP's right-down-forward camera coordinates."""
    fx, fy, cx, cy = map(float, intrinsics)
    pose = torch.as_tensor(camera_to_world, dtype=torch.float32, device=device)
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device) + 0.5,
        torch.arange(width, dtype=torch.float32, device=device) + 0.5,
        indexing="ij",
    )
    camera = torch.stack(((x - cx) / fx, (y - cy) / fy, torch.ones_like(x)), dim=-1)
    directions = F.normalize(camera @ pose[:3, :3].T, dim=-1)
    return pose[:3, 3].expand_as(directions), directions


def framework(vendor):
    vendor = Path(vendor).resolve()
    commit = subprocess.check_output(["git", "-C", str(vendor), "rev-parse", "HEAD"], text=True).strip()
    if commit != VENDOR_COMMIT:
        raise ValueError(f"Check out NeRF framework commit {VENDOR_COMMIT}")
    sys.path.insert(0, str(vendor))
    try:
        from src.models import NeRF
        from src.rendering import render_hierarchical, render_image
    finally:
        sys.path.pop(0)
    return NeRF, render_hierarchical, render_image


class Field(nn.Module):
    def __init__(self, vendor, config, scene):
        super().__init__()
        model, self.renderer, self.image_renderer = framework(vendor)
        options = {
            "num_frequencies_xyz": config["frequencies_xyz"],
            "num_frequencies_dir": config["frequencies_dir"],
            "hidden_dim": config["width"],
            "num_layers": config["layers"],
            "skip_connection_layer": (config["layers"] - 1) // 2,
        }
        self.coarse = model(**options)
        self.fine = model(**options)
        self.bounds = (scene.near, scene.far, config["n_coarse"], config["n_fine"])
        self.to(scene.device)

    def forward(self, origins, directions, perturb=True):
        return self.renderer(self.coarse, self.fine, origins, directions,
                             *self.bounds, perturb=perturb)

    def render(self, origins, directions, chunk=1024):
        return self.image_renderer(self.coarse, self.fine, origins, directions,
                                   *self.bounds, chunk_size=chunk)


def load_field(path, vendor, device, scene):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["scene_digest"] != scene.digest or state["framework_commit"] != VENDOR_COMMIT:
        raise ValueError("Checkpoint scene or NeRF framework does not match")
    if scene.device != device:
        raise ValueError("Scene and requested model device disagree")
    field = Field(vendor, state["config"], scene)
    field.load_state_dict(state["weights"], strict=True)
    field.eval()
    return field, state
