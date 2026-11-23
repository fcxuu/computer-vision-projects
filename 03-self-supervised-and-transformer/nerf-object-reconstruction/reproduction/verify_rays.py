"""Verify COLMAP camera conventions independently on CPU and MPS."""
import argparse
from pathlib import Path

import numpy as np
import torch

from field import Scene, camera_rays


def check_rays(device):
    pose = torch.eye(4)
    pose[:3, :3] = torch.tensor([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    pose[:3, 3] = torch.tensor([2., -3., 4.])
    intrinsics = [2., 4., 1.5, 1.5]
    origins, rays = camera_rays(3, 3, intrinsics, pose, device)
    expected = torch.tensor([
        [1., -0.25, 0.5], [1., -0.25, 0.], [1., -0.25, -0.5],
        [1., 0., 0.5], [1., 0., 0.], [1., 0., -0.5],
        [1., 0.25, 0.5], [1., 0.25, 0.], [1., 0.25, -0.5],
    ]).reshape(3, 3, 3)
    expected /= expected.norm(dim=-1, keepdim=True)
    torch.testing.assert_close(rays.cpu(), expected)
    torch.testing.assert_close(origins.cpu(), pose[:3, 3].expand_as(origins))
    print(f"{device.type}: transformed origins and calibrated unit world rays verified")


def check_scene(path):
    scene = Scene(path, torch.device("cpu"))
    groups = [{frame["name"] for frame in scene.frames(split)} for split in ("train", "val", "test")]
    assert len(set.union(*groups)) == sum(map(len, groups))
    frame = scene.frames("train")[0]
    pose = np.asarray(frame["camera_to_world"])
    rays = scene.rays(frame)[1].numpy()
    expected = pose[:3, :3][:, 2]
    fx, fy, cx, cy = frame["intrinsics"]
    x, y = int(cx), int(cy)
    assert 0 <= x < scene.width and 0 <= y < scene.height
    assert np.dot(expected, rays[y, x]) > 1 - (1 / fx**2 + 1 / fy**2)
    print(f"Scene cameras and disjoint splits verified: {'/'.join(str(len(group)) for group in groups)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path)
    args = parser.parse_args()
    check_rays(torch.device("cpu"))
    if torch.backends.mps.is_available():
        check_rays(torch.device("mps"))
    if args.scene:
        check_scene(args.scene)
