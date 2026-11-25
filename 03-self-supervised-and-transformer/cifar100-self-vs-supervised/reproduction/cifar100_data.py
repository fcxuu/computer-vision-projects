"""Official CIFAR-100 partitions and batched on-device image views."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.datasets import CIFAR100

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)
_LUMA = (0.299, 0.587, 0.114)


@dataclass(frozen=True)
class Partition:
    images: torch.Tensor
    targets: torch.Tensor

    def batches(self, batch: int, rng: torch.Generator | None = None, indices=None):
        """Iterate over local indices, keeping a possibly incomplete final batch."""
        if batch <= 0:
            raise ValueError("Batch size must be positive")
        if indices is None:
            order = torch.arange(len(self.targets), device="cpu")
        else:
            order = torch.as_tensor(indices, dtype=torch.int64, device="cpu")
        if rng is not None:
            order = order[torch.randperm(len(order), generator=rng, device="cpu")]
        for selection in order.split(batch):
            if selection.numel():
                local = selection.to(self.images.device)
                yield self.images.index_select(0, local), self.targets.index_select(0, local)


def _partition(data: np.ndarray, targets: np.ndarray, device) -> Partition:
    return Partition(
        torch.from_numpy(np.ascontiguousarray(data)).to(device),
        torch.from_numpy(np.ascontiguousarray(targets, dtype=np.int64)).to(device),
    )


def training_partitions(root: Path, seed: int, device) -> tuple[Partition, Partition, dict]:
    """Reserve exactly 50 official training images from each CIFAR-100 class."""
    dataset = CIFAR100(root=root, train=True, download=True)
    targets = np.asarray(dataset.targets, dtype=np.int64)
    if len(targets) != 50_000 or not np.array_equal(np.bincount(targets), np.full(100, 500)):
        raise ValueError("Unexpected CIFAR-100 training counts")
    rng = np.random.default_rng(seed)
    validation_mask = np.zeros(len(targets), dtype=bool)
    for label in range(100):
        candidates = np.flatnonzero(targets == label)
        validation_mask[rng.choice(candidates, size=50, replace=False)] = True
    split = {"train": np.flatnonzero(~validation_mask), "validation": np.flatnonzero(validation_mask)}
    train = _partition(dataset.data[split["train"]], targets[split["train"]], device)
    validation = _partition(dataset.data[split["validation"]], targets[split["validation"]], device)
    return train, validation, split


def test_partition(root: Path, device) -> Partition:
    """Read the official held-out CIFAR-100 test split."""
    dataset = CIFAR100(root=root, train=False, download=False)
    targets = np.asarray(dataset.targets, dtype=np.int64)
    if len(targets) != 10_000 or not np.array_equal(np.bincount(targets), np.full(100, 100)):
        raise ValueError("Unexpected CIFAR-100 test counts")
    return _partition(dataset.data, targets, device)


def _cropped_views(images: torch.Tensor, size: int, draws: torch.Tensor, strong: bool) -> torch.Tensor:
    # A square crop samples area uniformly and then chooses a random position.
    min_area = 0.20 if strong else 0.75
    extent = (min_area + (1.0 - min_area) * draws[:, 0]).sqrt()
    offset_x = (2 * draws[:, 1] - 1) * (1 - extent)
    offset_y = (2 * draws[:, 2] - 1) * (1 - extent)
    direction = torch.where(draws[:, 3] < 0.5, -1.0, 1.0)
    axis = (torch.arange(size, dtype=torch.float32, device=images.device) + 0.5) * (2 / size) - 1
    x = axis[None, :] * (extent * direction)[:, None] + offset_x[:, None]
    y = axis[None, :] * extent[:, None] + offset_y[:, None]
    # Clamp to pixel centers for edge interpolation on MPS using zeros padding.
    height, width = images.shape[-2:]
    x = x.clamp(-1 + 1 / width, 1 - 1 / width)
    y = y.clamp(-1 + 1 / height, 1 - 1 / height)
    grid = torch.stack((x[:, None, :].expand(-1, size, -1), y[:, :, None].expand(-1, -1, size)), dim=-1)
    return F.grid_sample(images, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def _color_views(images: torch.Tensor, draws: torch.Tensor) -> torch.Tensor:
    luma = images.new_tensor(_LUMA).view(1, 3, 1, 1)
    brightness = (0.6 + 0.8 * draws[:, 5]).view(-1, 1, 1, 1)
    contrast = (0.6 + 0.8 * draws[:, 6]).view(-1, 1, 1, 1)
    saturation = (0.6 + 0.8 * draws[:, 7]).view(-1, 1, 1, 1)
    jittered = (images * brightness).clamp(0, 1)
    gray_mean = (jittered * luma).sum(dim=1, keepdim=True).mean(dim=(2, 3), keepdim=True)
    jittered = (gray_mean + contrast * (jittered - gray_mean)).clamp(0, 1)
    gray = (jittered * luma).sum(dim=1, keepdim=True)
    jittered = (gray + saturation * (jittered - gray)).clamp(0, 1)
    images = torch.where((draws[:, 4] < 0.8).view(-1, 1, 1, 1), jittered, images)
    gray = (images * luma).sum(dim=1, keepdim=True)
    return torch.where((draws[:, 8] < 0.2).view(-1, 1, 1, 1), gray, images)


def transform(pixels: torch.Tensor, size: int, rng: torch.Generator | None = None, strong: bool = False) -> torch.Tensor:
    """Normalize NHWC uint8 inputs and optionally sample an independent view."""
    if size not in (32, 64):
        raise ValueError("CIFAR-100 target size must be 32 or 64")
    images = pixels.permute(0, 3, 1, 2).contiguous().to(dtype=torch.float32).div_(255)
    if rng is not None and len(images):
        draws = torch.rand((len(images), 9), generator=rng, device="cpu", dtype=torch.float32).to(images.device)
        images = _cropped_views(images, size, draws, strong)
        if strong:
            images = _color_views(images, draws)
    elif images.shape[-2:] != (size, size):
        images = F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)
    mean = images.new_tensor(_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(_STD).view(1, 3, 1, 1)
    return (images - mean) / std
