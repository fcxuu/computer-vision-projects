"""CIFAR-10 split and batched random crop, reflection, and normalization."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10

ROOT = Path(__file__).resolve().parents[3]
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


@dataclass
class Partition:
    images: torch.Tensor
    targets: torch.Tensor

    @classmethod
    def from_arrays(cls, images, labels, device):
        return cls(torch.from_numpy(images).to(device), torch.as_tensor(labels).to(device))

    def __len__(self):
        return len(self.targets)

    def batches(self, size, rng=None):
        device = self.images.device
        mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
        order = torch.randperm(len(self), generator=rng) if rng is not None else torch.arange(len(self))
        scale = torch.arange(32, dtype=torch.int32, device=device)
        for rows in order.split(size):
            rows = rows.to(device)
            pixels = self.images[rows]
            if rng is not None:
                count = len(rows)
                offsets = torch.randint(9, (count, 2), generator=rng, dtype=torch.int32).to(device)
                flip = (torch.rand(count, generator=rng) < 0.5).to(device)
                y = offsets[:, 0, None, None] + scale[None, :, None]
                x = offsets[:, 1, None, None] + torch.where(flip[:, None], 31 - scale, scale)[:, None, :]
                padded = F.pad(pixels.permute(0, 3, 1, 2).float(), (4, 4, 4, 4), mode="reflect")
                pixels = padded.permute(0, 2, 3, 1)[torch.arange(count, device=device)[:, None, None], y, x]
            normalized = (pixels.permute(0, 3, 1, 2).float() / 255 - mean) / std
            yield normalized.contiguous(), self.targets[rows]


def device_for(value):
    if value != "auto":
        device = torch.device(value)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    return device


def training_partitions(path, seed, device):
    dataset = CIFAR10(path, train=True, download=True)
    labels = np.asarray(dataset.targets, dtype=np.int64)
    rng = np.random.default_rng(seed)
    train, validation = [], []
    for category in range(10):
        indices = rng.permutation(np.flatnonzero(labels == category))
        validation.append(indices[:500])
        train.append(indices[500:])
    split = {"train": rng.permutation(np.concatenate(train)),
             "validation": rng.permutation(np.concatenate(validation))}
    partitions = [Partition.from_arrays(dataset.data[rows], labels[rows], device) for rows in split.values()]
    return (*partitions, split)


def test_partition(path, device):
    dataset = CIFAR10(path, train=False, download=True)
    return Partition.from_arrays(dataset.data, dataset.targets, device)
