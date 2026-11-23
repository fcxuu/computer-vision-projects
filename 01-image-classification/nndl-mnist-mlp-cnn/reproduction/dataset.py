"""MNIST IDX loading, fixed stratification, and train-only affine augmentation."""
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import affine_transform

# Both MNIST datasets have the same IDX wire format.
IDX_TOOLS = Path(__file__).resolve().parents[2] / "fashion-mnist-3layer" / "reproduction"
sys.path.insert(0, str(IDX_TOOLS))
try:
    from data_io import read_idx
finally:
    sys.path.pop(0)

BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist"
CHECKSUMS = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def load(root, split):
    count = {"train": 60_000, "t10k": 10_000}[split]
    return read_idx(root, split, count, BASE_URL, CHECKSUMS, 10)


def split_indices(labels, seed):
    """Keep close to a 1/6 validation fraction in every class."""
    rng = np.random.default_rng(seed)
    quotas = np.bincount(labels, minlength=10) / 6
    counts = quotas.astype(int)
    remainder = 10_000 - counts.sum()
    counts[np.argsort(-(quotas - counts), kind="stable")[:remainder]] += 1
    training, validation = [], []
    for category, count in enumerate(counts):
        indices = rng.permutation(np.flatnonzero(labels == category))
        validation.append(indices[:count])
        training.append(indices[count:])
    return rng.permutation(np.concatenate(training)), rng.permutation(np.concatenate(validation))


def augment(images, rng):
    result = images.copy().reshape(-1, 28, 28)
    rows = np.flatnonzero(rng.random(len(images)) < 0.5)
    angles = np.deg2rad(rng.uniform(-15, 15, len(rows)))
    scales = rng.uniform(0.9, 1.1, len(rows))
    offsets = rng.uniform(-2, 2, (len(rows), 2))
    center = np.array([13.5, 13.5])
    for row, angle, scale, shift in zip(rows, angles, scales, offsets):
        cosine, sine = np.cos(angle), np.sin(angle)
        inverse = np.array([[cosine, sine], [-sine, cosine]]) / scale
        result[row] = affine_transform(
            result[row], inverse, offset=center - inverse @ (center + shift),
            order=1, mode="constant", cval=0.0, prefilter=False,
        )
    return result.reshape(-1, 784)
