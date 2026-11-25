"""Read the Caltech CUB-200-2011 release and partition its training images."""

import hashlib
import random
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from PIL import Image, ImageOps


DATASET_NAME = "CUB_200_2011"
ARCHIVE_MD5 = "97eceeb196236b17998738112f37df78"
NUM_IMAGES = 11_788
NUM_CLASSES = 200
NUM_TRAIN = 5_994
NUM_TEST = 5_794


@dataclass(frozen=True)
class Sample:
    id: int
    path: Path
    label: int
    official_train: bool


def _read_table(path: Path) -> dict[int, str]:
    table: dict[int, str] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            raw_id, value = line.split(maxsplit=1)
            image_id = int(raw_id)
            if image_id in table:
                raise ValueError(f"Duplicate image ID {image_id} in {path}")
            table[image_id] = value.strip()
    if table.keys() != set(range(1, NUM_IMAGES + 1)):
        raise ValueError(f"Unexpected set of image IDs in {path}")
    return table


def read_samples(folder: Path) -> list[Sample]:
    """Validate metadata and return images in official image ID order."""
    paths = _read_table(folder / "images.txt")
    labels = _read_table(folder / "image_class_labels.txt")
    flags = _read_table(folder / "train_test_split.txt")
    class_ids = {int(label) for label in labels.values()}
    if class_ids != set(range(1, NUM_CLASSES + 1)):
        raise ValueError("Unexpected set of CUB class IDs")
    if set(flags.values()) != {"0", "1"}:
        raise ValueError("CUB split flags must be 0 or 1")
    train_count = sum(flag == "1" for flag in flags.values())
    if (train_count, len(flags) - train_count) != (NUM_TRAIN, NUM_TEST):
        raise ValueError("Unexpected sizes for the official CUB splits")
    if len(set(paths.values())) != NUM_IMAGES:
        raise ValueError("Duplicate image paths in images.txt")

    image_folder = folder / "images"
    samples = []
    for image_id in range(1, NUM_IMAGES + 1):
        relative_path = Path(paths[image_id])
        if not relative_path.parts or relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Invalid relative image path for ID {image_id}: {relative_path}")
        image_path = image_folder / relative_path
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        samples.append(Sample(image_id, image_path, int(labels[image_id]) - 1, flags[image_id] == "1"))
    return samples


def prepare(root: Path) -> Path:
    """Use a valid extracted tree or verify and extract the local official archive."""
    folder = root if root.name == DATASET_NAME else root / DATASET_NAME
    if folder.exists():
        read_samples(folder)
        return folder

    archive = folder.with_suffix(".tgz")
    with archive.open("rb") as source:
        digest = hashlib.file_digest(source, "md5").hexdigest()
        if digest != ARCHIVE_MD5:
            raise ValueError(f"CUB archive checksum mismatch for {archive}: {digest}")
        source.seek(0)
        with TemporaryDirectory(prefix=".cub-extract-", dir=folder.parent) as staging:
            with tarfile.open(fileobj=source, mode="r:gz") as package:
                package.extractall(staging, filter="data")
            extracted = Path(staging) / DATASET_NAME
            read_samples(extracted)
            try:
                extracted.rename(folder)
            except OSError:
                if not folder.is_dir():
                    raise
                read_samples(folder)
    return folder


def split_training(samples: list[Sample], seed: int) -> tuple[list[Sample], list[Sample]]:
    """Reserve roughly 15 percent of each class from the official training split."""
    per_class: dict[int, list[Sample]] = defaultdict(list)
    for sample in samples:
        if sample.official_train:
            per_class[sample.label].append(sample)

    randomizer = random.Random(seed)
    labels = sorted(per_class)
    quotas = {label: 0.15 * len(per_class[label]) for label in labels}
    validation_counts = {
        label: min(len(per_class[label]) - 1, max(1, int(quotas[label])))
        for label in labels
    }
    remainder = max(0, round(sum(quotas.values())) - sum(validation_counts.values()))
    eligible = [label for label in labels if validation_counts[label] < len(per_class[label]) - 1]
    eligible.sort(key=lambda label: (quotas[label] - validation_counts[label], randomizer.random()), reverse=True)
    for label in eligible[:remainder]:
        validation_counts[label] += 1
    train, val = [], []
    for label in labels:
        class_samples = sorted(per_class[label], key=lambda sample: sample.id)
        randomizer.shuffle(class_samples)
        val_count = validation_counts[label]
        val.extend(class_samples[:val_count])
        train.extend(class_samples[val_count:])
    train.sort(key=lambda sample: sample.id)
    val.sort(key=lambda sample: sample.id)
    return train, val


def image_tensors(
    samples: list[Sample], size: int, resize_mode: str = "square"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return EXIF-corrected RGB images as square uint8 CPU tensors and class IDs."""
    if size <= 0:
        raise ValueError("Image size must be positive")
    if resize_mode not in {"square", "letterbox"}:
        raise ValueError(f"Unknown image resize mode: {resize_mode!r}")
    images = torch.empty((len(samples), 3, size, size), dtype=torch.uint8, device="cpu")
    labels = torch.tensor([sample.label for sample in samples], dtype=torch.int64, device="cpu")
    for index, sample in enumerate(samples):
        with Image.open(sample.path) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            if resize_mode == "letterbox":
                factor = size / max(rgb.size)
                width, height = (max(1, round(edge * factor)) for edge in rgb.size)
                content = rgb.resize((width, height), Image.Resampling.BILINEAR)
                resized = Image.new("RGB", (size, size), (124, 116, 104))
                resized.paste(content, ((size - width) // 2, (size - height) // 2))
            else:
                resized = rgb.resize((size, size), Image.Resampling.BILINEAR)
            images[index].copy_(torch.from_numpy(np.array(resized)).permute(2, 0, 1))
    return images, labels
