"""Read PASCAL VOC 2007 and score predictions with its 11-point AP metric."""

import hashlib
import tarfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from PIL import Image

CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)
_CLASS_IDS = {name: index for index, name in enumerate(CLASSES)}
_SPLIT_COUNTS = {"train": 2501, "val": 2510, "test": 4952}
_ARCHIVES = {
    "trainval": ("VOCtrainval_06-Nov-2007.tar", "c52e279531787c972589f7e41ab4ae64"),
    "test": ("VOCtest_06-Nov-2007.tar", "b6e924de25625d8de591ea690078ad9f"),
}
_VOC_FOLDER = Path("VOCdevkit", "VOC2007")
Geometry = tuple[float, float, int, int, int, int]


@dataclass(frozen=True)
class Sample:
    id: str
    path: Path
    boxes: np.ndarray
    labels: np.ndarray
    difficult: np.ndarray


def _annotation(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes, labels, difficult = [], [], []
    for obj in ET.parse(path).getroot().findall("object"):
        box = obj.find("bndbox")
        if box is None:
            raise ValueError(f"Missing bounding box in {path}")
        xmin, ymin, xmax, ymax = (
            float(box.findtext(key, "")) for key in ("xmin", "ymin", "xmax", "ymax")
        )
        if not (1 <= xmin <= xmax and 1 <= ymin <= ymax):
            raise ValueError(f"Invalid VOC bounding box in {path}")
        boxes.append((xmin - 1, ymin - 1, xmax, ymax))
        labels.append(_CLASS_IDS[obj.findtext("name", "").strip()])
        flag = obj.findtext("difficult", "").strip()
        if flag not in {"0", "1"}:
            raise ValueError(f"Invalid difficult flag in {path}")
        difficult.append(flag == "1")
    return (
        np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        np.asarray(labels, dtype=np.int64),
        np.asarray(difficult, dtype=np.bool_),
    )


def _read_split(folder: Path, split: str) -> list[Sample]:
    index = folder / "ImageSets" / "Main" / f"{split}.txt"
    ids = index.read_text(encoding="utf-8").split()
    if len(ids) != _SPLIT_COUNTS[split] or len(set(ids)) != len(ids):
        raise ValueError(f"Unexpected VOC2007 {split} split count or duplicate IDs in {index}")
    samples = []
    for image_id in ids:
        if len(image_id) != 6 or not image_id.isascii() or not image_id.isdigit():
            raise ValueError(f"Invalid VOC image ID {image_id!r}")
        path = folder / "JPEGImages" / f"{image_id}.jpg"
        if not path.is_file():
            raise FileNotFoundError(path)
        annotation = _annotation(folder / "Annotations" / f"{image_id}.xml")
        samples.append(Sample(image_id, path, *annotation))
    return samples


def load_split(root: Path, split: str) -> list[Sample]:
    """Read the official split, extracting its verified archive on first use."""
    if split not in _SPLIT_COUNTS:
        raise ValueError(f"Unknown VOC2007 split {split!r}")
    archive_key = "test" if split == "test" else "trainval"
    destination = root / archive_key
    if destination.exists():
        return _read_split(destination / _VOC_FOLDER, split)

    filename, expected_md5 = _ARCHIVES[archive_key]
    with (root / filename).open("rb") as source:
        actual_md5 = hashlib.file_digest(source, "md5").hexdigest()
        if actual_md5 != expected_md5:
            raise ValueError(f"VOC archive checksum mismatch for {filename}: {actual_md5}")
        source.seek(0)
        with TemporaryDirectory(prefix=f".voc-{archive_key}-", dir=root) as temporary:
            staging = Path(temporary) / archive_key
            with tarfile.open(fileobj=source, mode="r:") as archive:
                archive.extractall(staging, filter="data")
            # Check both split indices before publishing a shared trainval tree.
            for release_split in (("test",) if archive_key == "test" else ("train", "val")):
                _read_split(staging / _VOC_FOLDER, release_split)
            try:
                staging.rename(destination)
            except OSError:
                if not destination.is_dir():
                    raise
    return _read_split(destination / _VOC_FOLDER, split)


def letterbox(image: Image.Image, size: int) -> tuple[torch.Tensor, Geometry]:
    """Resize while preserving aspect ratio and pad centrally to a square."""
    if size <= 0:
        raise ValueError("Letterbox size must be positive")
    orig_w, orig_h = image.size
    ratio = size / max(orig_w, orig_h)
    width = max(1, round(orig_w * ratio))
    height = max(1, round(orig_h * ratio))
    pad_x, pad_y = (size - width) // 2, (size - height) // 2
    resized = image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(resized, (pad_x, pad_y))
    tensor = torch.from_numpy(np.array(canvas)).permute(2, 0, 1).contiguous()
    geometry = width / orig_w, height / orig_h, pad_x, pad_y, orig_w, orig_h
    return tensor, geometry


def unletterbox(boxes: np.ndarray, geometry: Geometry) -> np.ndarray:
    """Map square input boxes to original-image pixel coordinates."""
    scale_x, scale_y, pad_x, pad_y, width, height = geometry
    coordinates = np.asarray(boxes, dtype=np.float32).copy().reshape(-1, 4)
    coordinates[:, 0::2] = np.clip((coordinates[:, 0::2] - pad_x) / scale_x, 0, width)
    coordinates[:, 1::2] = np.clip((coordinates[:, 1::2] - pad_y) / scale_y, 0, height)
    return coordinates


class VocDataset(torch.utils.data.Dataset):
    def __init__(self, samples: list[Sample], size: int):
        self.samples = samples
        self.size = size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict]:
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            pixels, geometry = letterbox(image, self.size)
        scale_x, scale_y, pad_x, pad_y, _, _ = geometry
        boxes = sample.boxes.copy()
        boxes[:, 0::2] = boxes[:, 0::2] * scale_x + pad_x
        boxes[:, 1::2] = boxes[:, 1::2] * scale_y + pad_y
        return pixels, {
            "boxes": torch.from_numpy(boxes),
            "labels": torch.from_numpy(sample.labels.copy()),
            "difficult": torch.from_numpy(sample.difficult.copy()),
            "id": sample.id,
            "geometry": geometry,
        }


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    extent = np.maximum(0, np.minimum(box[2:], boxes[:, 2:]) - np.maximum(box[:2], boxes[:, :2]))
    overlap = extent.prod(axis=1)
    union = (box[2:] - box[:2]).prod() + (boxes[:, 2:] - boxes[:, :2]).prod(axis=1) - overlap
    return np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0)


def _evaluate_class(samples: list[Sample], detections: list, label: int) -> dict:
    ground_truth = {}
    positives = 0
    for sample in samples:
        mask = sample.labels == label
        difficult = sample.difficult[mask]
        positives += int((~difficult).sum())
        if mask.any():
            ground_truth[sample.id] = (sample.boxes[mask], difficult, np.zeros(len(difficult), dtype=bool))

    matches = []
    ignored = 0
    for _, image_id, box in sorted(detections, key=lambda item: -item[0]):
        if image_id not in ground_truth:
            matches.append(False)
            continue
        boxes, difficult, used = ground_truth[image_id]
        overlaps = _iou(box, boxes)
        best = int(overlaps.argmax())
        if overlaps[best] < 0.5:
            matches.append(False)
        elif difficult[best]:
            ignored += 1
        else:
            matches.append(not used[best])
            used[best] = True

    hits = np.cumsum(matches, dtype=np.int64)
    ap = None
    if positives:
        recall = hits / positives
        precision = hits / np.arange(1, len(hits) + 1)
        ap = float(np.mean([precision[recall >= level / 10].max(initial=0) for level in range(11)]))
    true_positives = int(hits[-1]) if len(hits) else 0
    return {
        "ap": ap,
        "positives": positives,
        "detections": len(detections),
        "ignored": ignored,
        "true_positives": true_positives,
        "false_positives": len(matches) - true_positives,
    }


def evaluate(samples: list[Sample], predictions: dict[str, dict]) -> dict:
    """Score per-image detections at IoU 0.5. Classes without positives have null AP."""
    image_ids = {sample.id for sample in samples}
    unknown = predictions.keys() - image_ids
    if unknown:
        raise ValueError(f"Predictions contain unknown VOC image IDs: {sorted(unknown)}")
    by_class = [[] for _ in CLASSES]
    for image_id in sorted(predictions):
        output = predictions[image_id]
        boxes = np.asarray(output["boxes"], dtype=np.float32).reshape(-1, 4)
        labels = np.asarray(output["labels"]).reshape(-1)
        scores = np.asarray(output["scores"], dtype=np.float64).reshape(-1)
        if len(boxes) != len(labels) or len(boxes) != len(scores):
            raise ValueError(f"Mismatched prediction lengths for {image_id}")
        if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
            raise ValueError(f"Nonfinite prediction for {image_id}")
        for box, label, score in zip(boxes, labels, scores):
            if not 0 <= label < len(CLASSES) or label != int(label):
                raise ValueError(f"Invalid class index {label} for {image_id}")
            by_class[int(label)].append((score, image_id, box))

    per_class = {
        name: _evaluate_class(samples, detections, label)
        for label, (name, detections) in enumerate(zip(CLASSES, by_class))
    }
    aps = [item["ap"] for item in per_class.values() if item["ap"] is not None]
    return {"map50": float(np.mean(aps)) if aps else 0.0, "per_class": per_class}
