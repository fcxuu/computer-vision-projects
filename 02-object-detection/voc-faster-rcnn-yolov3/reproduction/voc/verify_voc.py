"""Run synthetic checks for VOC coordinate transforms and evaluation."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from PIL import Image

from voc import Sample, VocDataset, evaluate, unletterbox


def _sample(image_id, path, box, label=0, difficult=False):
    return Sample(
        image_id,
        path,
        np.array([box], dtype=np.float32),
        np.array([label], dtype=np.int64),
        np.array([difficult], dtype=bool),
    )


def main():
    with TemporaryDirectory() as folder:
        image_path = Path(folder) / "image.png"
        Image.new("RGB", (11, 7), (5, 10, 15)).save(image_path)
        box = [0, 0, 10, 6]
        first = _sample("first", image_path, box)
        second = _sample("second", image_path, box)
        difficult = _sample("difficult", image_path, box, difficult=True)

        pixels, target = VocDataset([first], 16)[0]
        assert pixels.shape == (3, 16, 16) and pixels.dtype == torch.uint8
        assert pixels[:, 0, 0].tolist() == [114, 114, 114]
        assert pixels[:, 3, 0].tolist() == [5, 10, 15]
        assert target["labels"].dtype == torch.int64
        assert target["difficult"].dtype == torch.bool
        assert target["boxes"].dtype == torch.float32 and target["id"] == first.id
        restored = unletterbox(target["boxes"].numpy(), target["geometry"])
        np.testing.assert_allclose(restored, first.boxes, atol=1e-6)
        np.testing.assert_allclose(
            unletterbox(np.array([[-50, -50, 50, 50]]), target["geometry"]),
            [[0, 0, 11, 7]],
        )

        predictions = {
            "first": {"boxes": [box], "labels": [0], "scores": [0.9]},
            "second": {"boxes": np.array([[0, 0, 5, 6]]), "labels": [0], "scores": [0.6]},
            "difficult": {"boxes": [box], "labels": [0], "scores": [1.0]},
        }
        samples = [first, second, difficult]
        perfect = evaluate(samples, predictions)
        assert perfect["map50"] == 1.0
        assert perfect["per_class"]["aeroplane"]["ignored"] == 1
        json.dumps(perfect, allow_nan=False)

        predictions["first"] = {"boxes": [box, box, box], "labels": [0, 0, 1], "scores": [0.9, 0.8, 0.99]}
        scored = evaluate(samples, predictions)
        assert np.isclose(scored["map50"], (6 + 5 * 2 / 3) / 11)
        assert scored["per_class"]["aeroplane"]["false_positives"] == 1
        assert scored["per_class"]["aeroplane"]["true_positives"] == 2
        assert scored["per_class"]["bicycle"]["false_positives"] == 1
        assert scored["per_class"]["bicycle"]["ap"] is None
        assert evaluate(samples, {})["map50"] == 0.0

    print("VOC checks passed: letterbox/inverse, clipping, tensor types, AP, difficult, duplicate and wrong-class detections.")


if __name__ == "__main__":
    main()
