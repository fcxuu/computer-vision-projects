"""Verify matched parameter budgets and clipped CutMix labeling."""
import numpy as np
import torch

from experiment import cutmix, param_counts
from models import build_model


def check_cutmix(device):
    originals = torch.arange(8, dtype=torch.float32, device=device)[:, None, None, None].repeat(1, 3, 7, 11)
    labels = torch.arange(8, device=device)
    mixed, other, weight = cutmix(originals, labels, np.random.default_rng(19), probability=1.0)
    changed = (mixed[:, 0] != originals[:, 0]).sum(dim=(1, 2)).cpu()
    assert 0 < weight < 1 and bool((other != labels).any())
    expected = round((1 - weight) * 7 * 11)
    for index, (label, replacement) in enumerate(zip(labels.tolist(), other.tolist())):
        assert changed[index] == (expected if replacement != label else 0)
        assert set(mixed[index, 0].flatten().tolist()) <= {label, replacement}
    skipped, other, weight = cutmix(originals, labels, np.random.default_rng(19), probability=0.0)
    assert skipped is originals and other is labels and weight == 1.0
    print(f"{device} CutMix replaced {expected}/77 pixels; targets follow the replacement permutation")


def check_models():
    counts = param_counts()
    gap = (max(counts.values()) - min(counts.values())) / min(counts.values())
    assert gap <= 0.03
    for key, count in counts.items():
        model = build_model(key)
        data = torch.randn(3, 3, 32, 32)
        logits = model(data)
        assert logits.shape == (3, 100)
        logits.square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        print(f"{key} {count:,} trainable parameters; finite gradient")
    print(f"Parameter gap {gap:.3%}")


if __name__ == "__main__":
    torch.set_num_threads(4)
    check_models()
    check_cutmix("cpu")
    if torch.backends.mps.is_available():
        check_cutmix("mps")
