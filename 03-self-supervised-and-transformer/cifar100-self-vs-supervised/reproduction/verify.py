"""Check positive-pair indexing and deterministic CPU/MPS image transforms."""
import torch

from cifar100_data import Partition, transform
from models import ntxent


def check_pairs(device):
    same = torch.eye(3, device=device, requires_grad=True)
    paired, retrieval = ntxent(same, same, 0.2)
    displaced, _ = ntxent(same, same.roll(1, dims=0), 0.2)
    paired.backward()
    assert torch.isfinite(same.grad).all()
    assert float(retrieval) == 1.0
    assert float(paired.detach()) < float(displaced.detach())
    print(f"{device} NT-Xent paired {paired.detach():.6f}, swapped {displaced.detach():.6f}")


def check_transforms():
    rng = torch.Generator().manual_seed(10)
    pixels = torch.randint(0, 256, (7, 32, 32, 3), generator=rng, dtype=torch.uint8)
    labels = torch.arange(len(pixels))
    partition = Partition(pixels, labels)
    batches = list(partition.batches(3))
    assert [len(y) for _, y in batches] == [3, 3, 1]
    torch.testing.assert_close(torch.cat([y for _, y in batches]), labels)
    first = transform(pixels, 64, torch.Generator().manual_seed(22), strong=True)
    rng = torch.Generator().manual_seed(22)
    second = transform(pixels, 64, rng, strong=True)
    different_view = transform(pixels, 64, rng, strong=True)
    torch.testing.assert_close(first, second)
    assert not torch.equal(first, different_view)
    if torch.backends.mps.is_available():
        mac = transform(pixels.to("mps"), 64, torch.Generator().manual_seed(22), strong=True)
        torch.testing.assert_close(mac.cpu(), first, atol=4e-5, rtol=4e-5)
    print("Data pipeline includes tail batch and generates matching seeded, distinct consecutive views")


if __name__ == "__main__":
    check_pairs("cpu")
    if torch.backends.mps.is_available():
        check_pairs("mps")
    check_transforms()
