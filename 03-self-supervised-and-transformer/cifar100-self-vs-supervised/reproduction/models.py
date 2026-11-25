"""ResNet-18 backbone, SimCLR projection MLP, and supervised probe."""
from hashlib import sha256

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def encoder(imagenet=False):
    network = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if imagenet else None)
    network.fc = nn.Identity()
    return network


class SimCLR(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = encoder()
        self.projector = nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 128))

    def forward(self, images):
        return self.projector(self.backbone(images))


class Classifier(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(512, 100)

    def forward(self, images):
        return self.head(self.backbone(images))


def ntxent(first, second, temperature):
    """Use the paired view as the sole positive and all other images as negatives."""
    if first.shape != second.shape or first.ndim != 2 or len(first) < 2:
        raise ValueError("NT-Xent needs paired matrices with at least two examples")
    features = F.normalize(torch.cat((first, second)), dim=1)
    count = len(first)
    rows = torch.arange(2 * count, device=features.device)
    similarities = (features @ features.T) / temperature
    similarities = similarities.masked_fill(rows[:, None] == rows[None, :], float("-inf"))
    labels = (rows + count) % (2 * count)
    return F.cross_entropy(similarities, labels), (similarities.argmax(1) == labels).float().mean()


def fingerprint(network):
    fingerprint = sha256()
    for key, value in network.state_dict().items():
        fingerprint.update(key.encode("utf-8"))
        fingerprint.update(value.detach().cpu().contiguous().numpy().tobytes())
    return fingerprint.hexdigest()
