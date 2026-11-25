"""CIFAR-sized VGG-A variants with a shared channel-width multiplier."""
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "codes" / "VGG_BatchNorm"))
try:
    from models.vgg import init_weights_
finally:
    sys.path.pop(0)

VARIANTS = ("vgg_a", "vgg_a_bn", "vgg_a_dropout")
LAYOUT = ((64,), (128,), (256, 256), (512, 512), (512, 512))


class VGG(nn.Module):
    def __init__(self, variant, width, activation="relu"):
        super().__init__()
        if variant not in VARIANTS or width <= 0:
            raise ValueError("invalid VGG variant or width")
        activate = {"relu": lambda: nn.ReLU(inplace=True), "gelu": nn.GELU}[activation]
        channels = 3
        blocks = []
        bn = variant == "vgg_a_bn"
        dropout = variant == "vgg_a_dropout"
        for stage in LAYOUT:
            for out in stage:
                scaled = max(1, round(out * width))
                blocks.append(nn.Conv2d(channels, scaled, 3, padding=1))
                if bn:
                    blocks.append(nn.BatchNorm2d(scaled))
                blocks.append(activate())
                channels = scaled
            blocks.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*blocks)
        head = []
        for _ in range(2):
            if dropout:
                head.append(nn.Dropout(0.5))
            head.append(nn.Linear(channels, channels))
            if bn:
                head.append(nn.BatchNorm1d(channels))
            head.append(activate())
        head.append(nn.Linear(channels, 10))
        self.classifier = nn.Sequential(*head)
        self.apply(init_weights_)

    def forward(self, inputs):
        return self.classifier(self.features(inputs).flatten(1))


def checkpoint(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = VGG(payload["variant"], payload["width"], payload.get("activation", "relu"))
    model.load_state_dict(payload["weights"], strict=True)
    return model.to(device), payload
