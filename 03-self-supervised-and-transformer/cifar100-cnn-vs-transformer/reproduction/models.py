"""Parameter-matched residual CNN and Vision Transformer for CIFAR-100."""

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.activate = nn.ReLU()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.activate(self.branch(images) + self.shortcut(images))


class ResidualCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, 3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            ResidualBlock(48, 64, stride=1),
            ResidualBlock(64, 96, stride=2),
            ResidualBlock(96, 128, stride=2),
            ResidualBlock(128, 144, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(144, 100)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(images)).flatten(1))


class TransformerBlock(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.mlp_norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(tokens)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        tokens = tokens + attended
        return tokens + self.mlp(self.mlp_norm(tokens))


class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        width = 160
        patch_size = 4
        num_patches = (32 // patch_size) ** 2
        self.patch_projection = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.empty(1, 1, width))
        self.position_embedding = nn.Parameter(torch.empty(1, num_patches + 1, width))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.encoder = nn.Sequential(*(TransformerBlock(width, heads=4) for _ in range(4)))
        self.norm = nn.LayerNorm(width)
        self.classifier = nn.Linear(width, 100)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patches = self.patch_projection(images).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(images.shape[0], -1, -1)
        tokens = torch.cat((cls, patches), dim=1) + self.position_embedding
        encoded = self.encoder(tokens)
        return self.classifier(self.norm(encoded[:, 0]))


def build_model(name: str) -> nn.Module:
    if name == "cnn":
        return ResidualCNN()
    if name == "vit":
        return VisionTransformer()
    raise ValueError(f"Unknown model {name!r}; expected 'cnn' or 'vit'")
