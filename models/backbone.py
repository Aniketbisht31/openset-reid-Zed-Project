"""Compact backbones for real-time person re-identification.

Supported architectures:

* **osnet_x0_5** — Lightweight Omni-Scale Network (~1.3 M params).
* **resnet18** — Standard ResNet-18 (ImageNet pretrained).
* **mobilenetv3_small** — MobileNetV3-Small (ImageNet pretrained).

All backbones output a 1-D feature vector via global average pooling;
the downstream :class:`models.heads.EmbeddingHead` projects it to the
final 256-D L2-normalised embedding.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


# ═══════════════════════════════════════════════════════════════════════
# OSNet x0.5  (simplified, self-contained implementation)
# ═══════════════════════════════════════════════════════════════════════

class _OSBlock(nn.Module):
    """Omni-scale feature-learning block (multi-stream + channel gate)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        mid = out_ch // 4

        def _conv_bn_relu(ic, oc, k, p):
            return nn.Sequential(
                nn.Conv2d(ic, oc, k, padding=p, bias=False),
                nn.BatchNorm2d(oc),
                nn.ReLU(inplace=True),
            )

        # Three parallel streams with receptive fields 3, 5, 7
        self.s1 = nn.Sequential(
            _conv_bn_relu(in_ch, mid, 1, 0),
            _conv_bn_relu(mid, mid, 3, 1),
        )
        self.s2 = nn.Sequential(
            _conv_bn_relu(in_ch, mid, 1, 0),
            _conv_bn_relu(mid, mid, 3, 1),
            _conv_bn_relu(mid, mid, 3, 1),
        )
        self.s3 = nn.Sequential(
            _conv_bn_relu(in_ch, mid, 1, 0),
            _conv_bn_relu(mid, mid, 3, 1),
            _conv_bn_relu(mid, mid, 3, 1),
            _conv_bn_relu(mid, mid, 3, 1),
        )

        # Unified channel-attention gate
        cat_ch = mid * 3
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(cat_ch, cat_ch, 1, bias=True),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(cat_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

        # Shortcut
        self.shortcut = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False),
                          nn.BatchNorm2d(out_ch))
            if in_ch != out_ch
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([self.s1(x), self.s2(x), self.s3(x)], dim=1)
        cat = cat * self.gate(cat)
        return self.relu(self.fuse(cat) + self.shortcut(x))


class OSNet_x0_5(nn.Module):
    """Omni-Scale Network (×0.5 width) — ~1.3 M parameters."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.stage2 = nn.Sequential(
            _OSBlock(32, 128),
            nn.AvgPool2d(2, 2),
        )
        self.stage3 = nn.Sequential(
            _OSBlock(128, 192),
            nn.AvgPool2d(2, 2),
        )
        self.stage4 = nn.Sequential(
            _OSBlock(192, 256),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.out_channels = 256
        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.gap(x).flatten(1)


# ═══════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════

def build_backbone(name: str = "osnet_x0_5", pretrained: bool = True) -> nn.Module:
    """Instantiate a backbone and attach an ``out_channels`` attribute.

    Args:
        name:       ``'osnet_x0_5'`` | ``'resnet18'`` | ``'mobilenetv3_small'``
        pretrained: Load ImageNet weights where applicable.

    Returns:
        ``nn.Module`` whose forward pass maps ``[B,3,H,W] → [B, C]``.
    """
    if name == "osnet_x0_5":
        backbone = OSNet_x0_5()
        # OSNet is trained from scratch — no ImageNet checkpoint bundled.
        return backbone

    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base = models.resnet18(weights=weights)
        backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        backbone.out_channels = 512
        return backbone

    if name == "mobilenetv3_small":
        weights = (
            models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        )
        base = models.mobilenet_v3_small(weights=weights)
        backbone = nn.Sequential(
            base.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        backbone.out_channels = 576
        return backbone

    raise ValueError(
        f"Unknown backbone '{name}'.  "
        "Choose from: osnet_x0_5, resnet18, mobilenetv3_small"
    )
