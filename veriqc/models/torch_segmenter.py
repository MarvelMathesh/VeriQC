"""Compact U-Net for thin-crack segmentation (KolektorSDD).

The encoder is a torchvision ResNet-18; the decoder uses the four encoder
stages.  The input geometry is stored with the checkpoint so training and
inference cannot silently disagree.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

__all__ = ["UNet", "Segmenter"]

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        encoder = resnet18(weights=weights)
        self.stem = nn.Sequential(
            encoder.conv1, encoder.bn1, encoder.relu
        )
        self.pool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = _ConvBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = _ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = _ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = _ConvBlock(96, 32)
        self.up0 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec0 = _ConvBlock(16, 16)
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)
        p0 = self.pool(s0)
        c1 = self.layer1(p0)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        d4 = self.dec4(torch.cat([self.up4(c4), c3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), c2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), c1], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), s0], dim=1))
        d0 = self.dec0(self.up0(d1))
        return self.head(d0)


@dataclass(frozen=True)
class SegmenterConfig:
    input_width: int = 384
    input_height: int = 1024


class Segmenter:
    """Inference wrapper around the trained U-Net."""

    def __init__(self, model: UNet, config: SegmenterConfig, device: str = "cpu") -> None:
        self.model = model.to(device).eval()
        self.config = config
        self.device = torch.device(device)
        self._mean = torch.tensor(_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(_STD, device=self.device).view(1, 3, 1, 1)

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "cpu") -> "Segmenter":
        blob = torch.load(path, map_location="cpu", weights_only=True)
        raw = blob.get("config", {})
        config = SegmenterConfig(
            input_width=int(raw.get("input_width", 384)),
            input_height=int(raw.get("input_height", 1024)),
        )
        model = UNet(pretrained=False)
        model.load_state_dict(blob["state_dict"])
        return cls(model, config, device=device)

    def _tensor(self, image: Image.Image) -> torch.Tensor:
        resized = image.convert("RGB").resize(
            (self.config.input_width, self.config.input_height),
            Image.BILINEAR,
        )
        array = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return (tensor - self._mean) / self._std

    @torch.no_grad()
    def predict_proba(self, image: Image.Image) -> tuple[np.ndarray, float]:
        started = time.perf_counter()
        tensor = self._tensor(image)
        logits = self.model(tensor)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
        prob_image = Image.fromarray(prob.astype(np.float32), mode="F").resize(
            image.size, Image.BILINEAR
        )
        heatmap = np.asarray(prob_image, dtype=np.float32)
        elapsed = (time.perf_counter() - started) * 1000.0
        return heatmap, elapsed

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "config": {
                    "input_width": self.config.input_width,
                    "input_height": self.config.input_height,
                },
            },
            target,
        )

    def export_onnx(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        dummy = torch.zeros(
            1, 3, self.config.input_height, self.config.input_width, device=self.device
        )
        torch.onnx.export(
            self.model,
            dummy,
            str(target),
            input_names=["image"],
            output_names=["logits"],
            opset_version=17,
            dynamic_axes=None,
        )
        return target
