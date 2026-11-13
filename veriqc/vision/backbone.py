"""Patch feature embedder for the Anomaly Gate.

Follows the PatchCore recipe (Roth et al., CVPR 2022):
- Feature maps extracted from intermediate stages of a frozen ImageNet CNN.
- Local 3x3 average pooling to incorporate neighborhood context.
- Bilinear upsampling to a common spatial grid.
- L2-normalization so Euclidean distances equal cosine distances.

Supported production backbones:
- 'efficientnet_b0': The Pareto winner (AMD-1). 5.29M parameters, 0.9502 AUROC.
- 'mobilenet_v3_small': The ultra-low latency edge variant (2.54M parameters, 2.87ms on RPi5).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

__all__ = ["PatchEmbedder", "IMAGENET_MEAN", "IMAGENET_STD"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class _MobileNetTrunk(nn.Module):
    """Expose feature stages f2, f3 from MobileNetV3."""

    def __init__(self, features: nn.Sequential, stages: tuple[int, int] = (2, 3)):
        super().__init__()
        self.features = features
        self.stages = stages

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outs: list[torch.Tensor] = []
        h = x
        for i, layer in enumerate(self.features):
            h = layer(h)
            if i in self.stages:
                outs.append(h)
        return outs


def _build_backbone(name: str) -> tuple[nn.Module, str]:
    import torchvision.models as tvm

    if name == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.DEFAULT)
        return m, "efficientnet"
    if name == "mobilenet_v3_small":
        m = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.DEFAULT)
        m.features = _MobileNetTrunk(m.features, (2, 3))
        return m, "mobilenet"
    raise ValueError(f"Unsupported production backbone {name!r}. Use 'efficientnet_b0' or 'mobilenet_v3_small'.")


class PatchEmbedder:
    """Extracts locally-aggregated, L2-normalized patch features from an image."""

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        input_size: int = 224,
        aggr_kernel: int = 3,
        device: torch.device | str = "cpu",
    ):
        self.backbone_name = backbone
        self.input_size = input_size
        self.aggr_kernel = aggr_kernel
        self.device = torch.device(device)

        model, kind = _build_backbone(backbone)
        self.model = model.to(self.device).eval()
        self.kind = kind
        self.stages: list[torch.Tensor] = []

        if kind == "efficientnet":
            # Extract features from stage 2 (features[2]) and stage 3 (features[3])
            def hook_f2(module, inp, out):
                self.stages.append(out)

            def hook_f3(module, inp, out):
                self.stages.append(out)

            self.model.features[2].register_forward_hook(hook_f2)
            self.model.features[3].register_forward_hook(hook_f3)

        self.mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def preprocess(self, images: Sequence[Image.Image | np.ndarray]) -> torch.Tensor:
        """Convert a batch of PIL or numpy images to normalized tensor."""
        tensors = []
        for img in images:
            if isinstance(img, np.ndarray):
                im = Image.fromarray(img).convert("RGB")
            elif isinstance(img, Image.Image):
                im = img.convert("RGB")
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")

            im = im.resize((self.input_size, self.input_size), Image.BILINEAR)
            arr = np.array(im, dtype=np.uint8, copy=True)
            tensors.append(torch.from_numpy(arr).permute(2, 0, 1).float().div(255.0))

        batch = torch.stack(tensors, dim=0).to(self.device)
        return (batch - self.mean) / self.std

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning patch embeddings of shape (B, C_total, H, W)."""
        self.stages.clear()
        if self.kind == "mobilenet":
            self.stages = self.model.features(x)
        elif self.kind == "efficientnet":
            self.model(x)

        if len(self.stages) < 2:
            raise RuntimeError(f"Expected at least 2 feature stages, got {len(self.stages)}")

        # Smooth each stage with average pooling
        pad = self.aggr_kernel // 2
        smoothed = [
            F.avg_pool2d(s, kernel_size=self.aggr_kernel, stride=1, padding=pad)
            for s in self.stages[:2]
        ]

        # Resample fine stage to coarse stage spatial grid
        target_h, target_w = smoothed[0].shape[-2:]
        aligned = [smoothed[0]]
        for s in smoothed[1:]:
            if s.shape[-2:] != (target_h, target_w):
                s = F.interpolate(s, size=(target_h, target_w), mode="bilinear", align_corners=False)
            aligned.append(s)

        # Concatenate channels and normalize
        concat = torch.cat(aligned, dim=1)  # (B, C, H, W)
        norm = torch.linalg.norm(concat, dim=1, keepdim=True).clamp_min(1e-6)
        return concat / norm
