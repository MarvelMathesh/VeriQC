"""PatchCore anomaly model with separate image-score and localization layers.

Only two mid-level ResNet stages are executed.  The image score comes from the
``layer2`` bank, which preserves fine texture; the localization map comes from
the ``layer3`` bank, which is more semantic.  ``layer4`` is never computed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from veriqc.models.types import AnomalyResult

__all__ = ["PatchCore", "PatchCoreConfig"]

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class PatchCoreConfig:
    input_size: int = 448
    backbone: str = "resnet18"
    image_layer: str = "layer2"
    pixel_layer: str = "layer3"
    bank_size: int = 4000
    coreset_pool: int = 20000
    aggr_kernel: int = 3
    batch_size: int = 8
    seed: int = 20261121


class _ResNetFeatures(torch.nn.Module):
    def __init__(self, stages: Sequence[str], pretrained: bool) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        allowed = {"layer2", "layer3"}
        if not set(stages) <= allowed:
            raise ValueError(f"unsupported ResNet-18 stage set: {stages}")
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
        self.stem = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.stages = tuple(stages)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        x = self.layer1(x)
        out: dict[str, torch.Tensor] = {}
        if "layer2" in self.stages:
            x = self.layer2(x)
            out["layer2"] = x
        else:
            x = self.layer2(x)
        if "layer3" in self.stages:
            x = self.layer3(x)
            out["layer3"] = x
        return out


class PatchCore:
    def __init__(self, config: PatchCoreConfig, device: str = "cpu", *, pretrained: bool = True) -> None:
        if config.backbone != "resnet18":
            raise ValueError(f"unsupported anomaly backbone: {config.backbone!r}")
        self.config = config
        self.device = torch.device(device)
        stages = tuple(dict.fromkeys((config.image_layer, config.pixel_layer)))
        self.extractor = _ResNetFeatures(stages, pretrained=pretrained).to(self.device).eval()
        self.image_bank: torch.Tensor | None = None
        self.pixel_bank: torch.Tensor | None = None
        self._mean = torch.tensor(_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(_STD, device=self.device).view(1, 3, 1, 1)

    @property
    def is_fitted(self) -> bool:
        return self.image_bank is not None and self.pixel_bank is not None

    @property
    def bank_size(self) -> int:
        return int(self.image_bank.shape[0]) if self.image_bank is not None else 0

    def _letterbox(self, image: Image.Image) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        size = self.config.input_size
        width, height = image.size
        scale = size / max(width, height)
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        ox = (size - new_w) // 2
        oy = (size - new_h) // 2
        resized = image.convert("RGB").resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("RGB", (size, size), (0, 0, 0))
        canvas.paste(resized, (ox, oy))
        array = np.asarray(canvas, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return (tensor - self._mean) / self._std, (ox, oy, new_w, new_h)

    @torch.no_grad()
    def _stage_features(self, batch: torch.Tensor, stage: str) -> torch.Tensor:
        features = self.extractor(batch)[stage]
        pad = self.config.aggr_kernel // 2
        features = F.avg_pool2d(features, self.config.aggr_kernel, stride=1, padding=pad)
        return features / torch.linalg.vector_norm(features, dim=1, keepdim=True).clamp_min(1e-6)

    @staticmethod
    def _coreset(patches: torch.Tensor, count: int, seed: int) -> torch.Tensor:
        generator = torch.Generator(device=patches.device.type).manual_seed(seed)
        selected = torch.empty(count, dtype=torch.long, device=patches.device)
        selected[0] = int(torch.randint(0, patches.shape[0], (1,), generator=generator, device=patches.device).item())
        min_distance = torch.full((patches.shape[0],), float("inf"), device=patches.device)
        for index in range(1, count):
            current = patches[selected[index - 1]]
            for offset in range(0, patches.shape[0], 8192):
                block = patches[offset : offset + 8192]
                distance = torch.linalg.vector_norm(block - current, dim=1)
                min_distance[offset : offset + 8192] = torch.minimum(
                    min_distance[offset : offset + 8192], distance
                )
            selected[index] = int(torch.argmax(min_distance).item())
        return patches[selected].contiguous().cpu()

    @torch.no_grad()
    def fit(self, images: Sequence[str | Path | Image.Image]) -> "PatchCore":
        if not images:
            raise ValueError("no normal images supplied")
        stage_pools: dict[str, list[torch.Tensor]] = {
            self.config.image_layer: [],
            self.config.pixel_layer: [],
        }
        for start in range(0, len(images), self.config.batch_size):
            chunk = images[start : start + self.config.batch_size]
            tensors = []
            for image in chunk:
                if isinstance(image, (str, Path)):
                    with Image.open(image) as opened:
                        tensors.append(self._letterbox(opened.convert("RGB"))[0])
                else:
                    tensors.append(self._letterbox(image)[0])
            batch = torch.cat(tensors, dim=0)
            for stage in stage_pools:
                features = self._stage_features(batch, stage)
                b, c, h, w = features.shape
                stage_pools[stage].append(features.permute(0, 2, 3, 1).reshape(-1, c))
        generator = torch.Generator().manual_seed(self.config.seed)
        pools = {}
        for stage, parts in stage_pools.items():
            patches = torch.cat(parts, dim=0)
            if patches.shape[0] > self.config.coreset_pool:
                index = torch.randperm(patches.shape[0], generator=generator)[: self.config.coreset_pool]
                patches = patches[index]
            count = min(self.config.bank_size, patches.shape[0])
            pools[stage] = self._coreset(patches, count, self.config.seed)
        self.image_bank = pools[self.config.image_layer]
        self.pixel_bank = pools[self.config.pixel_layer]
        return self

    @torch.no_grad()
    def score(self, image: str | Path | Image.Image | np.ndarray) -> AnomalyResult:
        if not self.is_fitted:
            raise RuntimeError("the anomaly model is not fitted")
        started = time.perf_counter()
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif isinstance(image, (str, Path)):
            with Image.open(image) as opened:
                image = opened.convert("RGB")
        else:
            image = image.convert("RGB")
        tensor, content = self._letterbox(image)
        image_features = self._stage_features(tensor, self.config.image_layer)
        pixel_features = self._stage_features(tensor, self.config.pixel_layer)
        _, c, _, _ = image_features.shape
        image_patches = image_features.permute(0, 2, 3, 1).reshape(-1, c)
        image_distance = torch.cdist(image_patches, self.image_bank.to(self.device)).min(dim=1)[0]
        image_score = float(image_distance.max().item())
        _, c, h, w = pixel_features.shape
        pixel_patches = pixel_features.permute(0, 2, 3, 1).reshape(-1, c)
        pixel_distance = torch.cdist(pixel_patches, self.pixel_bank.to(self.device)).min(dim=1)[0]
        score_map = pixel_distance.view(1, 1, h, w)
        upsampled = F.interpolate(
            score_map, size=(self.config.input_size, self.config.input_size), mode="bilinear", align_corners=False
        )[0, 0].cpu().numpy()
        heatmap = self._to_native(upsampled, content, image.size)
        elapsed = (time.perf_counter() - started) * 1000.0
        return AnomalyResult(image_score=image_score, heatmap=heatmap, latency_ms=elapsed)

    @staticmethod
    def _to_native(
        heatmap: np.ndarray,
        content: tuple[int, int, int, int],
        native_size: tuple[int, int],
    ) -> np.ndarray:
        ox, oy, new_w, new_h = content
        crop = heatmap[oy : oy + new_h, ox : ox + new_w]
        if crop.size == 0:
            return np.zeros((native_size[1], native_size[0]), dtype=np.float32)
        return cv2.resize(crop, native_size, interpolation=cv2.INTER_LINEAR)

    def save(self, path: str | Path) -> None:
        if not self.is_fitted:
            raise RuntimeError("cannot save an unfitted anomaly model")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.config),
                "state_dict": self.extractor.state_dict(),
                "image_bank": self.image_bank,
                "pixel_bank": self.pixel_bank,
            },
            target,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "PatchCore":
        blob = torch.load(path, map_location="cpu", weights_only=True)
        config = PatchCoreConfig(**blob["config"])
        model = cls(config, device=device, pretrained=False)
        model.extractor.load_state_dict(blob["state_dict"])
        model.extractor.to(model.device).eval()
        model.image_bank = blob["image_bank"].float()
        model.pixel_bank = blob["pixel_bank"].float()
        return model

    def export_backbone_onnx(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        dummy = torch.zeros(1, 3, self.config.input_size, self.config.input_size, device=self.device)
        torch.onnx.export(
            self.extractor,
            dummy,
            str(target),
            input_names=["image"],
            output_names=list(self.extractor.stages),
            opset_version=17,
            dynamic_axes=None,
        )
        return target
