"""The Anomaly Gate: Normals-only patch-memory anomaly detection.

Implements the winning PatchCore recipe (Roth et al., CVPR 2022; AMD-1):
1. Embeds normal images into locally-aggregated patch features.
2. Selects a greedy k-center coreset (10% ratio) as the reference memory bank.
3. Computes nearest-neighbor patch distance against the bank to produce an image-level
   anomaly score and a 2D spatial anomaly heatmap.
4. Serializes the compact memory bank to disk (<3 MB).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from veriqc.vision.backbone import PatchEmbedder

__all__ = ["GateConfig", "GateScore", "AnomalyGate", "greedy_coreset"]


def greedy_coreset(
    features: torch.Tensor,
    n_coreset: int,
    seed: int = 0,
    chunk: int = 8192,
) -> torch.Tensor:
    """Fast GPU-accelerated greedy k-center coreset selection."""
    n = features.shape[0]
    if n_coreset >= n:
        return torch.arange(n, device=features.device)

    g = torch.Generator(device=features.device.type).manual_seed(seed)
    selected = torch.empty(n_coreset, dtype=torch.long, device=features.device)

    first = int(torch.randint(0, n, (1,), generator=g, device=features.device).item())
    selected[0] = first
    min_dist = torch.full((n,), float("inf"), device=features.device)

    for i in range(1, n_coreset):
        cur = features[selected[i - 1]]
        for s in range(0, n, chunk):
            block = features[s : s + chunk]
            d = torch.linalg.vector_norm(block - cur, dim=1)
            min_dist[s : s + chunk] = torch.minimum(min_dist[s : s + chunk], d)
        selected[i] = torch.argmax(min_dist)

    return selected


@dataclass
class GateConfig:
    backbone: str = "efficientnet_b0"
    input_size: int = 224
    coreset_ratio: float = 0.1
    max_exemplars: int = 2000  # Cap at 2000 exemplars (~1.5 MB memory bank, edge-budget clean)
    aggr_kernel: int = 3
    batch_size: int = 16
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0


@dataclass
class GateScore:
    """Anomaly score and spatial localization."""

    image_scores: np.ndarray  # (B,) floats
    heatmaps: np.ndarray      # (B, H, W) floats in [0, 1]
    latency_ms: float


class AnomalyGate:
    """Production Anomaly Gate with Coreset-kNN memory bank."""

    def __init__(self, config: GateConfig | None = None):
        self.config = config or GateConfig()
        self.device = torch.device(self.config.device)
        self.embedder = PatchEmbedder(
            backbone=self.config.backbone,
            input_size=self.config.input_size,
            aggr_kernel=self.config.aggr_kernel,
            device=self.device,
        )
        self.memory_bank: torch.Tensor | None = None  # (M, C)
        self.grid: tuple[int, int] | None = None
        self.fit_time_seconds: float = 0.0

    @property
    def is_fitted(self) -> bool:
        return self.memory_bank is not None

    def memory_bytes(self) -> int:
        if self.memory_bank is None:
            return 0
        return self.memory_bank.nelement() * self.memory_bank.element_size()

    @torch.no_grad()
    def fit(self, normal_images: Sequence[str | Path | Image.Image]) -> "AnomalyGate":
        """Build coreset memory bank from normal calibration images."""
        t0 = time.perf_counter()
        loaded = []
        for img in normal_images:
            if isinstance(img, (str, Path)):
                loaded.append(Image.open(img).convert("RGB"))
            elif isinstance(img, Image.Image):
                loaded.append(img.convert("RGB"))
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")

        all_patches: list[torch.Tensor] = []
        bs = self.config.batch_size
        for i in range(0, len(loaded), bs):
            chunk = loaded[i : i + bs]
            x = self.embedder.preprocess(chunk)
            emb = self.embedder.embed(x)  # (B, C, H, W)
            b, c, h, w = emb.shape
            self.grid = (h, w)
            flat = emb.permute(0, 2, 3, 1).reshape(-1, c)  # (B*H*W, C)
            all_patches.append(flat)

        stacked = torch.cat(all_patches, dim=0)
        n_coreset = min(
            self.config.max_exemplars,
            max(1, int(round(stacked.shape[0] * self.config.coreset_ratio))),
        )
        indices = greedy_coreset(stacked, n_coreset=n_coreset, seed=self.config.seed)
        self.memory_bank = stacked[indices].contiguous()

        self.fit_time_seconds = time.perf_counter() - t0
        return self

    @torch.no_grad()
    def score(
        self,
        images: Sequence[str | Path | Image.Image | np.ndarray],
    ) -> GateScore:
        """Compute anomaly score and heatmap for a batch of images."""
        if not self.is_fitted:
            raise RuntimeError("AnomalyGate must be fitted or loaded before scoring.")

        t0 = time.perf_counter()
        loaded = []
        for img in images:
            if isinstance(img, (str, Path)):
                loaded.append(Image.open(img).convert("RGB"))
            elif isinstance(img, Image.Image):
                loaded.append(img.convert("RGB"))
            elif isinstance(img, np.ndarray):
                loaded.append(Image.fromarray(img).convert("RGB"))
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")

        all_scores: list[float] = []
        all_heatmaps: list[np.ndarray] = []
        bs = self.config.batch_size
        patch_chunk_size = 4096

        for i in range(0, len(loaded), bs):
            chunk = loaded[i : i + bs]
            x = self.embedder.preprocess(chunk)
            emb = self.embedder.embed(x)  # (B, C, H, W)
            b, c, h, w = emb.shape

            patches = emb.permute(0, 2, 3, 1).reshape(b * h * w, c)  # (B*HW, C)

            # Chunked distance against memory bank to prevent VRAM spikes
            min_dists_chunks = []
            for p_start in range(0, patches.shape[0], patch_chunk_size):
                p_chunk = patches[p_start : p_start + patch_chunk_size]
                d = torch.cdist(p_chunk, self.memory_bank)
                min_dists_chunks.append(d.min(dim=1)[0])
            min_dists = torch.cat(min_dists_chunks, dim=0)
            score_maps = min_dists.view(b, 1, h, w)  # (B, 1, H, W)

            # Upsample score map to image resolution for heatmaps
            upsampled = F.interpolate(
                score_maps,
                size=(self.config.input_size, self.config.input_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)  # (B, input_size, input_size)

            # Image-level score is the max patch score
            img_scores = score_maps.amax(dim=(-1, -2, -3)).cpu().numpy()
            all_scores.extend(img_scores.tolist())
            all_heatmaps.append(upsampled.cpu().numpy())

        heatmaps = np.concatenate(all_heatmaps, axis=0) if all_heatmaps else np.zeros((0, self.config.input_size, self.config.input_size))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return GateScore(
            image_scores=np.array(all_scores, dtype=np.float32),
            heatmaps=heatmaps,
            latency_ms=elapsed_ms,
        )

    def save(self, path: str | Path) -> None:
        """Serialize memory bank and config to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted gate.")
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self.config,
                "memory_bank": self.memory_bank.cpu(),
                "grid": self.grid,
            },
            save_path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "AnomalyGate":
        """Load fitted memory bank and config from disk."""
        data = torch.load(path, map_location="cpu", weights_only=False)
        cfg = data["config"]
        if device:
            cfg.device = device
        gate = cls(cfg)
        gate.memory_bank = data["memory_bank"].to(gate.device)
        gate.grid = data["grid"]
        return gate
