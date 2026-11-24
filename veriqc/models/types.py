"""Shared model output types."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Detection", "AnomalyResult"]


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class AnomalyResult:
    image_score: float
    heatmap: np.ndarray
    latency_ms: float
