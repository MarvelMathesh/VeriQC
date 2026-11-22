"""Acquisition quality gate.

The statistics are deliberately simple and gain-invariant where possible:
Laplacian variance divided by grey variance, plus the fraction of pixels at
the sensor limits.  A failed gate sends the frame to ``REVIEW`` before any
model runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

__all__ = ["QualitySpec", "QualityReport", "assess_quality"]


@dataclass(frozen=True)
class QualitySpec:
    min_width: int = 64
    min_height: int = 64
    blur_ratio_min: float = 0.02
    clipped_max: float = 0.25
    max_pixels: int = 25_000_000


@dataclass(frozen=True)
class QualityReport:
    ok: bool
    flags: tuple[str, ...]
    statistics: dict[str, float | int]


def assess_quality(image: Image.Image, spec: QualitySpec | None = None) -> QualityReport:
    spec = spec or QualitySpec()
    width, height = image.size
    flags: list[str] = []
    pixels = width * height
    if width < spec.min_width or height < spec.min_height:
        flags.append("resolution")
    if pixels > spec.max_pixels:
        flags.append("pixel_count")
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    variance = float(gray.var())
    laplacian = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    blur_ratio = laplacian / variance if variance > 1e-6 else 0.0
    clipped = float(((gray <= 2) | (gray >= 253)).mean())
    if blur_ratio < spec.blur_ratio_min:
        flags.append("blur")
    if clipped > spec.clipped_max:
        flags.append("clipped")
    return QualityReport(
        ok=not flags,
        flags=tuple(flags),
        statistics={
            "width": width,
            "height": height,
            "pixels": pixels,
            "blur_ratio": round(blur_ratio, 6),
            "clipped_fraction": round(clipped, 6),
            "grey_variance": round(variance, 2),
        },
    )
