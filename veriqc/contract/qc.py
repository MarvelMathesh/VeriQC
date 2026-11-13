"""Optical quality assessment gate for edge inspection frames.

Evaluates frame blur (Laplacian variance), mean luminance, and saturation
against calibration acceptance bands derived from normal reference images.
Frames failing acceptance bands are flagged and routed to REVIEW ('re-image'),
preventing optical corruption and lighting drift from triggering false alarms.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["QualityBand", "QualityReport", "assess_quality"]


@dataclass
class QualityBand:
    """Acceptance band for one quality statistic."""

    lo: float
    hi: float
    name: str

    def contains(self, v: float) -> bool:
        return self.lo <= v <= self.hi


@dataclass
class QualityReport:
    ok: bool
    flags: list[str] = field(default_factory=list)
    stats: dict[str, float] = field(default_factory=dict)


@dataclass
class QualityCalibration:
    """Bands calibrated from normal images."""

    blur: QualityBand
    luminance: QualityBand
    saturation: QualityBand
    n_samples: int = 0

    def to_dict(self) -> dict:
        return {
            "blur": {"lo": self.blur.lo, "hi": self.blur.hi},
            "luminance": {"lo": self.luminance.lo, "hi": self.luminance.hi},
            "saturation": {"lo": self.saturation.lo, "hi": self.saturation.hi},
            "n_samples": self.n_samples,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "QualityCalibration":
        def band(name: str, d2: dict) -> QualityBand:
            return QualityBand(lo=float(d2["lo"]), hi=float(d2["hi"]), name=name)

        return cls(
            blur=band("blur", d["blur"]),
            luminance=band("luminance", d["luminance"]),
            saturation=band("saturation", d["saturation"]),
            n_samples=int(d.get("n_samples", 0)),
        )


def frame_stats(rgb: np.ndarray) -> dict[str, float]:
    """Blur (variance of Laplacian), mean luminance and mean saturation.

    ``rgb`` is HxWx3 uint8.
    """
    gray = rgb.astype(np.float32).mean(axis=2)
    lap = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    blur = float(lap.var())
    luminance = float(gray.mean())
    mx = rgb.max(axis=2).astype(np.float32)
    mn = rgb.min(axis=2).astype(np.float32)
    saturation = float((mx - mn).mean())
    return {"blur": blur, "luminance": luminance, "saturation": saturation}


def calibrate_quality(
    frames: list[np.ndarray],
    blur_quantiles: tuple[float, float] = (0.01, 0.99),
    lum_quantiles: tuple[float, float] = (0.01, 0.99),
    sat_quantiles: tuple[float, float] = (0.01, 0.99),
    blur_slack: float = 0.0,
) -> QualityCalibration:
    """Calibrate the acceptance bands from a set of good frames."""
    stats = [frame_stats(f) for f in frames]
    blur = np.array([s["blur"] for s in stats])
    lum = np.array([s["luminance"] for s in stats])
    sat = np.array([s["saturation"] for s in stats])
    return QualityCalibration(
        blur=QualityBand(
            lo=float(np.quantile(blur, blur_quantiles[0]) - blur_slack),
            hi=float(np.quantile(blur, blur_quantiles[1]) + blur_slack),
            name="blur",
        ),
        luminance=QualityBand(
            lo=float(np.quantile(lum, lum_quantiles[0])), hi=float(np.quantile(lum, lum_quantiles[1])), name="luminance"
        ),
        saturation=QualityBand(
            lo=float(np.quantile(sat, sat_quantiles[0])), hi=float(np.quantile(sat, sat_quantiles[1])), name="saturation"
        ),
        n_samples=len(stats),
    )


def assess_quality(rgb: np.ndarray | Image.Image, cal: QualityCalibration) -> QualityReport:
    """Check a frame against the calibrated bands."""
    if not isinstance(rgb, np.ndarray):
        rgb = np.asarray(rgb.convert("RGB"), dtype=np.uint8)
    st = frame_stats(rgb)
    flags: list[str] = []
    if not cal.blur.contains(st["blur"]):
        flags.append("blur")
    if not cal.luminance.contains(st["luminance"]):
        flags.append("luminance")
    if not cal.saturation.contains(st["saturation"]):
        flags.append("saturation")
    return QualityReport(ok=not flags, flags=flags, stats=st)
