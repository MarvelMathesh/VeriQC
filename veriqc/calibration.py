"""Per-profile empirical calibration.

Thresholds are order statistics of held-out normal scores.  The p-value is the
conservative split-conformal tail value, which makes REVIEW/FAIL decisions
comparable across channels without assuming a score distribution.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

__all__ = ["ChannelCalibration", "RegionCalibration", "ProfileCalibration"]


@dataclass
class ChannelCalibration:
    scores: list[float] = field(default_factory=list)
    mode: str = "scores"

    def p_value(self, score: float) -> float:
        if not self.scores:
            return 1.0
        greater = sum(1 for value in self.scores if value >= score)
        return (1.0 + greater) / (len(self.scores) + 1.0)

    def threshold_for_alpha(self, alpha: float) -> float:
        if not self.scores:
            return float("inf")
        ordered = sorted(self.scores)
        index = min(len(ordered) - 1, max(0, int(math.ceil((len(ordered) + 1) * (1.0 - alpha))) - 1))
        return float(ordered[index])

    def to_dict(self) -> dict[str, Any]:
        return {"scores": [round(float(v), 8) for v in self.scores], "mode": self.mode}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChannelCalibration":
        return cls(
            scores=[float(v) for v in data.get("scores", [])],
            mode=str(data.get("mode", "scores")),
        )


@dataclass
class RegionCalibration:
    """Empirical normal-region tail for localisation thresholds."""

    scores: list[float] = field(default_factory=list)
    threshold: float = float("inf")
    min_area_px: int = 8
    false_regions_per_image: float | None = None
    method: str = "normal_region_false_positive"

    def p_value(self, score: float) -> float:
        if not self.scores:
            return 1.0
        greater = sum(1 for value in self.scores if value >= score)
        return (1.0 + greater) / (len(self.scores) + 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scores": [round(float(v), 8) for v in self.scores],
            "threshold": round(float(self.threshold), 8),
            "min_area_px": int(self.min_area_px),
            "false_regions_per_image": (
                round(float(self.false_regions_per_image), 6)
                if self.false_regions_per_image is not None
                else None
            ),
            "method": self.method,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RegionCalibration":
        return cls(
            scores=[float(v) for v in data.get("scores", [])],
            threshold=float(data.get("threshold", float("inf"))),
            min_area_px=int(data.get("min_area_px", 8)),
            false_regions_per_image=(
                float(data["false_regions_per_image"])
                if data.get("false_regions_per_image") is not None
                else None
            ),
            method=str(data.get("method", "normal_region_false_positive")),
        )


def fit_region_calibration(
    maps: Iterable[np.ndarray],
    *,
    min_area_px: int = 8,
    prior_threshold: float = 0.5,
    target_false_regions: float = 0.5,
    quantiles: int = 60,
) -> RegionCalibration:
    """Choose a localisation threshold from normal-region statistics.

    The image-level operating threshold is the starting point.  It is raised
    only when the mean number of connected false regions per normal image
    exceeds the target.  The measured rate and any surviving normal-region
    maxima are recorded, so an empty null sample is visible rather than
    presented as a significance claim.
    """

    values = [np.asarray(item, dtype=np.float32) for item in maps]
    values = [item for item in values if item.size]
    if not values:
        return RegionCalibration(threshold=float(prior_threshold), min_area_px=min_area_px)
    pooled = np.concatenate([item.reshape(-1) for item in values])

    def measure(threshold: float) -> tuple[list[float], float]:
        counts: list[int] = []
        scores: list[float] = []
        for item in values:
            hot = (item >= threshold).astype(np.uint8)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(hot, connectivity=8)
            regions = 0
            for index in range(1, count):
                if stats[index, cv2.CC_STAT_AREA] < min_area_px:
                    continue
                regions += 1
                scores.append(float(item[labels == index].max()))
            counts.append(regions)
        return scores, float(np.mean(counts)) if counts else 0.0

    # Start from the image-level operating threshold.  Only raise it if the
    # normal false-region rate is unacceptable; never trade away localisation
    # sensitivity merely because a high threshold has an empty null sample.
    scores, false_rate = measure(float(prior_threshold))
    if false_rate <= target_false_regions:
        return RegionCalibration(
            scores=scores,
            threshold=float(prior_threshold),
            min_area_px=min_area_px,
            false_regions_per_image=false_rate,
        )

    candidates = np.unique(
        np.concatenate(
            [
                np.quantile(pooled, np.linspace(0.90, 0.9999, quantiles)),
                np.asarray([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99], dtype=np.float32),
            ]
        )
    )
    for threshold in np.sort(candidates):
        if threshold <= prior_threshold:
            continue
        candidate_scores, candidate_rate = measure(float(threshold))
        if candidate_rate <= target_false_regions:
            return RegionCalibration(
                scores=candidate_scores,
                threshold=float(threshold),
                min_area_px=min_area_px,
                false_regions_per_image=candidate_rate,
            )
    return RegionCalibration(
        scores=scores,
        threshold=float(prior_threshold),
        min_area_px=min_area_px,
        false_regions_per_image=false_rate,
    )


@dataclass
class ProfileCalibration:
    profile_id: str
    rule: str
    alpha_fail: float
    alpha_review: float
    anomaly: ChannelCalibration | None = None
    known: ChannelCalibration | None = None
    region: RegionCalibration | None = None
    class_thresholds: dict[str, float] = field(default_factory=dict)
    screening_fail: float | None = None
    screening_review: float | None = None
    sample_sizes: dict[str, int] = field(default_factory=dict)
    note: str = ""

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha_fail < self.alpha_review <= 1.0):
            raise ValueError("calibration requires 0 < alpha_fail < alpha_review <= 1")
        if self.rule not in ("known_only", "anomaly_only"):
            raise ValueError(f"unknown decision rule: {self.rule!r}")
        if self.screening_fail is not None and self.screening_review is not None:
            if self.screening_fail <= self.screening_review:
                raise ValueError(
                    "screening_fail must be greater than screening_review for a "
                    "high-score defect decision"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "veriqc.calibration/1",
            "profile_id": self.profile_id,
            "rule": self.rule,
            "alpha_fail": self.alpha_fail,
            "alpha_review": self.alpha_review,
            "anomaly": self.anomaly.to_dict() if self.anomaly else None,
            "known": self.known.to_dict() if self.known else None,
            "region": self.region.to_dict() if self.region else None,
            "class_thresholds": {str(k): round(float(v), 8) for k, v in self.class_thresholds.items()},
            "screening_fail": self.screening_fail,
            "screening_review": self.screening_review,
            "sample_sizes": dict(self.sample_sizes),
            "note": self.note,
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "ProfileCalibration":
        data = json.loads(Path(path).read_text())
        return cls(
            profile_id=str(data["profile_id"]),
            rule=str(data["rule"]),
            alpha_fail=float(data["alpha_fail"]),
            alpha_review=float(data["alpha_review"]),
            anomaly=ChannelCalibration.from_dict(data["anomaly"]) if data.get("anomaly") else None,
            known=ChannelCalibration.from_dict(data["known"]) if data.get("known") else None,
            region=RegionCalibration.from_dict(data["region"]) if data.get("region") else None,
            class_thresholds={
                str(k): float(v) for k, v in data.get("class_thresholds", {}).items()
            },
            screening_fail=(float(data["screening_fail"]) if data.get("screening_fail") is not None else None),
            screening_review=(float(data["screening_review"]) if data.get("screening_review") is not None else None),
            sample_sizes={str(k): int(v) for k, v in data.get("sample_sizes", {}).items()},
            note=str(data.get("note", "")),
        )
