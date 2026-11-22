"""Ordinal severity policy.

Severity is a function of defect type, relative area and location.  Confidence
is deliberately not an input.  The policy is an application decision, not a
physical measurement, and is reported as such.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

__all__ = ["SeverityPolicy", "SeverityGrade", "zone_for_bbox"]


@dataclass(frozen=True)
class SeverityGrade:
    severity: str
    score: float
    basis: dict[str, float | str]


@dataclass(frozen=True)
class SeverityPolicy:
    area_thresholds: Mapping[str, float]
    class_risk: Mapping[str, float]
    zone_criticality: Mapping[str, float]
    grade_thresholds: Mapping[str, float]
    default_risk: float = 1.0

    @classmethod
    def load(cls, path: str | Path) -> "SeverityPolicy":
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls(
            area_thresholds={str(k): float(v) for k, v in data["area_thresholds"].items()},
            class_risk={str(k): float(v) for k, v in data.get("class_risk", {}).items()},
            zone_criticality={str(k): float(v) for k, v in data.get("zone_criticality", {}).items()},
            grade_thresholds={str(k): float(v) for k, v in data["grade_thresholds"].items()},
            default_risk=float(data.get("default_risk", 1.0)),
        )

    def grade(self, defect_class: str, area_ratio: float, zone: str | None) -> SeverityGrade:
        minor = float(self.area_thresholds["minor"])
        major = float(self.area_thresholds["major"])
        critical = float(self.area_thresholds["critical"])
        if area_ratio < minor:
            base = 0.5
        elif area_ratio < major:
            base = 1.0
        elif area_ratio < critical:
            base = 2.0
        else:
            base = 3.0
        risk = float(self.class_risk.get(defect_class, self.default_risk))
        zone_factor = float(self.zone_criticality.get(zone or "center", 1.0))
        score = base * risk * zone_factor
        if score >= float(self.grade_thresholds["critical"]):
            severity = "critical"
        elif score >= float(self.grade_thresholds["major"]):
            severity = "major"
        else:
            severity = "minor"
        return SeverityGrade(
            severity=severity,
            score=round(score, 4),
            basis={
                "area_ratio": round(float(area_ratio), 8),
                "area_band": "minor" if base == 0.5 else "major" if base == 1.0 else "critical" if base == 2.0 else "extreme",
                "class_risk": risk,
                "zone": zone or "center",
                "zone_factor": zone_factor,
            },
        )


def zone_for_bbox(
    bbox: tuple[float, float, float, float],
    image_size: tuple[int, int],
    edge_band: float = 0.12,
    corner_band: float = 0.22,
) -> str:
    width, height = image_size
    if width <= 0 or height <= 0:
        return "center"
    x = 0.5 * (bbox[0] + bbox[2]) / width
    y = 0.5 * (bbox[1] + bbox[3]) / height
    dx = abs(x - 0.5)
    dy = abs(y - 0.5)
    if dx > 0.5 - corner_band and dy > 0.5 - corner_band:
        return "corner"
    if dx > 0.5 - edge_band or dy > 0.5 - edge_band:
        return "edge"
    return "center"
