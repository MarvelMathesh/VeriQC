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
from typing import Any

__all__ = ["ChannelCalibration", "ProfileCalibration"]


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
class ProfileCalibration:
    profile_id: str
    rule: str
    alpha_fail: float
    alpha_review: float
    anomaly: ChannelCalibration | None = None
    known: ChannelCalibration | None = None
    fused: ChannelCalibration | None = None
    screening_fail: float | None = None
    screening_review: float | None = None
    sample_sizes: dict[str, int] = field(default_factory=dict)
    note: str = ""

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha_fail < self.alpha_review <= 1.0):
            raise ValueError("calibration requires 0 < alpha_fail < alpha_review <= 1")
        if self.rule not in ("known_only", "anomaly_only", "dual"):
            raise ValueError(f"unknown decision rule: {self.rule!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "veriqc.calibration/1",
            "profile_id": self.profile_id,
            "rule": self.rule,
            "alpha_fail": self.alpha_fail,
            "alpha_review": self.alpha_review,
            "anomaly": self.anomaly.to_dict() if self.anomaly else None,
            "known": self.known.to_dict() if self.known else None,
            "fused": self.fused.to_dict() if self.fused else None,
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
            fused=ChannelCalibration.from_dict(data["fused"]) if data.get("fused") else None,
            screening_fail=(float(data["screening_fail"]) if data.get("screening_fail") is not None else None),
            screening_review=(float(data["screening_review"]) if data.get("screening_review") is not None else None),
            sample_sizes={str(k): int(v) for k, v in data.get("sample_sizes", {}).items()},
            note=str(data.get("note", "")),
        )
