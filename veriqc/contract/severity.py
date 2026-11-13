"""Deterministic severity policy engine for quality inspection.

Evaluates defect severity (minor, major, critical) deterministically from:
1. Physical defect area (mm² converted from pixels via scale calibration)
2. Defect class risk multipliers
3. Zone criticality weighting (e.g. edge vs surface)
4. Anomaly score margin relative to the failure line

Rules are fully auditable, loaded from versioned YAML policies, and recorded
in the DefectRecord alongside step-by-step reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

__all__ = ["SeverityPolicy", "SeverityOutcome", "load_policy"]


@dataclass
class SeverityOutcome:
    severity: str
    reasons: list[str] = field(default_factory=list)


@dataclass
class SeverityPolicy:
    """Versioned severity rules.

    Expected YAML shape::

        version: 1
        mm_per_px: 0.25          # from the one-time scale calibration
        area_thresholds:          # mm^2 -> severity
            minor: 1.0
            major: 25.0
            critical: 100.0
        class_risk:               # multiplies the area thresholds
            scratches: 1.5
            crazing: 1.2
        zone_criticality:         # multiplies the area thresholds
            edge: 1.3
            center: 1.0
        score_margin:             # extra severity when the score is far above tau_fail
            critical_margin: 0.5
    """

    version: int = 1
    mm_per_px: float = 1.0
    area_thresholds: dict[str, float] = field(default_factory=lambda: {"minor": 1.0, "major": 25.0, "critical": 100.0})
    class_risk: dict[str, float] = field(default_factory=dict)
    zone_criticality: dict[str, float] = field(default_factory=dict)
    score_margin: dict[str, float] = field(default_factory=lambda: {"critical_margin": 0.5})
    source_path: str | None = None

    @classmethod
    def from_dict(cls, d: dict, source_path: str | None = None) -> "SeverityPolicy":
        return cls(
            version=int(d.get("version", 1)),
            mm_per_px=float(d.get("mm_per_px", 1.0)),
            area_thresholds=dict(d.get("area_thresholds", {})) or cls().area_thresholds,
            class_risk=dict(d.get("class_risk", {})),
            zone_criticality=dict(d.get("zone_criticality", {})),
            score_margin=dict(d.get("score_margin", {})) or cls().score_margin,
            source_path=source_path,
        )

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "mm_per_px": self.mm_per_px,
            "area_thresholds": self.area_thresholds,
            "class_risk": self.class_risk,
            "zone_criticality": self.zone_criticality,
            "score_margin": self.score_margin,
        }

    # -- evaluation --------------------------------------------------------
    def area_mm2(self, area_px: float | None) -> float | None:
        if area_px is None:
            return None
        return float(area_px) * self.mm_per_px * self.mm_per_px

    def severity(
        self,
        area_px: float | None = None,
        defect_class: str | None = None,
        zone: str | None = None,
        score: float | None = None,
        fail_threshold: float | None = None,
    ) -> SeverityOutcome:
        reasons: list[str] = []
        area_mm2 = self.area_mm2(area_px)
        if area_mm2 is None:
            return SeverityOutcome(severity="minor", reasons=["no mask area available"])

        th = dict(self.area_thresholds)
        risk = self.class_risk.get(defect_class or "", 1.0)
        crit = self.zone_criticality.get(zone or "", 1.0)
        if risk != 1.0:
            reasons.append(f"class risk x{risk:.2f} ({defect_class})")
        if crit != 1.0:
            reasons.append(f"zone criticality x{crit:.2f} ({zone})")

        eff_minor = th.get("minor", 0.0) * risk * crit
        eff_major = th.get("major", float("inf")) * risk * crit
        eff_critical = th.get("critical", float("inf")) * risk * crit

        if area_mm2 >= eff_critical:
            sev = "critical"
        elif area_mm2 >= eff_major:
            sev = "major"
        elif area_mm2 >= eff_minor:
            sev = "minor"
        else:
            sev = "minor"
        reasons.insert(0, f"mask area {area_mm2:.1f} mm^2 (thresholds {eff_minor:.1f}/{eff_major:.1f}/{eff_critical:.1f})")

        # score margin: a score far above the fail line escalates severity
        if score is not None and fail_threshold is not None:
            margin = float(self.score_margin.get("critical_margin", 0.0))
            if margin > 0 and score > fail_threshold * (1.0 + margin):
                if sev == "minor":
                    sev = "major"
                elif sev == "major":
                    sev = "critical"
                reasons.append(f"score {score:.3f} exceeds fail threshold by more than {margin:.0%}")

        return SeverityOutcome(severity=sev, reasons=reasons)


def load_policy(path: str | Path) -> SeverityPolicy:
    p = Path(path)
    d = yaml.safe_load(p.read_text()) or {}
    return SeverityPolicy.from_dict(d, source_path=str(p))
