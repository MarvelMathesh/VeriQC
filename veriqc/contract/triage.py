"""The triage engine: two evidence channels, three verdicts, one error budget.

Combines the normals-only anomaly gate score and the supervised detector findings
under split-conformal calibration thresholds:

- PASS: anomaly score is at or below the calibrated review threshold.
- FAIL: confirmed defect with high anomaly score and confirmed class above floor.
- REVIEW: borderline scores, ambiguous classes, or disagreement between channels.

Supported triage rules:
- ``agreement``: production default; a FAIL requires both channels to agree,
  demoting discrepancies to REVIEW rather than making false alarm decisions.
- ``class_floor``: agreement rule with per-class confidence floors.
- ``score_only``: anomaly channel only (ablation baseline).
- ``union``: either channel triggers a FAIL (ablation baseline).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from veriqc.contract.record import DefectRecord

__all__ = ["TriageConfig", "TriageDecision", "triage", "RULES"]

RuleName = Literal["veriqc_md", "agreement", "class_floor", "score_only"]

RULES: tuple[str, ...] = ("veriqc_md", "agreement", "class_floor", "score_only")


@dataclass
class TriageConfig:
    """Everything the triage engine needs that is not evidence."""

    alpha_review: float = 0.05
    alpha_fail: float = 0.01
    rule: str = "agreement"
    #: per-class confidence floors; classes missing from the map use `default_floor`
    class_floors: dict[str, float] = field(default_factory=dict)
    default_floor: float = 0.25
    #: relative hysteresis band around the thresholds (0 disables it)
    hysteresis: float = 0.0
    #: if True a class below its floor never produces a FAIL, only REVIEW
    demote_weak_class: bool = True

    def floor_for(self, defect_class: str | None) -> float:
        if defect_class is None:
            return self.default_floor
        return self.class_floors.get(defect_class, self.default_floor)


@dataclass
class TriageDecision:
    """Outcome of the triage engine."""

    verdict: str
    reasons: list[str] = field(default_factory=list)
    severity_candidate: bool = False
    class_confirmed: bool = False


def _hysteresis_band(value: float, threshold: float, rel: float) -> tuple[float, float]:
    """Widen a threshold into a band so small score jitter cannot flip a verdict."""
    if rel <= 0:
        return threshold, threshold
    return threshold * (1.0 - rel), threshold * (1.0 + rel)


def triage(
    anomaly_score: float | None,
    detection: dict | None,
    thresholds: dict[str, float],
    config: TriageConfig,
    qc_ok: bool = True,
    qc_flags: list[str] | None = None,
) -> TriageDecision:
    """Turn the two evidence channels into a verdict.

    Parameters
    ----------
    anomaly_score : the gate's image score, or None if the gate did not run
    detection : dict with keys ``class`` (str|None), ``confidence`` (float),
        ``bbox``; None if the detector did not run
    thresholds : {'review': tau_review, 'fail': tau_fail}
    """
    flags = qc_flags or []
    reasons: list[str] = []

    if not qc_ok:
        return TriageDecision(
            verdict="REVIEW",
            reasons=[f"image quality gate failed ({', '.join(flags)})"] if flags else ["image quality gate failed"],
        )

    det = detection or {}
    det_class = det.get("class")
    det_conf = float(det.get("confidence") or 0.0)
    floor = config.floor_for(det_class)
    class_confirmed = det_class is not None and det_conf >= floor
    if det_class is not None and not class_confirmed:
        reasons.append(
            f"class {det_class} confidence {det_conf:.2f} below floor {floor:.2f}"
        )

    tau_review = thresholds.get("review")
    tau_fail = thresholds.get("fail")
    s = anomaly_score

    if tau_fail is None or tau_review is None:
        raise ValueError("both review and fail thresholds are required")

    # hysteresis: thresholds are relaxed for a FAIL and tightened for a PASS
    fail_lo, fail_hi = _hysteresis_band(tau_fail, tau_fail, config.hysteresis)
    rev_lo, rev_hi = _hysteresis_band(tau_review, tau_review, config.hysteresis)

    score_high = s is not None and s > fail_lo
    score_low = s is not None and s <= rev_hi

    rule = config.rule
    verdict = "PASS"
    if rule == "score_only":
        if score_high:
            verdict = "FAIL"
            reasons.append(f"anomaly score {s:.3f} above fail threshold {tau_fail:.3f}")
        elif s is not None and s > rev_lo:
            verdict = "REVIEW"
            reasons.append(f"anomaly score {s:.3f} above review threshold {tau_review:.3f}")
        else:
            reasons.append(f"anomaly score {s:.3f} at or below review threshold {tau_review:.3f}")

    elif rule in ("union", "veriqc_md"):
        # union of the channels: either channel can raise a FAIL
        if score_high:
            verdict = "FAIL"
            reasons.append(f"anomaly score {s:.3f} above fail threshold {tau_fail:.3f}")
        elif class_confirmed:
            verdict = "FAIL"
            reasons.append(f"classifier confirmed {det_class} at {det_conf:.2f}")
        elif s is not None and s > rev_lo:
            verdict = "REVIEW"
            reasons.append(f"anomaly score {s:.3f} above review threshold {tau_review:.3f}")
        else:
            reasons.append("no channel raised a finding")

    elif rule in ("agreement", "class_floor"):
        # intersection of the channels: a FAIL needs both
        if score_high and class_confirmed:
            verdict = "FAIL"
            reasons.append(
                f"anomaly score {s:.3f} above {tau_fail:.3f} and {det_class} confirmed at {det_conf:.2f}"
            )
        elif score_high and not class_confirmed:
            verdict = "REVIEW"
            reasons.append(
                f"anomaly score {s:.3f} above fail threshold but no confirmed class "
                f"(best {det_class or 'none'} at {det_conf:.2f} vs floor {floor:.2f})"
            )
        elif class_confirmed and s is not None and s > rev_lo:
            verdict = "REVIEW"
            reasons.append(f"class {det_class} confirmed but anomaly score {s:.3f} below fail threshold")
        elif s is not None and s > rev_lo:
            verdict = "REVIEW"
            reasons.append(f"anomaly score {s:.3f} above review threshold {tau_review:.3f}")
        else:
            reasons.append("both channels below their thresholds")

    else:
        raise ValueError(f"unknown triage rule {rule!r}; expected one of {RULES}")

    if verdict == "FAIL" and config.demote_weak_class and det_class is not None:
        if not class_confirmed and rule in ("veriqc_md",):
            verdict = "REVIEW"
            reasons.append("demoted to REVIEW: class below its confidence floor")

    return TriageDecision(
        verdict=verdict,
        reasons=reasons,
        severity_candidate=verdict == "FAIL",
        class_confirmed=class_confirmed,
    )
