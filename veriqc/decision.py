"""Fail-closed decision rule over calibrated channel scores."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from veriqc.calibration import ProfileCalibration

__all__ = ["Decision", "decide"]


@dataclass(frozen=True)
class Decision:
    verdict: str
    rule: str
    known_score: float | None = None
    anomaly_score: float | None = None
    known_p_value: float | None = None
    anomaly_p_value: float | None = None
    fused_p_value: float | None = None
    thresholds: dict[str, float] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()


def _usable(score: float | None) -> bool:
    return score is not None and math.isfinite(float(score))


def decide(
    calibration: ProfileCalibration,
    *,
    known_score: float | None,
    anomaly_score: float | None,
    known_ok: bool = True,
    anomaly_ok: bool = True,
) -> Decision:
    """Apply the profile's rule.  A missing or non-finite channel is REVIEW."""

    rule = calibration.rule
    reasons: list[str] = []

    if rule == "known_only":
        if not known_ok or not _usable(known_score):
            return Decision(
                verdict="REVIEW",
                rule=rule,
                known_score=known_score,
                thresholds=_thresholds(calibration),
                reasons=("known channel unavailable or non-finite",),
            )
        fail = calibration.screening_fail
        review = calibration.screening_review
        if fail is None or review is None:
            raise RuntimeError("screening calibration has no thresholds")
        thresholds = {"known_fail": fail, "known_review": review}
        if known_score >= fail:
            verdict = "FAIL"
        elif known_score >= review:
            verdict = "REVIEW"
            reasons.append("known-channel score is in the review band")
        else:
            verdict = "PASS"
        return Decision(
            verdict=verdict,
            rule=rule,
            known_score=known_score,
            thresholds=thresholds,
            reasons=tuple(reasons),
        )

    if rule == "anomaly_only":
        if not anomaly_ok or not _usable(anomaly_score) or calibration.anomaly is None:
            return Decision(
                verdict="REVIEW",
                rule=rule,
                anomaly_score=anomaly_score,
                thresholds=_thresholds(calibration),
                reasons=("anomaly channel unavailable or non-finite",),
            )
        p_value = calibration.anomaly.p_value(anomaly_score)
        if p_value <= calibration.alpha_fail:
            verdict = "FAIL"
        elif p_value <= calibration.alpha_review:
            verdict = "REVIEW"
            reasons.append("anomaly p-value is in the review band")
        else:
            verdict = "PASS"
        return Decision(
            verdict=verdict,
            rule=rule,
            anomaly_score=anomaly_score,
            anomaly_p_value=p_value,
            thresholds=_thresholds(calibration),
            reasons=tuple(reasons),
        )

    # dual: the fused statistic is calibrated on paired normal scores.
    if not known_ok or not _usable(known_score) or calibration.known is None:
        return Decision(
            verdict="REVIEW",
            rule=rule,
            known_score=known_score,
            anomaly_score=anomaly_score,
            thresholds=_thresholds(calibration),
            reasons=("known channel unavailable or non-finite",),
        )
    if not anomaly_ok or not _usable(anomaly_score) or calibration.anomaly is None:
        return Decision(
            verdict="REVIEW",
            rule=rule,
            known_score=known_score,
            anomaly_score=anomaly_score,
            thresholds=_thresholds(calibration),
            reasons=("anomaly channel unavailable or non-finite",),
        )
    if calibration.fused is None:
        raise RuntimeError("dual calibration has no fused null distribution")
    known_p = calibration.known.p_value(known_score)
    anomaly_p = calibration.anomaly.p_value(anomaly_score)
    fused_p = calibration.fused.p_value(1.0 - min(known_p, anomaly_p))
    if fused_p <= calibration.alpha_fail:
        verdict = "FAIL"
    elif fused_p <= calibration.alpha_review:
        verdict = "REVIEW"
        reasons.append("the fused channel p-value is in the review band")
    else:
        verdict = "PASS"
    return Decision(
        verdict=verdict,
        rule=rule,
        known_score=known_score,
        anomaly_score=anomaly_score,
        known_p_value=known_p,
        anomaly_p_value=anomaly_p,
        fused_p_value=fused_p,
        thresholds=_thresholds(calibration),
        reasons=tuple(reasons),
    )


def _thresholds(calibration: ProfileCalibration) -> dict[str, float]:
    values: dict[str, float] = {
        "alpha_fail": calibration.alpha_fail,
        "alpha_review": calibration.alpha_review,
    }
    if calibration.anomaly is not None and calibration.anomaly.scores:
        values["anomaly_fail"] = calibration.anomaly.threshold_for_alpha(calibration.alpha_fail)
        values["anomaly_review"] = calibration.anomaly.threshold_for_alpha(calibration.alpha_review)
    if calibration.known is not None and calibration.known.scores:
        values["known_fail"] = calibration.known.threshold_for_alpha(calibration.alpha_fail)
        values["known_review"] = calibration.known.threshold_for_alpha(calibration.alpha_review)
    if calibration.screening_fail is not None:
        values["known_fail"] = calibration.screening_fail
    if calibration.screening_review is not None:
        values["known_review"] = calibration.screening_review
    if calibration.fused is not None and calibration.fused.scores:
        values["fused_fail"] = calibration.fused.threshold_for_alpha(calibration.alpha_fail)
        values["fused_review"] = calibration.fused.threshold_for_alpha(calibration.alpha_review)
    return values
