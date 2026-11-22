from __future__ import annotations

from veriqc.calibration import ChannelCalibration, ProfileCalibration
from veriqc.decision import decide


def screening() -> ProfileCalibration:
    return ProfileCalibration(
        profile_id="steel",
        rule="known_only",
        alpha_fail=0.1,
        alpha_review=0.3,
        screening_fail=0.8,
        screening_review=0.4,
    )


def anomaly() -> ProfileCalibration:
    return ProfileCalibration(
        profile_id="carpet",
        rule="anomaly_only",
        alpha_fail=0.05,
        alpha_review=0.10,
        anomaly=ChannelCalibration(scores=list(range(1, 101))),
    )


def dual() -> ProfileCalibration:
    return ProfileCalibration(
        profile_id="commutator",
        rule="dual",
        alpha_fail=0.05,
        alpha_review=0.10,
        known=ChannelCalibration(scores=list(range(1, 101))),
        anomaly=ChannelCalibration(scores=list(range(1, 101))),
        fused=ChannelCalibration(scores=[0.0] * 100),
    )


def test_screening_verdicts() -> None:
    assert decide(screening(), known_score=0.9, anomaly_score=None).verdict == "FAIL"
    assert decide(screening(), known_score=0.5, anomaly_score=None).verdict == "REVIEW"
    assert decide(screening(), known_score=0.1, anomaly_score=None).verdict == "PASS"
    assert decide(screening(), known_score=None, anomaly_score=None).verdict == "REVIEW"


def test_anomaly_verdicts() -> None:
    assert decide(anomaly(), known_score=None, anomaly_score=1000.0).verdict == "FAIL"
    assert decide(anomaly(), known_score=None, anomaly_score=95.0).verdict == "REVIEW"
    assert decide(anomaly(), known_score=None, anomaly_score=0.0).verdict == "PASS"
    assert decide(anomaly(), known_score=None, anomaly_score=None).verdict == "REVIEW"


def test_dual_missing_channel_fails_closed() -> None:
    assert decide(dual(), known_score=1.0, anomaly_score=None).verdict == "REVIEW"
    assert decide(dual(), known_score=None, anomaly_score=1.0).verdict == "REVIEW"


def test_non_finite_scores_fail_closed() -> None:
    assert decide(screening(), known_score=float("nan"), anomaly_score=None).verdict == "REVIEW"
    assert decide(anomaly(), known_score=None, anomaly_score=float("nan")).verdict == "REVIEW"
    assert decide(dual(), known_score=float("nan"), anomaly_score=1.0).verdict == "REVIEW"


def test_dual_uses_the_strongest_channel() -> None:
    result = decide(dual(), known_score=1.0, anomaly_score=1.0)
    assert result.verdict == "PASS"
    result = decide(dual(), known_score=1000.0, anomaly_score=1000.0)
    assert result.verdict == "FAIL"
    assert result.known_p_value is not None and result.anomaly_p_value is not None
