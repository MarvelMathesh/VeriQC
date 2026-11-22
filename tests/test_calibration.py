from __future__ import annotations

import pytest

from veriqc.calibration import ChannelCalibration, ProfileCalibration


def test_p_value_is_conservative_and_monotone() -> None:
    channel = ChannelCalibration(scores=[1.0, 2.0, 3.0, 4.0])
    assert channel.p_value(0.0) == pytest.approx(1.0)
    assert channel.p_value(5.0) == pytest.approx(0.2)
    assert channel.p_value(2.0) == pytest.approx(0.8)
    assert channel.p_value(2.5) < channel.p_value(2.0)


def test_threshold_orders_are_valid() -> None:
    channel = ChannelCalibration(scores=list(range(1, 101)))
    threshold = channel.threshold_for_alpha(0.05)
    assert threshold == pytest.approx(96.0)


def test_calibration_roundtrip(tmp_path) -> None:
    calibration = ProfileCalibration(
        profile_id="p",
        rule="dual",
        alpha_fail=0.05,
        alpha_review=0.10,
        known=ChannelCalibration(scores=[0.1, 0.2]),
        anomaly=ChannelCalibration(scores=[0.3, 0.4]),
        sample_sizes={"known_cal": 2, "anomaly_cal": 2},
    )
    path = tmp_path / "cal.json"
    calibration.save(path)
    loaded = ProfileCalibration.load(path)
    assert loaded.profile_id == "p"
    assert loaded.rule == "dual"
    assert loaded.anomaly is not None
    assert loaded.anomaly.scores == [0.3, 0.4]


def test_invalid_alpha_rejected() -> None:
    with pytest.raises(ValueError):
        ProfileCalibration("p", "dual", 0.1, 0.05)
