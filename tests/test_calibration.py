from __future__ import annotations

import numpy as np
import pytest

from veriqc.calibration import (
    ChannelCalibration,
    ProfileCalibration,
    RegionCalibration,
    fit_region_calibration,
)


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
        rule="anomaly_only",
        alpha_fail=0.05,
        alpha_review=0.10,
        anomaly=ChannelCalibration(scores=[0.3, 0.4]),
        sample_sizes={"anomaly_cal": 2},
    )
    path = tmp_path / "cal.json"
    calibration.save(path)
    loaded = ProfileCalibration.load(path)
    assert loaded.profile_id == "p"
    assert loaded.rule == "anomaly_only"
    assert loaded.anomaly is not None
    assert loaded.anomaly.scores == [0.3, 0.4]


def test_invalid_alpha_rejected() -> None:
    with pytest.raises(ValueError):
        ProfileCalibration("p", "anomaly_only", 0.1, 0.05)


def test_screening_threshold_order_is_enforced() -> None:
    with pytest.raises(ValueError, match="screening_fail"):
        ProfileCalibration(
            "p",
            "known_only",
            0.1,
            0.3,
            screening_fail=0.2,
            screening_review=0.4,
        )


def test_region_calibration_uses_prior_when_normal_is_quiet() -> None:
    normal = np.zeros((32, 32), dtype=np.float32)
    normal[10:12, 10:12] = 0.2
    calibration = fit_region_calibration(
        [normal], prior_threshold=0.5, target_false_regions=0.5
    )
    assert calibration.threshold == pytest.approx(0.5)
    assert calibration.false_regions_per_image == 0.0
    assert calibration.p_value(0.8) <= 1.0


def test_region_calibration_raises_threshold_when_prior_is_noisy() -> None:
    noisy = np.zeros((32, 32), dtype=np.float32)
    noisy[4:12, 4:12] = 0.8
    calibration = fit_region_calibration(
        [noisy], prior_threshold=0.5, target_false_regions=0.5
    )
    assert calibration.threshold > 0.5
    assert calibration.false_regions_per_image is not None


def test_region_calibration_roundtrip() -> None:
    region = RegionCalibration(scores=[0.2, 0.4], threshold=0.5, false_regions_per_image=0.1)
    assert region.p_value(0.4) == pytest.approx(0.6666667)
    assert RegionCalibration.from_dict(region.to_dict()) == region
