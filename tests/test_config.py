from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from veriqc.calibration import ProfileCalibration
from veriqc.config import load_profile

REPO = Path(__file__).resolve().parents[1]
PROFILES = sorted((REPO / "configs" / "profiles").glob("*.yaml"))


@pytest.mark.parametrize("path", PROFILES, ids=lambda path: path.stem)
def test_profile_and_calibration_rules_are_consistent(path: Path) -> None:
    profile = load_profile(path, validate=False)
    calibration = ProfileCalibration.load(profile.calibration_path)
    expected = {
        "detect": "known_only",
        "segment": "known_only",
        "anomaly": "anomaly_only",
    }[profile.task]
    assert calibration.profile_id == profile.profile_id
    assert calibration.rule == expected


def test_task_channel_consistency_is_enforced() -> None:
    profile = load_profile(REPO / "configs" / "profiles" / "commutator.yaml", validate=False)
    with pytest.raises(ValueError):
        replace(profile, task="anomaly").validate()
    with pytest.raises(ValueError):
        replace(profile, task="segment+anomaly").validate()


def test_artifact_hash_mismatch_is_rejected() -> None:
    profile = load_profile(REPO / "configs" / "profiles" / "commutator.yaml", validate=False)
    if not all(path.exists() for path in profile.artifact_paths().values()):
        pytest.skip("commutator runtime artifacts are not present")
    expected = dict(profile.expected_hashes)
    expected[next(iter(expected))] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        replace(profile, expected_hashes=expected).validate()


def test_quality_defaults_are_shared() -> None:
    profile = load_profile(REPO / "configs" / "profiles" / "commutator.yaml", validate=False)
    assert profile.quality.max_pixels > 0
    assert profile.quality.blur_ratio_min > 0
