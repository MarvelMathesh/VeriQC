from __future__ import annotations

from veriqc.severity import SeverityPolicy, zone_for_bbox


def policy() -> SeverityPolicy:
    return SeverityPolicy(
        area_thresholds={"minor": 0.001, "major": 0.01, "critical": 0.05},
        class_risk={"crack": 1.5, "scratch": 1.0},
        zone_criticality={"center": 1.0, "edge": 1.3, "corner": 1.4},
        grade_thresholds={"major": 1.5, "critical": 3.0},
    )


def test_severity_increases_with_area_and_risk() -> None:
    low = policy().grade("crack", 0.0001, "center")
    high = policy().grade("crack", 0.2, "corner")
    assert low.severity == "minor"
    assert high.severity == "critical"
    assert high.score > low.score


def test_confidence_is_not_an_input() -> None:
    first = policy().grade("crack", 0.002, "edge")
    second = policy().grade("crack", 0.002, "edge")
    assert first == second


def test_zone_geometry() -> None:
    assert zone_for_bbox((40, 40, 60, 60), (100, 100)) == "center"
    assert zone_for_bbox((0, 40, 5, 60), (100, 100)) == "edge"
    assert zone_for_bbox((0, 0, 5, 5), (100, 100)) == "corner"
