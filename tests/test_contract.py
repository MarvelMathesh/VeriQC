"""Unit tests for the VeriQC contract: DefectRecord, QC, Severity, and Triage."""

from __future__ import annotations

import json
import pytest
from PIL import Image
import numpy as np

from veriqc.contract.qc import QualityBand, QualityCalibration, assess_quality
from veriqc.contract.record import DefectRecord
from veriqc.contract.severity import SeverityPolicy
from veriqc.contract.triage import TriageConfig, triage


def test_defect_record_canonical_json_and_hash():
    r1 = DefectRecord(part_id="P-001", verdict="PASS", timestamp=100.0)
    r2 = DefectRecord(part_id="P-001", verdict="PASS", timestamp=100.0)
    assert r1.to_json() == r2.to_json()
    assert r1.content_hash() == r2.content_hash()

    r3 = DefectRecord(part_id="P-001", verdict="FAIL", timestamp=100.0)
    assert r1.content_hash() != r3.content_hash()


def test_defect_record_helpers():
    r = DefectRecord(
        part_id="P-002",
        verdict="FAIL",
        defect_class="scratches",
        class_confidence=0.88,
        anomaly_score=0.92,
        mask_area_mm2=14.5,
        defect_bbox=(10.0, 20.0, 50.0, 60.0),
    )
    numbers = r.numbers()
    assert 0.88 in numbers
    assert 0.92 in numbers
    assert 14.5 in numbers
    assert 10.0 in numbers
    assert r.class_words() == ["scratches"]


def test_qc_gate():
    calib = QualityCalibration(
        blur=QualityBand(lo=0.0, hi=1000.0, name="blur"),
        luminance=QualityBand(lo=20.0, hi=240.0, name="luminance"),
        saturation=QualityBand(lo=0.0, hi=255.0, name="saturation"),
    )
    # Good image
    good_img = Image.fromarray(np.full((100, 100, 3), 128, dtype=np.uint8))
    report = assess_quality(good_img, calib)
    assert report.ok is True

    # Completely black image (underexposed)
    black_img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    report_black = assess_quality(black_img, calib)
    assert report_black.ok is False
    assert any("luminance" in f for f in report_black.flags)


def test_triage_agreement_rule():
    cfg = TriageConfig(
        rule="agreement",
        class_floors={"scratches": 0.50},
        default_floor=0.30,
    )
    thresholds = {"review": 0.30, "fail": 0.60}

    # Case 1: Anomaly low, no detection -> PASS
    dec = triage(anomaly_score=0.15, detection=None, thresholds=thresholds, config=cfg)
    assert dec.verdict == "PASS"

    # Case 2: Anomaly above fail, but no detector confirmation -> REVIEW
    dec = triage(anomaly_score=0.75, detection=None, thresholds=thresholds, config=cfg)
    assert dec.verdict == "REVIEW"
    assert "no confirmed class" in dec.reasons[0]

    # Case 3: Anomaly above fail AND detector confirms scratches above floor -> FAIL
    det = {"class": "scratches", "confidence": 0.85, "bbox": [0, 0, 10, 10]}
    dec = triage(anomaly_score=0.75, detection=det, thresholds=thresholds, config=cfg)
    assert dec.verdict == "FAIL"

    # Case 4: QC gate failed -> early REVIEW
    dec = triage(anomaly_score=0.85, detection=det, thresholds=thresholds, config=cfg, qc_ok=False, qc_flags=["blur_low"])
    assert dec.verdict == "REVIEW"


def test_severity_policy():
    policy = SeverityPolicy(
        version=1,
        mm_per_px=0.5,
        area_thresholds={"minor": 5.0, "major": 20.0, "critical": 50.0},
        class_risk={"scratches": 1.0, "inclusion": 0.5},
    )
    # 8.0 px * 0.5^2 = 2.0 mm2 -> minor
    out = policy.severity(area_px=8.0, defect_class="scratches", zone="surface", score=0.6, fail_threshold=0.5)
    assert out.severity == "minor"

    # 120.0 px * 0.5^2 = 30.0 mm2 -> major
    out = policy.severity(area_px=120.0, defect_class="scratches", zone="surface", score=0.6, fail_threshold=0.5)
    assert out.severity == "major"
