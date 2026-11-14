"""Unit tests for the deterministic advisory verifier."""

from __future__ import annotations

import pytest

from veriqc.advisory.cards import SopCard
from veriqc.advisory.verifier import parse_slots, verify_advisory
from veriqc.contract.record import DefectRecord


def test_parse_slots():
    text = (
        "ACTIONS: A01, A02\n"
        "EXPLAIN: Scratches observed on the surface, 12.5 mm2. Likely due to roller contact.\n"
        "LOG: P-100 FAIL major scratches 12.5mm2"
    )
    parsed = parse_slots(text)
    assert parsed["ACTIONS"] == "A01, A02"
    assert "12.5 mm2" in parsed["EXPLAIN"]
    assert parsed["LOG"].startswith("P-100")


def test_verify_advisory_valid_and_hallucination():
    record = DefectRecord(
        part_id="P-100",
        verdict="FAIL",
        defect_class="scratches",
        class_confidence=0.91,
        severity="major",
        mask_area_mm2=12.5,
        zone="surface",
    )
    card_a01 = SopCard(
        id="A01",
        defect_class="scratches",
        summary="Surface Scratch SOP",
        immediate_actions=["Halt line"],
        probable_causes=["handling contact"],
        escalation="Notify supervisor",
    )
    cards = [card_a01]

    # 1. Valid output
    valid_text = (
        "ACTIONS: A01\n"
        "EXPLAIN: A major scratches defect of 12.5 mm2 detected on the surface. Probable cause per card A01 is handling contact; halt line.\n"
        "LOG: P-100 FAIL major scratches surface 12.5mm2"
    )
    res = verify_advisory(valid_text, record, cards)
    assert res.ok is True
    assert len(res.violations) == 0

    # 2. Hallucination Trap 1: Invalid Action ID
    bad_action_text = (
        "ACTIONS: A99\n"
        "EXPLAIN: A major scratches defect of 12.5 mm2 detected on the surface. Likely handling contact.\n"
        "LOG: P-100 FAIL major scratches 12.5mm2"
    )
    res_bad_action = verify_advisory(bad_action_text, record, cards)
    assert res_bad_action.ok is False
    assert any("A99" in v for v in res_bad_action.violations)

    # 3. Hallucination Trap 2: Invented number not in record
    invented_num_text = (
        "ACTIONS: A01\n"
        "EXPLAIN: A major scratches defect of 99.9 mm2 detected on the surface. Likely handling contact.\n"
        "LOG: P-100 FAIL major scratches 99.9mm2"
    )
    res_bad_num = verify_advisory(invented_num_text, record, cards)
    assert res_bad_num.ok is False
    assert any("99.9" in v for v in res_bad_num.violations)

    # 4. Hallucination Trap 3: Unhedged causal assertion
    unhedged_text = (
        "ACTIONS: A01\n"
        "EXPLAIN: A major scratches defect of 12.5 mm2 detected on the surface. The root cause is broken guide rails.\n"
        "LOG: P-100 FAIL major scratches surface 12.5mm2"
    )
    res_unhedged = verify_advisory(unhedged_text, record, cards)
    assert res_unhedged.ok is False
    assert any("hedged" in v for v in res_unhedged.violations)
