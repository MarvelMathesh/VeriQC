from __future__ import annotations

from veriqc.advisory.cards import Card
from veriqc.advisory.guidance import build_prompt, compose, parse_selection, verify_selection
from veriqc.inspection import Finding, InspectionResult


def card() -> Card:
    return Card(
        id="c",
        defect_class="crack",
        summary="a break in the surface",
        probable_causes=("handling damage", "stress at a step"),
        immediate_actions=("quarantine the part", "compare with the batch"),
        escalation="if it repeats, hold the batch",
        profile_id="p",
    )


def result() -> InspectionResult:
    return InspectionResult(
        inspection_id="i",
        timestamp=1.0,
        part_id="P1",
        profile_id="p",
        verdict="FAIL",
        findings=[
            Finding(
                finding_id="known-0",
                source="known",
                defect_class="crack",
                confidence=0.8,
                score=0.8,
                bbox=(1, 2, 10, 20),
                area_px=100,
                area_ratio=0.002,
                zone="edge",
                severity="major",
                severity_status="policy",
            )
        ],
    )


def test_prompt_contains_findings_and_options() -> None:
    prompt = build_prompt(result(), card())
    assert "class=crack" in prompt
    assert "CAUSE options" in prompt
    assert "ACTION options" in prompt


def test_natural_model_format_is_accepted() -> None:
    selection = parse_selection("CAUSE: 1 - handling damage\nACTION: 2 - compare with the batch\nNOTE: check the fixture first.", card())
    assert selection is not None
    assert selection.cause_index == 1
    assert selection.action_index == 2


def test_malformed_or_out_of_range_is_rejected() -> None:
    assert parse_selection("CAUSE: 9\nACTION: 1\nNOTE: ok", card()) is None
    assert parse_selection("CAUSE: 1\nACTION: 1", card()) is None
    assert parse_selection("CAUSE: 1\nACTION: 1\nNOTE: ok\n\nCAUSE: 2\nACTION: 2\nNOTE: later", card()) is not None


def test_verifier_rejects_digits_and_unsupported_classes() -> None:
    good = parse_selection("CAUSE: 1\nACTION: 1\nNOTE: check the fixture before the next part.", card())
    assert good is not None
    assert verify_selection(good, result()).ok
    digits = parse_selection("CAUSE: 1\nACTION: 1\nNOTE: check 3 parts.", card())
    assert digits is not None
    assert not verify_selection(digits, result()).ok
    other = parse_selection("CAUSE: 1\nACTION: 1\nNOTE: the scratch is on the edge.", card())
    assert other is not None
    assert not verify_selection(other, result()).ok


def test_compose_is_code_owned() -> None:
    selection = parse_selection("CAUSE: 1\nACTION: 1\nNOTE: check the fixture first.", card())
    assert selection is not None
    text = compose(selection, result(), card())
    assert "handling damage" in text
    assert "quarantine the part" in text
    assert "check the fixture first" in text
