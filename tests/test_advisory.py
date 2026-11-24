from __future__ import annotations

from pathlib import Path

from veriqc.advisory.cards import load_cards
from veriqc.advisory.executorch import AdvisorResult
from veriqc.advisory.worker import AdvisoryWorker
from veriqc.config import load_profile
from veriqc.inspection import Finding, InspectionResult

REPO = Path(__file__).resolve().parents[1]


def _result() -> InspectionResult:
    return InspectionResult(
        inspection_id="i",
        timestamp=1.0,
        part_id="P",
        profile_id="commutator",
        verdict="FAIL",
        findings=[
            Finding(
                finding_id="known-0",
                source="known",
                defect_class="crack",
                confidence=0.9,
                score=0.9,
                bbox=(1, 1, 10, 10),
                area_px=81,
                area_ratio=0.001,
                zone="edge",
                severity="major",
            )
        ],
    )


class FakeRunner:
    tokenizer = None

    def __init__(self, text: str):
        self.text = text

    def generate(self, prompt, max_new_tokens=None, timeout_s=None):
        return AdvisorResult(text=self.text, ok=True, path="llm")


def test_worker_accepts_compact_pipe_format() -> None:
    profile = load_profile(REPO / "configs/profiles/commutator.yaml")
    worker = AdvisoryWorker(load_cards(profile.cards_path), FakeRunner("CAUSE: 1 | ACTION: 1 | NOTE: check the fixture first."))
    outcome = worker.advise_result(_result())
    assert outcome.accepted is True
    assert "quarantine the part" in outcome.text or "handling" in outcome.text


def test_worker_falls_back_when_generation_raises() -> None:
    class Broken:
        tokenizer = None

        def restart(self):
            self.restarted = True

        def generate(self, prompt, max_new_tokens=None, timeout_s=None):
            raise RuntimeError("timeout")

    profile = load_profile(REPO / "configs/profiles/commutator.yaml")
    runner = Broken()
    worker = AdvisoryWorker(load_cards(profile.cards_path), runner)
    outcome = worker.advise_result(_result())
    assert outcome.accepted is False
    assert outcome.path == "template"
    assert outcome.timed_out is True
    assert getattr(runner, "restarted", False) is True
