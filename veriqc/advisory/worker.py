"""Generate, verify and publish one advisory.  The template is the floor."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from veriqc.advisory.cards import CardLibrary
from veriqc.advisory.guidance import (
    build_prompt,
    compose,
    parse_selection,
    verify_selection,
)
from veriqc.inspection import InspectionResult

__all__ = ["AdvisoryOutcome", "AdvisoryWorker"]


@dataclass
class AdvisoryOutcome:
    text: str
    template: str
    path: str
    accepted: bool
    latency_ms: float = 0.0
    cause: str | None = None
    action: str | None = None
    note: str | None = None
    verification: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "accepted" if self.accepted else "template",
            "text": self.text,
            "template": self.template,
            "path": self.path,
            "accepted": self.accepted,
            "latency_ms": round(self.latency_ms, 1),
            "cause": self.cause,
            "action": self.action,
            "note": self.note,
            "verification": dict(self.verification),
            "model": dict(self.model),
        }


class AdvisoryWorker:
    def __init__(
        self,
        cards: CardLibrary,
        runner=None,
        *,
        max_new_tokens: int = 48,
        deadline_s: float = 60.0,
    ) -> None:
        self.cards = cards
        self.runner = runner
        self.max_new_tokens = int(max_new_tokens)
        self.deadline_s = float(deadline_s)
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self.runner is not None

    def template_for(self, result: InspectionResult) -> tuple[str, str | None, str | None, str | None]:
        finding = result.findings[0] if result.findings else None
        defect_class = finding.defect_class if finding else "generic"
        severity = finding.severity if finding else None
        card = self.cards.primary(defect_class, severity)
        if card is None:
            return "Inspection requires review; see the structured findings.", None, None, None
        return (
            self.cards.template(card, defect_class),
            card.id,
            card.probable_causes[0] if card.probable_causes else None,
            card.immediate_actions[0] if card.immediate_actions else None,
        )

    def advise_result(self, result: InspectionResult) -> AdvisoryOutcome:
        started = time.perf_counter()
        template, card_id, default_cause, default_action = self.template_for(result)
        if result.verdict == "PASS":
            return AdvisoryOutcome(
                text="Part accepted. Continue the line.",
                template="Part accepted. Continue the line.",
                path="pass",
                accepted=True,
            )
        if self.runner is None:
            return AdvisoryOutcome(
                text=template,
                template=template,
                path="template",
                accepted=False,
                cause=default_cause,
                action=default_action,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                verification={"ok": False, "violations": ["advisory runtime unavailable"]},
            )
        finding = result.findings[0] if result.findings else None
        card = self.cards.primary(finding.defect_class if finding else "generic", finding.severity if finding else None)
        if card is None or not card.probable_causes or not card.immediate_actions:
            return AdvisoryOutcome(
                text=template,
                template=template,
                path="template",
                accepted=False,
                cause=default_cause,
                action=default_action,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                verification={"ok": False, "violations": ["no card options"]},
            )
        try:
            prompt = build_prompt(result, card, tokenizer=getattr(self.runner, "tokenizer", None))
        except Exception as exc:
            return AdvisoryOutcome(
                text=template,
                template=template,
                path="template",
                accepted=False,
                cause=default_cause,
                action=default_action,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                verification={"ok": False, "violations": [f"prompt error: {exc!r}"]},
            )

        generated = self._generate(prompt)
        if generated is None:
            return AdvisoryOutcome(
                text=template,
                template=template,
                path="template",
                accepted=False,
                cause=default_cause,
                action=default_action,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                verification={"ok": False, "violations": ["generation timeout or busy"]},
            )
        selection = parse_selection(generated.text, card)
        if selection is None:
            return AdvisoryOutcome(
                text=template,
                template=template,
                path="template",
                accepted=False,
                cause=default_cause,
                action=default_action,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                verification={
                    "ok": False,
                    "violations": ["the model did not return a valid three-line selection"],
                    "raw": generated.text[:500],
                },
                model=generated.to_dict(),
            )
        verification = verify_selection(selection, result)
        if not verification.ok:
            return AdvisoryOutcome(
                text=template,
                template=template,
                path="template",
                accepted=False,
                cause=default_cause,
                action=default_action,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                verification={"ok": False, "violations": verification.violations},
                model=generated.to_dict(),
            )
        text = compose(selection, result, card)
        return AdvisoryOutcome(
            text=text,
            template=template,
            path="llm",
            accepted=True,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            cause=card.probable_causes[selection.cause_index - 1],
            action=card.immediate_actions[selection.action_index - 1],
            note=selection.note,
            verification={"ok": True, "violations": [], "card_id": card_id},
            model=generated.to_dict(),
        )

    def _generate(self, prompt: str):
        holder: dict[str, Any] = {}

        def run() -> None:
            try:
                holder["result"] = self.runner.generate(prompt, max_new_tokens=self.max_new_tokens)
            except Exception as exc:
                holder["error"] = exc
            finally:
                self._lock.release()

        if not self._lock.acquire(blocking=False):
            return None
        thread = threading.Thread(target=run, daemon=True, name="VeriQC-Advisory")
        thread.start()
        thread.join(self.deadline_s)
        if thread.is_alive():
            return None
        if "error" in holder:
            return None
        return holder.get("result")
