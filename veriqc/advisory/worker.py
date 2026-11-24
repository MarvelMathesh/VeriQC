"""Generate, verify and publish one advisory.  The template is the floor."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from veriqc.advisory.cards import CardLibrary
from veriqc.advisory.constants import ADVISORY_DEADLINE_S, ADVISORY_MAX_NEW_TOKENS
from veriqc.advisory.guidance import (
    Selection,
    build_prompt,
    compose,
    parse_selection,
    primary_finding,
    verify_selection,
)
from veriqc.advisory.process_runner import AdvisoryTimeout
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
    timed_out: bool = False
    error: str | None = None
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
            "timed_out": self.timed_out,
            "error": self.error,
            "verification": dict(self.verification),
            "model": dict(self.model),
        }


class AdvisoryWorker:
    def __init__(
        self,
        cards: CardLibrary,
        runner=None,
        *,
        max_new_tokens: int = ADVISORY_MAX_NEW_TOKENS,
        deadline_s: float = ADVISORY_DEADLINE_S,
        context_provider=None,
    ) -> None:
        self.cards = cards
        self.runner = runner
        self.max_new_tokens = int(max_new_tokens)
        self.deadline_s = float(deadline_s)
        self.context_provider = context_provider

    @property
    def available(self) -> bool:
        return self.runner is not None

    def template_for(self, result: InspectionResult) -> tuple[str, str | None, str | None, str | None]:
        finding = primary_finding(result)
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
        finding = primary_finding(result)
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
            context = self.context_provider() if self.context_provider is not None else None
            prompt = build_prompt(
                result,
                card,
                tokenizer=getattr(self.runner, "tokenizer", None),
                context=context,
            )
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

        generated, generation_error = self._generate(prompt)
        if generation_error is not None:
            timed_out = generation_error.startswith("timeout:")
            return AdvisoryOutcome(
                text=template,
                template=template,
                path="template",
                accepted=False,
                cause=default_cause,
                action=default_action,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                timed_out=timed_out,
                error=generation_error,
                verification={
                    "ok": False,
                    "violations": [generation_error],
                },
            )
        if generated is None or not generated.ok:
            error = (
                generated.error
                if generated is not None and generated.error
                else "advisory runtime returned no result"
            )
            return AdvisoryOutcome(
                text=template,
                template=template,
                path="template",
                accepted=False,
                cause=default_cause,
                action=default_action,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                error=error,
                verification={"ok": False, "violations": [f"runtime error: {error}"]},
                model=generated.to_dict() if generated is not None else {},
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
                    "violations": ["the model did not return a valid selection"],
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
        if not selection.note:
            selection = Selection(
                selection.cause_index,
                selection.action_index,
                self._default_note(result),
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

    @staticmethod
    def _default_note(result: InspectionResult) -> str:
        finding = primary_finding(result)
        if finding is None:
            return "check the process before the next part"
        if finding.severity == "critical":
            return "isolate the part and inspect the fixture"
        if finding.zone in ("edge", "corner"):
            return "check the fixture alignment before the next part"
        return "check the process before the next part"

    def _restart(self) -> None:
        restart = getattr(self.runner, "restart", None)
        if restart is not None:
            try:
                restart()
            except Exception:
                pass

    def _generate(self, prompt: str) -> tuple[Any | None, str | None]:
        """Return ``(result, error)`` and never conflate an error with a timeout."""

        try:
            result = self.runner.generate(
                prompt,
                max_new_tokens=self.max_new_tokens,
                timeout_s=self.deadline_s,
            )
        except AdvisoryTimeout as exc:
            self._restart()
            return None, f"timeout: {exc}"
        except Exception as exc:
            self._restart()
            return None, f"runtime error: {exc!r}"
        if result is None:
            return None, "runtime error: advisory runtime returned no result"
        if not result.ok:
            self._restart()
        return result, None
