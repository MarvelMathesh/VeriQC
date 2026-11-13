"""The advisory worker: generate, verify, retry once, fall back to a template.

Failure recovery lifecycle:
1. Slot-constrained LLM generation attempt.
2. Deterministic verification against SOP cards and DefectRecord numbers.
3. On verification failure: retry once at temperature 0 with a stricter correction prompt.
4. On secondary failure: fall back to a deterministic template card summary.
Every advisory outcome is recorded with its audit path label and verification reasons.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from veriqc.advisory.cards import CardLibrary, SopCard
from veriqc.advisory.prompts import build_prompt, build_template
from veriqc.advisory.verifier import VerificationResult, verify_advisory
from veriqc.contract.record import DefectRecord

__all__ = ["AdvisoryOutcome", "AdvisoryWorker"]


@dataclass
class AdvisoryOutcome:
    """The advisory plus its provenance and verification evidence."""

    record: DefectRecord
    text: str
    path: str  # llm | retry | template
    verified: bool
    violations: list[str] = field(default_factory=list)
    checks_run: list[str] = field(default_factory=list)
    action_ids: list[str] = field(default_factory=list)
    ttft_ms: float = 0.0
    decode_ms: float = 0.0
    tokens_per_second: float = 0.0
    prompt_tokens: int = 0
    generated_tokens: int = 0
    wall_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "verified": self.verified,
            "violations": list(self.violations),
            "checks_run": list(self.checks_run),
            "action_ids": list(self.action_ids),
            "ttft_ms": round(self.ttft_ms, 1),
            "decode_ms": round(self.decode_ms, 1),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "wall_ms": round(self.wall_ms, 1),
        }


class AdvisoryWorker:
    """Owns the generate -> verify -> retry -> template policy."""

    def __init__(
        self,
        card_library: CardLibrary,
        runner=None,
        max_cards: int = 3,
        max_new_tokens: int = 96,
        temperature: float = 0.0,
    ):
        self.cards = card_library
        self.runner = runner
        self.max_cards = max_cards
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    # -- helpers -----------------------------------------------------------
    def retrieve(self, record: DefectRecord) -> list[SopCard]:
        return self.cards.retrieve(record.defect_class, record.severity, limit=self.max_cards)

    def _tokenizer(self):
        return getattr(self.runner, "tokenizer", None) if self.runner is not None else None

    def _strict_prompt(self, record: DefectRecord, cards: list[SopCard], violations: list[str]) -> str:
        base = build_prompt(record, cards, tokenizer=self._tokenizer())
        extra = (
            "\n\nYour previous answer was rejected because: "
            + "; ".join(violations[:4])
            + ". Answer again, obeying every rule exactly. "
            "Use only the numbers and the class from the record, and hedge any cause."
        )
        return base + extra

    # -- main entry point ---------------------------------------------------
    def advise(self, record: DefectRecord) -> AdvisoryOutcome:
        t0 = time.perf_counter()
        cards = self.retrieve(record)

        if self.runner is None:
            text = build_template(record, cards)
            vr = verify_advisory(text, record, cards)
            return AdvisoryOutcome(
                record=record,
                text=text,
                path="template",
                verified=vr.ok,
                violations=vr.violations,
                checks_run=vr.checks_run,
                action_ids=vr.action_ids,
                wall_ms=(time.perf_counter() - t0) * 1000,
            )

        # ---- first pass ----------------------------------------------------
        prompt = build_prompt(record, cards, tokenizer=self._tokenizer())
        res = self.runner.generate(prompt, max_new_tokens=self.max_new_tokens)
        vr = verify_advisory(res.text, record, cards)
        if vr.ok:
            return AdvisoryOutcome(
                record=record,
                text=res.text,
                path="llm",
                verified=True,
                violations=[],
                checks_run=vr.checks_run,
                action_ids=vr.action_ids,
                ttft_ms=res.ttft_ms,
                decode_ms=res.decode_ms,
                tokens_per_second=res.tokens_per_second,
                prompt_tokens=res.prompt_tokens,
                generated_tokens=res.generated_tokens,
                wall_ms=(time.perf_counter() - t0) * 1000,
            )

        # ---- retry once, stricter prompt, temperature 0 ---------------------
        retry_prompt = self._strict_prompt(record, cards, vr.violations)
        res2 = self.runner.generate(retry_prompt, max_new_tokens=self.max_new_tokens)
        vr2 = verify_advisory(res2.text, record, cards)
        if vr2.ok:
            return AdvisoryOutcome(
                record=record,
                text=res2.text,
                path="retry",
                verified=True,
                violations=[],
                checks_run=vr2.checks_run,
                action_ids=vr2.action_ids,
                ttft_ms=res2.ttft_ms,
                decode_ms=res2.decode_ms,
                tokens_per_second=res2.tokens_per_second,
                prompt_tokens=res2.prompt_tokens,
                generated_tokens=res2.generated_tokens,
                wall_ms=(time.perf_counter() - t0) * 1000,
            )

        # ---- template fallback ----------------------------------------------
        text = build_template(record, cards)
        vr3 = verify_advisory(text, record, cards)
        return AdvisoryOutcome(
            record=record,
            text=text,
            path="template",
            verified=vr3.ok,
            violations=vr2.violations,
            checks_run=vr3.checks_run,
            action_ids=vr3.action_ids,
            ttft_ms=res.ttft_ms,
            decode_ms=res.decode_ms,
            tokens_per_second=res.tokens_per_second,
            prompt_tokens=res.prompt_tokens,
            generated_tokens=res.generated_tokens,
            wall_ms=(time.perf_counter() - t0) * 1000,
        )

    # -- ablation helper ----------------------------------------------------
    def raw_generate(self, record: DefectRecord) -> tuple[str, VerificationResult, list[SopCard]]:
        """Generate without verifying -- used for the A4 ablation."""
        cards = self.retrieve(record)
        if self.runner is None:
            return "", VerificationResult(ok=False, violations=["no runner"]), cards
        prompt = build_prompt(record, cards, tokenizer=self._tokenizer())
        res = self.runner.generate(prompt, max_new_tokens=self.max_new_tokens)
        return res.text, verify_advisory(res.text, record, cards), cards
