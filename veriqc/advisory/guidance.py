"""Findings-only advisory prompt, parser and deterministic verifier."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from veriqc.advisory.cards import Card
from veriqc.inspection import InspectionResult

__all__ = [
    "Selection",
    "Verification",
    "build_prompt",
    "parse_selection",
    "verify_selection",
    "compose",
]

_SYSTEM = (
    "QC assistant. Use only the findings and numbered options. Reply exactly "
    "CAUSE:<n>|ACTION:<n>|NOTE:<max ten words, no digits>. "
    "Do not invent classes or measurements."
)
_ANSWER = "CAUSE:1|ACTION:2|NOTE:check fixture before next part"
_CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n"
    "{{ message['content'] }}<|eot_id|>"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    "{% endif %}"
)
_DIGIT_RE = re.compile(r"\d")
_CLASS_WORDS = {
    "crack": "crack", "cracks": "crack", "cracked": "crack",
    "scratch": "scratches", "scratches": "scratches",
    "crazing": "crazing", "inclusion": "inclusion", "inclusions": "inclusion",
    "patch": "patches", "patches": "patches", "pitted": "pitted_surface",
    "pits": "pitted_surface", "scale": "rolled-in_scale",
    "bubble": "unknown_anomaly", "dent": "unknown_anomaly", "hole": "unknown_anomaly",
    "contamination": "unknown_anomaly", "stain": "unknown_anomaly",
}
_TERM_RE = re.compile(
    r"(?<![\w-])(" + "|".join(re.escape(k) for k in sorted(_CLASS_WORDS, key=len, reverse=True)) + r")(?![\w-])",
    re.I,
)
_IDENTIFIER_RE = re.compile(r"\b[A-Za-z]{1,8}[_-]\d{1,8}\b")


@dataclass(frozen=True)
class Selection:
    cause_index: int
    action_index: int
    note: str


@dataclass
class Verification:
    ok: bool
    violations: list[str] = field(default_factory=list)


def primary_finding(result: InspectionResult):
    """The finding the operator should act on first."""

    if not result.findings:
        return None
    rank = {"critical": 3, "major": 2, "minor": 1}
    return max(
        result.findings,
        key=lambda item: (
            rank.get(item.severity or "", 0),
            float(item.area_px or 0.0),
            float(item.confidence or 0.0),
        ),
    )


def _finding_text(result: InspectionResult) -> str:
    lines = [f"profile={result.profile_id}", f"verdict={result.verdict}"]
    for finding in result.findings[:4]:
        parts = [f"class={finding.defect_class}", f"source={finding.source}"]
        if finding.severity:
            parts.append(f"severity={finding.severity}")
        if finding.zone:
            parts.append(f"zone={finding.zone}")
        if finding.area_ratio is not None:
            parts.append(f"area={finding.area_ratio:.6f}")
        if finding.confidence is not None:
            parts.append(f"confidence={finding.confidence:.2f}")
        if finding.region_p_value is not None:
            parts.append(f"region_p={finding.region_p_value:.3f}")
        lines.append(";".join(parts))
    return " | ".join(lines)


def _short(text: str, max_words: int = 6) -> str:
    return " ".join(str(text).split()[:max_words])


def build_prompt(
    result: InspectionResult,
    card: Card,
    tokenizer=None,
    context: dict[str, Any] | None = None,
) -> str:
    causes = " ".join(
        f"{index}.{_short(text, 6)}" for index, text in enumerate(card.probable_causes, 1)
    )
    actions = " ".join(
        f"{index}.{_short(text, 6)}" for index, text in enumerate(card.immediate_actions, 1)
    )
    context_line = ""
    if context:
        verdicts = context.get("verdicts") or {}
        defects = context.get("defects") or {}
        if verdicts or defects:
            context_line = (
                f"\nRecent {context.get('window', 50)}: "
                f"PASS{verdicts.get('PASS', 0)} REVIEW{verdicts.get('REVIEW', 0)} "
                f"FAIL{verdicts.get('FAIL', 0)} "
                + " ".join(f"{name}{count}" for name, count in list(defects.items())[:3])
            )
    user = (
        f"Findings {_finding_text(result)}{context_line}\n"
        f"CAUSE options: {causes}\n"
        f"ACTION options: {actions}\n"
        f"Example: {_ANSWER}\n"
        f"Write exactly one line now, starting with CAUSE:"
    )
    if tokenizer is None:
        return f"{_SYSTEM}\n\n{user}"
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
        chat_template=_CHAT_TEMPLATE,
    )


def _resolve_option(value: str, options: Sequence[str]) -> int | None:
    match = re.search(r"\b(\d+)\b", value)
    if match:
        index = int(match.group(1))
        if 1 <= index <= len(options):
            return index
    normalized = " ".join(value.lower().split())
    best_index = None
    best_ratio = 0.0
    for index, option in enumerate(options, 1):
        ratio = difflib.SequenceMatcher(
            None, normalized, " ".join(option.lower().split())
        ).ratio()
        if ratio > best_ratio:
            best_index, best_ratio = index, ratio
    return best_index if best_ratio >= 0.45 else None


def parse_selection(text: str, card: Card) -> Selection | None:
    fields: dict[str, str] = {}
    for raw in re.split(r"[|\n]+", text):
        match = re.match(r"^\s*(CAUSE|ACTION|NOTE)\s*[:=]\s*(.*?)\s*$", raw, flags=re.I)
        if not match:
            continue
        key = match.group(1).upper()
        # The 1B model sometimes emits several candidate blocks.  The first
        # complete block is the one the verifier evaluates; later blocks are
        # ignored rather than used to reject an otherwise valid answer.
        if key not in fields:
            fields[key] = match.group(2)
        if {"CAUSE", "ACTION", "NOTE"} <= set(fields):
            break
    if not {"CAUSE", "ACTION"} <= set(fields):
        return None
    cause_index = _resolve_option(fields["CAUSE"], card.probable_causes)
    action_index = _resolve_option(fields["ACTION"], card.immediate_actions)
    if cause_index is None or action_index is None:
        return None
    note = fields.get("NOTE", "").strip().strip("\"'")
    if len(note.split()) > 40:
        return None
    return Selection(cause_index, action_index, note)


def verify_selection(selection: Selection, result: InspectionResult) -> Verification:
    violations: list[str] = []
    note = selection.note
    if _DIGIT_RE.search(note):
        violations.append("note contains a numeral")
    if _IDENTIFIER_RE.search(note):
        violations.append("note contains a record identifier")
    known_classes = {finding.defect_class for finding in result.findings}
    mentioned = {_CLASS_WORDS[m.group(1).lower()] for m in _TERM_RE.finditer(note)}
    unsupported = mentioned - known_classes
    if unsupported:
        violations.append(f"note mentions unsupported classes: {sorted(unsupported)}")
    if re.search(r"\b(?:definitely|certainly|confirmed)\b", note, re.I):
        violations.append("note overstates certainty")
    if re.search(r"\b(?:ignore|skip|bypass)\b", note, re.I):
        violations.append("note suggests bypassing inspection")
    return Verification(ok=not violations, violations=violations)


def compose(selection: Selection, result: InspectionResult, card: Card) -> str:
    cause = card.probable_causes[selection.cause_index - 1]
    action = card.immediate_actions[selection.action_index - 1]
    note = selection.note.strip()
    if not note.endswith((".", "!", "?")):
        note += "."
    finding = primary_finding(result)
    primary = finding.defect_class.replace("_", " ") if finding is not None else "finding"
    escalation = f" Escalate if: {card.escalation}." if card.escalation else ""
    return (
        f"{primary.capitalize()}: {card.summary}. "
        f"Probable cause: {cause}. Do now: {action}. {note}{escalation}"
    )
