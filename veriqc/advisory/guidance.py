"""Findings-only advisory prompt, parser and deterministic verifier."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

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
    "You are an industrial quality-control assistant. Use only the structured findings "
    "and the numbered options. Do not invent measurements, defect classes or actions. "
    "Return exactly three lines: CAUSE: <number>, ACTION: <number>, NOTE: <one short "
    "sentence without digits>. Do not return a numbered list; the first word must be CAUSE."
)
_ANSWER = """CAUSE: 1
ACTION: 2
NOTE: Check the fixture before the next part."""
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


def _finding_text(result: InspectionResult) -> str:
    lines = [f"profile: {result.profile_id}", f"verdict: {result.verdict}"]
    for finding in result.findings:
        parts = [f"class={finding.defect_class}", f"source={finding.source}"]
        if finding.confidence is not None:
            parts.append(f"confidence={finding.confidence:.2f}")
        if finding.score is not None:
            parts.append(f"score={finding.score:.3f}")
        if finding.area_ratio is not None:
            parts.append(f"area_ratio={finding.area_ratio:.6f}")
        if finding.zone:
            parts.append(f"zone={finding.zone}")
        if finding.severity:
            parts.append(f"severity={finding.severity}")
        lines.append("finding: " + "; ".join(parts))
    return "\n".join(lines)


def build_prompt(result: InspectionResult, card: Card, tokenizer=None) -> str:
    causes = "\n".join(f"{index}. {text}" for index, text in enumerate(card.probable_causes, 1))
    actions = "\n".join(f"{index}. {text}" for index, text in enumerate(card.immediate_actions, 1))
    user = (
        f"Findings:\n{_finding_text(result)}\n\n"
        f"CAUSE options:\n{causes}\n\n"
        f"ACTION options:\n{actions}\n\n"
        f"Example answer:\n{_ANSWER}\n\nWrite the three lines now. Start with CAUSE:."
    )
    if tokenizer is None:
        return f"{_SYSTEM}\n\n{user}"
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


def parse_selection(text: str, card: Card) -> Selection | None:
    fields: dict[str, str] = {}
    for raw in text.splitlines():
        match = re.match(r"^\s*(CAUSE|ACTION|NOTE)\s*[:=]\s*(.*?)\s*$", raw, flags=re.I)
        if not match:
            continue
        key = match.group(1).upper()
        # The 1B model sometimes emits several candidate blocks.  The first
        # complete block is the one the verifier evaluates; later blocks are
        # ignored rather than used to reject an otherwise valid answer.
        if key not in fields:
            fields[key] = match.group(2)
        if set(fields) == {"CAUSE", "ACTION", "NOTE"}:
            break
    if set(fields) != {"CAUSE", "ACTION", "NOTE"}:
        return None
    cause_match = re.search(r"\b(\d+)\b", fields["CAUSE"])
    action_match = re.search(r"\b(\d+)\b", fields["ACTION"])
    if not cause_match or not action_match:
        return None
    cause_index = int(cause_match.group(1))
    action_index = int(action_match.group(1))
    if not (1 <= cause_index <= len(card.probable_causes)):
        return None
    if not (1 <= action_index <= len(card.immediate_actions)):
        return None
    note = fields["NOTE"].strip().strip("\"'")
    if not note or len(note.split()) > 40:
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
    if "unknown_anomaly" in unsupported and "unknown_anomaly" in known_classes:
        unsupported.discard("unknown_anomaly")
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
    primary = result.findings[0].defect_class.replace("_", " ") if result.findings else "finding"
    escalation = f" Escalate if: {card.escalation}." if card.escalation else ""
    return (
        f"{primary.capitalize()}: {card.summary}. "
        f"Probable cause: {cause}. Do now: {action}. {note}{escalation}"
    )
