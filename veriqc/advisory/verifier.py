"""Deterministic claim verifier for on-device advisory outputs.

Verifies language model generation against the typed DefectRecord and retrieved SOP cards:
1. Action IDs must belong strictly to the retrieved SOP card set.
2. Numeric values (area, confidence, scores) must match the DefectRecord within tolerance.
3. Defect class, zone, and severity terminology must not contradict the record.
4. Causal assertions must carry explicit hedging words ('probable', 'likely', 'suspect').
5. Audit log lines must remain consistent with the final verdict and part identifier.

Returns a structured VerificationResult containing specific violation flags.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from veriqc.advisory.cards import SopCard
from veriqc.contract.record import DefectRecord

__all__ = ["VerificationResult", "parse_slots", "verify_advisory"]

#: words that make a cause statement acceptable
HEDGES = (
    "probable",
    "possibly",
    "possible",
    "likely",
    "may",
    "might",
    "suspect",
    "suspected",
    "suggests",
    "suggested",
    "could",
    "appears",
    "check",
)

#: words that indicate the text is making a causal claim
CAUSE_MARKERS = (
    "cause",
    "caused by",
    "because",
    "due to",
    "result of",
    "root cause",
)

#: severity words and their canonical form
SEVERITY_WORDS = {
    "minor": "minor",
    "major": "major",
    "critical": "critical",
    "severe": "critical",
    "minority": "minor",
}

_ACTION_RE = re.compile(r"\bA\d{2}\b")
#: A number that is not part of an identifier or float prefix.
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.])(\d+(?:\.\d+)?)(?![A-Za-z0-9_.])")
_FENCE_RE = re.compile(r"```.*?```", re.S)
_SPECIAL_RE = re.compile(r"<\|[^|]*\|>")
_ID_RE = re.compile(r"\b[A-Z]{1,3}-?\d{2,6}\b")
_UNIT_RE = re.compile(r"(?<=\d)(?:mm2|mm\^2|px|%)\b", re.I)
_CONF_RE = re.compile(r"\bconf(?=\d)", re.I)


def _strip_identifiers(text: str) -> str:
    """Remove card ids, part ids, unit suffixes and special tokens before number extraction.

    "A06" must not be read as the number 6, and "P-0412" is an identifier,
    not a measurement.  Both are checked separately (card ids against the
    retrieved set, part id against the record).
    """
    text = _SPECIAL_RE.sub(" ", text)
    text = _ACTION_RE.sub(" ", text)
    text = _ID_RE.sub(" ", text)
    text = _CONF_RE.sub(" ", text)
    text = _UNIT_RE.sub(" ", text)
    return text


@dataclass
class VerificationResult:
    ok: bool
    violations: list[str] = field(default_factory=list)
    action_ids: list[str] = field(default_factory=list)
    numbers: list[float] = field(default_factory=list)
    checks_run: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "violations": list(self.violations),
            "action_ids": list(self.action_ids),
            "numbers": list(self.numbers),
            "checks_run": list(self.checks_run),
        }


def parse_slots(text: str) -> dict[str, str]:
    """Parse the fixed slot output.

    Expected shape::

        ACTIONS: A07, A12
        EXPLAIN: ...
        LOG: ...

    Small models sometimes drop the ``ACTIONS:`` label and write the ids on
    their own line, so a leading line of card ids is accepted as the ACTIONS
    slot.
    """
    text = _FENCE_RE.sub(" ", text)
    text = _SPECIAL_RE.sub(" ", text)
    slots: dict[str, str] = {"ACTIONS": "", "EXPLAIN": "", "LOG": ""}
    current = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = re.match(r"^(ACTIONS|EXPLAIN|LOG)\s*:\s*(.*)$", line, flags=re.I)
        if m:
            current = m.group(1).upper()
            slots[current] = (slots[current] + " " + m.group(2)).strip()
        elif current:
            slots[current] = (slots[current] + " " + line).strip()
        elif not slots["ACTIONS"] and _ACTION_RE.fullmatch(line.replace(",", " ").strip()):
            # bare card-id line: treat it as the ACTIONS slot
            slots["ACTIONS"] = line
    return slots


def _numbers_in(text: str) -> list[float]:
    out = []
    for m in _NUMBER_RE.finditer(text):
        try:
            out.append(float(m.group(1)))
        except ValueError:
            continue
    return out


def _matches_record_number(value: float, record_numbers: list[float], tol_rel: float = 0.02, tol_abs: float = 0.05) -> bool:
    for rn in record_numbers:
        if abs(value - rn) <= tol_abs:
            return True
        if rn != 0 and abs(value - rn) / abs(rn) <= tol_rel:
            return True
    return False


def verify_advisory(
    text: str,
    record: DefectRecord,
    cards: list[SopCard],
    tolerance_rel: float = 0.02,
    tolerance_abs: float = 0.05,
) -> VerificationResult:
    """Check an advisor output against the Defect Record and the retrieved cards."""
    res = VerificationResult(ok=True)
    slots = parse_slots(text)

    if not slots["ACTIONS"] and not slots["EXPLAIN"]:
        res.ok = False
        res.violations.append("no ACTIONS or EXPLAIN slot found")
        return res

    # -- 1. action IDs must come from the retrieved cards --------------------
    allowed_ids = {c.id for c in cards}
    used_ids = _ACTION_RE.findall(slots["ACTIONS"])
    res.action_ids = used_ids
    res.checks_run.append("action_ids")
    for cid in used_ids:
        if cid not in allowed_ids:
            res.ok = False
            res.violations.append(f"action id {cid} is not in the retrieved card set {sorted(allowed_ids)}")
    if not used_ids and record.verdict == "FAIL":
        res.ok = False
        res.violations.append("FAIL advisory contains no action id")

    # -- 2. every number must come from the record --------------------------
    res.checks_run.append("numbers")
    record_numbers = record.numbers()
    for slot_name in ("ACTIONS", "EXPLAIN", "LOG"):
        cleaned = _strip_identifiers(slots[slot_name])
        for value in _numbers_in(cleaned):
            res.numbers.append(value)
            if not _matches_record_number(value, record_numbers, tolerance_rel, tolerance_abs):
                res.ok = False
                res.violations.append(
                    f"number {value} in {slot_name} does not appear in the Defect Record"
                )

    # -- 3. class words -----------------------------------------------------
    res.checks_run.append("class")
    text_lower = (slots["EXPLAIN"] + " " + slots["LOG"]).lower()
    known_classes = {
        "crazing",
        "inclusion",
        "patches",
        "pitted surface",
        "pitted_surface",
        "rolled in scale",
        "rolled-in_scale",
        "scratches",
        "scratch",
    }
    record_words = set(record.class_words())
    for cls in known_classes:
        mentioned = cls in text_lower or cls.replace("_", " ") in text_lower
        if not mentioned:
            continue
        normalised = cls.replace("_", " ")
        if record.defect_class is None:
            res.ok = False
            res.violations.append(f"advisory names defect class '{normalised}' but the record has none")
        elif normalised != record.defect_class.replace("_", " ") and not (
            normalised in record.defect_class.replace("_", " ")
        ):
            res.ok = False
            res.violations.append(
                f"advisory names defect class '{normalised}' but the record says '{record.defect_class}'"
            )

    # -- 4. severity words --------------------------------------------------
    res.checks_run.append("severity")
    if record.severity:
        for word, canonical in SEVERITY_WORDS.items():
            if re.search(rf"\b{word}\b", text_lower) and canonical != record.severity:
                res.ok = False
                res.violations.append(
                    f"advisory calls the defect '{word}' but the record severity is '{record.severity}'"
                )
    elif any(re.search(rf"\b{w}\b", text_lower) for w in SEVERITY_WORDS):
        res.ok = False
        res.violations.append("advisory states a severity but the record has none")

    # -- 5. zone words ------------------------------------------------------
    res.checks_run.append("zone")
    zones = ("edge", "center", "centre", "corner", "surface", "interior")
    for z in zones:
        if re.search(rf"\b{z}\b", text_lower):
            if record.zone is None:
                res.ok = False
                res.violations.append(f"advisory names zone '{z}' but the record has none")
            elif z not in record.zone.lower() and record.zone.lower() not in z:
                res.ok = False
                res.violations.append(f"advisory names zone '{z}' but the record says '{record.zone}'")

    # -- 6. causal claims must be hedged ------------------------------------
    res.checks_run.append("hedging")
    lowered = text_lower
    if any(marker in lowered for marker in CAUSE_MARKERS):
        if not any(h in lowered for h in HEDGES):
            res.ok = False
            res.violations.append("causal claim is not hedged")
    # a card's cause phrase used verbatim must also be hedged
    for card in cards:
        for cause in card.probable_causes:
            key = cause.split()[0].lower() if cause else ""
            if key and key in lowered and not any(h in lowered for h in HEDGES):
                res.ok = False
                res.violations.append(f"cause '{cause[:40]}...' used without hedging")

    # -- 7. the log line must stay consistent with the verdict --------------
    res.checks_run.append("verdict")
    if record.verdict == "FAIL" and re.search(r"\bPASS\b", slots["LOG"]):
        res.ok = False
        res.violations.append("log line says PASS for a FAIL record")
    if record.verdict == "PASS" and slots["ACTIONS"]:
        res.ok = False
        res.violations.append("PASS record carries action ids")

    # -- 8. the audit line must carry the record's own part id --------------
    res.checks_run.append("part_id")
    if record.part_id and slots["LOG"]:
        other_ids = [m for m in _ID_RE.findall(slots["LOG"]) if m.upper() != record.part_id.upper()]
        if other_ids:
            res.ok = False
            res.violations.append(
                f"log line names {other_ids} but the record is {record.part_id}"
            )

    return res
