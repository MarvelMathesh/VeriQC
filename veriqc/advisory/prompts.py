"""Prompt construction and deterministic template fallback.

Constructs bounded slot-filling prompts containing system rules, one typed
DefectRecord, and up to three retrieved SOP cards. The constrained prompt keeps
token count bounded (~600-900 tokens) for low time-to-first-token (TTFT) on edge CPUs.
Includes deterministic rule-based template generation for verified fallbacks.
"""

from __future__ import annotations

from veriqc.advisory.cards import SopCard
from veriqc.contract.record import DefectRecord

__all__ = ["build_prompt", "build_template", "format_record_for_prompt"]

SYSTEM_RULES = """You are an inspection assistant for a factory quality station.
You are given one Defect Record and a list of allowed SOP cards.
You must fill exactly these slots and nothing else:

ACTIONS: list of up to three card IDs from the cards you were given, separated by commas.
EXPLAIN: two short sentences describing the finding and the action to take.
LOG: one short line for the audit log.

Rules you must follow:
- Only use card IDs that appear in the cards you were given.
- Only use numbers that appear in the Defect Record.
- Never state a cause as fact. Use hedged wording such as "probable" or "may".
- Never change the defect class, the severity or the zone.
- Copy the part_id and every number exactly from the Defect Record.
- Do not add any other text."""

#: A worked example.  A 1B model does not reliably reproduce a slot format
#: from an abstract instruction alone; one concrete example fixes the format
#: and costs about 150 tokens of prompt.
ONE_SHOT = """Example of the required format:

Defect Record:
part_id: P-0001
verdict: FAIL
defect_class: scratches
class_confidence: 0.91
severity: major
zone: edge
mask_area_mm2: 14.2
anomaly_score: 2.400

Allowed SOP cards:
A06 [scratches] linear marks in the rolling direction
  - action: inspect the guide rail and adjust alignment
  - action: check the roller surface for trapped scale
  - probable cause: handling contact during transport
  - escalation: if scratches reappear after adjustment, escalate to the line supervisor

Correct answer:
ACTIONS: A06
EXPLAIN: The record shows a major scratches defect of 14.2 mm2 in the edge zone. Probable cause per card A06 is handling contact during transport; inspect the guide rail and adjust alignment.
LOG: P-0001 FAIL major scratches edge 14.2mm2 conf0.91"""


def format_record_for_prompt(record: DefectRecord) -> str:
    """Render the Defect Record as a compact, unambiguous block."""
    lines = [
        f"part_id: {record.part_id}",
        f"verdict: {record.verdict}",
    ]
    if record.defect_class:
        lines.append(f"defect_class: {record.defect_class}")
    if record.class_confidence is not None:
        lines.append(f"class_confidence: {record.class_confidence:.2f}")
    if record.severity:
        lines.append(f"severity: {record.severity}")
    if record.zone:
        lines.append(f"zone: {record.zone}")
    if record.mask_area_mm2 is not None:
        lines.append(f"mask_area_mm2: {record.mask_area_mm2:.1f}")
    if record.anomaly_score is not None:
        lines.append(f"anomaly_score: {record.anomaly_score:.3f}")
    if record.defect_bbox:
        x1, y1, x2, y2 = record.defect_bbox
        lines.append(f"bbox_xyxy: {x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}")
    if record.qc_flags:
        lines.append(f"quality_flags: {', '.join(record.qc_flags)}")
    return "\n".join(lines)


def build_prompt(
    record: DefectRecord,
    cards: list[SopCard],
    one_shot: bool = True,
    tokenizer=None,
) -> str:
    """The full prompt: rules + worked example + record + cards.

    When a tokenizer is supplied the prompt is wrapped in the Llama 3.2 chat
    template.  This matters: Llama 3.2 1B/3B are instruction-tuned models and
    they answer markedly better in their chat format than when the same
    content is fed as raw text (the raw-text version echoes the prompt).
    """
    card_block = "\n".join(c.to_prompt() for c in cards) if cards else "(no cards available)"
    user_parts = []
    if one_shot:
        user_parts.append(ONE_SHOT)
    user_parts.append(f"Defect Record:\n{format_record_for_prompt(record)}")
    user_parts.append(f"Allowed SOP cards:\n{card_block}")
    user_parts.append("Now answer with the three slots, in exactly the format of the example.")
    user = "\n\n".join(user_parts)

    if tokenizer is not None:
        try:
            text = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_RULES},
                    {"role": "user", "content": user},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            return text
        except Exception:
            pass  # fall back to the plain format below

    return f"{SYSTEM_RULES}\n\n{user}"


def build_template(record: DefectRecord, cards: list[SopCard]) -> str:
    """The deterministic fallback, in exactly the slot format."""
    ids = [c.id for c in cards[:3]]
    if record.defect_class:
        finding = f"{record.severity or 'minor'} {record.defect_class}"
    else:
        finding = "an anomaly"
    if record.mask_area_mm2 is not None:
        finding += f" of {record.mask_area_mm2:.1f} mm2"
    if record.zone:
        finding += f" in the {record.zone} zone"

    if record.verdict == "FAIL":
        action_text = cards[0].immediate_actions[0] if cards and cards[0].immediate_actions else "inspect the affected area"
        explain = (
            f"The record shows a {finding}. Probable cause per card {ids[0] if ids else 'A00'}: "
            f"{cards[0].probable_causes[0] if cards and cards[0].probable_causes else 'handling contact'}; "
            f"{action_text}."
        )
        actions = ", ".join(ids)
    elif record.verdict == "REVIEW":
        explain = (
            f"The record shows a {finding} that needs a human check before it is called. "
            "Re-image the part and inspect the marked region."
        )
        actions = ", ".join(ids[:1])
    else:
        explain = "The part matches the normal reference. No action is required."
        actions = ""

    log_bits = [record.part_id, record.verdict]
    if record.defect_class:
        log_bits.append(record.defect_class)
    if record.severity:
        log_bits.append(record.severity)
    if record.zone:
        log_bits.append(record.zone)
    if record.mask_area_mm2 is not None:
        log_bits.append(f"{record.mask_area_mm2:.1f}mm2")
    if record.class_confidence is not None:
        log_bits.append(f"conf{record.class_confidence:.2f}")
    return f"ACTIONS: {actions}\nEXPLAIN: {explain}\nLOG: {' '.join(log_bits)}"
