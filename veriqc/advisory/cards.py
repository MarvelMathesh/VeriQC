"""Standard Operating Procedure (SOP) cards for edge advisory reasoning.

Provides structured domain knowledge: defect class, hedged probable causes,
immediate operator actions, escalation rules, and stable alphanumeric IDs.
Retrieved cards form the bounded grounding set for the language advisor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

__all__ = ["SopCard", "CardLibrary", "load_cards"]


@dataclass
class SopCard:
    id: str
    defect_class: str
    summary: str
    probable_causes: list[str] = field(default_factory=list)
    immediate_actions: list[str] = field(default_factory=list)
    escalation: str = ""
    severity_scope: list[str] = field(default_factory=list)
    requires_zone: bool = False

    def to_prompt(self) -> str:
        lines = [f"{self.id} [{self.defect_class}] {self.summary}"]
        for a in self.immediate_actions:
            lines.append(f"  - action: {a}")
        for c in self.probable_causes:
            lines.append(f"  - probable cause: {c}")
        if self.escalation:
            lines.append(f"  - escalation: {self.escalation}")
        return "\n".join(lines)


@dataclass
class CardLibrary:
    cards: list[SopCard]

    def __post_init__(self) -> None:
        self._by_id = {c.id: c for c in self.cards}

    def by_id(self, card_id: str) -> SopCard | None:
        return self._by_id.get(card_id)

    def retrieve(self, defect_class: str | None, severity: str | None, limit: int = 3) -> list[SopCard]:
        """Cards for a class (and optionally a severity), capped at ``limit``.

        If the class is unknown or has no cards, the generic cards are used so
        that the advisor can still say something useful.
        """
        out = [c for c in self.cards if c.defect_class == defect_class]
        if severity:
            scoped = [c for c in out if not c.severity_scope or severity in c.severity_scope]
            if scoped:
                out = scoped
        if not out:
            out = [c for c in self.cards if c.defect_class == "generic"]
        return out[:limit]

    def all_ids(self) -> set[str]:
        return set(self._by_id)


def load_cards(path: str | Path) -> CardLibrary:
    data = yaml.safe_load(Path(path).read_text()) or {}
    raw = data.get("cards", data if isinstance(data, list) else [])
    cards = []
    for c in raw:
        cards.append(
            SopCard(
                id=str(c["id"]),
                defect_class=str(c.get("defect_class", "generic")),
                summary=str(c.get("summary", "")),
                probable_causes=[str(x) for x in c.get("probable_causes", [])],
                immediate_actions=[str(x) for x in c.get("immediate_actions", [])],
                escalation=str(c.get("escalation", "")),
                severity_scope=[str(x) for x in c.get("severity_scope", [])],
                requires_zone=bool(c.get("requires_zone", False)),
            )
        )
    return CardLibrary(cards)
