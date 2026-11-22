"""Advisory cards and bounded SOP retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

__all__ = ["Card", "CardLibrary", "load_cards"]


@dataclass(frozen=True)
class Card:
    id: str
    defect_class: str
    summary: str
    probable_causes: tuple[str, ...] = ()
    immediate_actions: tuple[str, ...] = ()
    escalation: str = ""
    severity_scope: tuple[str, ...] = ()
    profile_id: str = ""

@dataclass
class CardLibrary:
    profile_id: str
    cards: list[Card] = field(default_factory=list)

    def __post_init__(self) -> None:
        seen = set()
        for card in self.cards:
            if card.id in seen:
                raise ValueError(f"duplicate card id {card.id!r}")
            seen.add(card.id)
            if card.profile_id and card.profile_id != self.profile_id:
                raise ValueError(f"card {card.id!r} belongs to {card.profile_id!r}")

    def retrieve(self, defect_class: str | None, severity: str | None = None) -> list[Card]:
        direct = [card for card in self.cards if card.defect_class == defect_class]
        if severity:
            scoped = [card for card in direct if not card.severity_scope or severity in card.severity_scope]
            if scoped:
                direct = scoped
        if direct:
            return direct
        return [card for card in self.cards if card.defect_class == "generic"]

    def primary(self, defect_class: str | None, severity: str | None = None) -> Card | None:
        cards = self.retrieve(defect_class, severity)
        return cards[0] if cards else None

    def template(self, card: Card, defect_class: str) -> str:
        causes = "; ".join(card.probable_causes[:2])
        actions = "; ".join(card.immediate_actions[:2])
        escalation = f" Escalate if: {card.escalation}." if card.escalation else ""
        return (
            f"{defect_class.replace('_', ' ').capitalize()}: {card.summary}. "
            f"Probable cause: {causes}. Do now: {actions}.{escalation}"
        )


def load_cards(path: str | Path) -> CardLibrary:
    data = yaml.safe_load(Path(path).read_text()) or {}
    profile_id = str(data.get("profile_id", ""))
    if not profile_id:
        raise ValueError("card file has no profile_id")
    cards = []
    for raw in data.get("cards", []):
        cards.append(
            Card(
                id=str(raw["id"]),
                defect_class=str(raw["defect_class"]),
                summary=str(raw.get("summary", "")),
                probable_causes=tuple(str(v) for v in raw.get("probable_causes", ())),
                immediate_actions=tuple(str(v) for v in raw.get("immediate_actions", ())),
                escalation=str(raw.get("escalation", "")),
                severity_scope=tuple(str(v) for v in raw.get("severity_scope", ())),
                profile_id=profile_id,
            )
        )
    return CardLibrary(profile_id=profile_id, cards=cards)
