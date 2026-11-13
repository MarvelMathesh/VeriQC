"""The Defect Record: typed, immutable contract between vision and language.

The vision pipeline outputs defect class, severity, physical area, confidence,
and quality flags into this structured record. Downstream advisory models
are constrained strictly to the evidence present in this record.
Includes deterministic canonical JSON serialization and SHA-256 content hashing.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = ["Verdict", "Severity", "DefectRecord", "SCHEMA_VERSION"]

SCHEMA_VERSION = "veriqc.record/1.0"

Verdict = str  # "PASS" | "REVIEW" | "FAIL"
Severity = str  # "minor" | "major" | "critical" | None


@dataclass
class DefectRecord:
    """One inspection decision and everything needed to audit it."""

    part_id: str
    verdict: Verdict
    timestamp: float = field(default_factory=time.time)

    # evidence
    defect_class: str | None = None
    class_confidence: float | None = None
    defect_bbox: tuple[float, float, float, float] | None = None  # xyxy, pixels
    zone: str | None = None
    anomaly_score: float | None = None
    mask_area_px: float | None = None
    mask_area_mm2: float | None = None
    severity: Severity = None
    severity_reasons: list[str] = field(default_factory=list)

    # decision bookkeeping
    thresholds: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    qc_flags: list[str] = field(default_factory=list)
    qc_ok: bool = True
    triage_rule: str = ""
    model_versions: dict[str, str] = field(default_factory=dict)
    latency_ms: dict[str, float] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        if d["defect_bbox"] is not None:
            d["defect_bbox"] = [float(v) for v in d["defect_bbox"]]
        d["schema_version"] = SCHEMA_VERSION
        return d

    def to_json(self) -> str:
        """Canonical JSON: sorted keys, no whitespace drift."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=str)

    def content_hash(self) -> str:
        """SHA-256 of the canonical JSON (the record's identity)."""
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:16]

    # -- helpers used by the verifier -------------------------------------
    def numbers(self) -> list[float]:
        """Every numeric value the advisory is allowed to mention."""
        vals: list[float] = []
        for v in (
            self.class_confidence,
            self.anomaly_score,
            self.mask_area_px,
            self.mask_area_mm2,
        ):
            if v is not None:
                vals.append(float(v))
        if self.defect_bbox:
            vals.extend(float(x) for x in self.defect_bbox)
        return vals

    def class_words(self) -> list[str]:
        """Words that name the defect class in the advisory."""
        if not self.defect_class:
            return []
        return [w for w in self.defect_class.replace("-", " ").replace("_", " ").split() if w]
