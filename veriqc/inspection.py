"""Canonical inspection result and finding schema."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping

__all__ = ["Finding", "InspectionResult", "new_inspection_id"]

Severity = Literal["minor", "major", "critical"]
Verdict = Literal["PASS", "REVIEW", "FAIL"]


def new_inspection_id() -> str:
    return uuid.uuid4().hex


@dataclass(frozen=True)
class Finding:
    finding_id: str
    source: Literal["known", "anomaly"]
    defect_class: str
    confidence: float | None = None
    score: float | None = None
    bbox: tuple[float, float, float, float] | None = None
    area_px: float | None = None
    area_ratio: float | None = None
    region_p_value: float | None = None
    zone: str | None = None
    severity: Severity | None = None
    severity_status: Literal["none", "policy"] = "none"
    severity_basis: Mapping[str, float | str] = field(default_factory=dict)
    provenance: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["bbox"] = list(self.bbox) if self.bbox is not None else None
        data["severity_basis"] = dict(self.severity_basis)
        data["provenance"] = dict(self.provenance)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Finding":
        bbox = data.get("bbox")
        return cls(
            finding_id=str(data["finding_id"]),
            source=data["source"],
            defect_class=str(data["defect_class"]),
            confidence=data.get("confidence"),
            score=data.get("score"),
            bbox=tuple(float(v) for v in bbox) if bbox else None,
            area_px=data.get("area_px"),
            area_ratio=data.get("area_ratio"),
            region_p_value=data.get("region_p_value"),
            zone=data.get("zone"),
            severity=data.get("severity"),
            severity_status=data.get("severity_status", "none"),
            severity_basis=dict(data.get("severity_basis", {})),
            provenance=dict(data.get("provenance", {})),
        )


@dataclass
class InspectionResult:
    inspection_id: str
    timestamp: float
    part_id: str
    profile_id: str
    verdict: Verdict
    findings: list[Finding] = field(default_factory=list)
    anomaly_score: float | None = None
    anomaly_p_value: float | None = None
    # Reserved for the rejected dual-channel ablation; deployed profiles leave it None.
    fused_p_value: float | None = None
    known_score: float | None = None
    known_p_value: float | None = None
    decision_rule: str = ""
    thresholds: dict[str, float] = field(default_factory=dict)
    acquisition: dict[str, Any] = field(default_factory=dict)
    latency_ms: dict[str, float] = field(default_factory=dict)
    artifact_hashes: dict[str, str] = field(default_factory=dict)
    backend: str = ""
    device: str = ""
    image_sha256: str = ""
    overlay: str | None = None
    reasons: list[str] = field(default_factory=list)
    advisory: dict[str, Any] | None = None
    operator: dict[str, Any] | None = None

    def to_dict(self, include_record_hash: bool = False) -> dict[str, Any]:
        data = asdict(self)
        data["findings"] = [finding.to_dict() for finding in self.findings]
        if include_record_hash:
            data["record_hash"] = self.record_hash()
        return data

    def record_hash(self) -> str:
        payload = self.to_dict(include_record_hash=False)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InspectionResult":
        return cls(
            inspection_id=str(data["inspection_id"]),
            timestamp=float(data["timestamp"]),
            part_id=str(data["part_id"]),
            profile_id=str(data["profile_id"]),
            verdict=data["verdict"],
            findings=[Finding.from_dict(item) for item in data.get("findings", [])],
            anomaly_score=data.get("anomaly_score"),
            anomaly_p_value=data.get("anomaly_p_value"),
            fused_p_value=data.get("fused_p_value"),
            known_score=data.get("known_score"),
            known_p_value=data.get("known_p_value"),
            decision_rule=str(data.get("decision_rule", "")),
            thresholds={str(k): float(v) for k, v in data.get("thresholds", {}).items()},
            acquisition=dict(data.get("acquisition", {})),
            latency_ms={str(k): float(v) for k, v in data.get("latency_ms", {}).items()},
            artifact_hashes={str(k): str(v) for k, v in data.get("artifact_hashes", {}).items()},
            backend=str(data.get("backend", "")),
            device=str(data.get("device", "")),
            image_sha256=str(data.get("image_sha256", "")),
            overlay=data.get("overlay"),
            reasons=[str(v) for v in data.get("reasons", [])],
            advisory=data.get("advisory"),
            operator=data.get("operator"),
        )
