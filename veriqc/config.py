"""Profile configuration and artifact integrity.

A profile is one inspection problem: a source, a known-defect model, an
optional normal-only anomaly model, one calibration file, one severity policy
and one advisory card set.  Paths are repository-relative.  When
``configs/artifacts.json`` is present, every runtime artifact is checked
against its recorded SHA-256 before the profile is accepted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from veriqc import REPO_ROOT, repo_path, sha256_file
from veriqc.quality import QualitySpec

__all__ = ["Profile", "load_profile", "list_profiles", "ARTIFACTS_MANIFEST"]

ARTIFACTS_MANIFEST = REPO_ROOT / "configs" / "artifacts.json"
_TASKS = ("detect", "segment", "anomaly", "segment+anomaly")
_KNOWN_TASKS = ("detect", "segment", "segment+anomaly")
_ANOMALY_TASKS = ("anomaly", "segment+anomaly")


@dataclass(frozen=True)
class ModelSpec:
    kind: str
    weights: str
    input_size: int | tuple[int, int] | None = None
    classes: tuple[str, ...] = ()
    conf_threshold: float = 0.01
    iou_threshold: float = 0.5
    score_threshold: float | None = None
    report_threshold: float = 0.25
    min_area_px: int = 8

    @property
    def path(self) -> Path:
        return repo_path(self.weights)


@dataclass(frozen=True)
class AnomalySpec:
    enabled: bool
    weights: str | None
    input_size: int = 448
    backbone: str = "resnet18"
    image_layer: str = "layer2"
    pixel_layer: str = "layer3"
    bank_size: int = 4000

    @property
    def path(self) -> Path | None:
        return repo_path(self.weights) if self.weights else None


@dataclass(frozen=True)
class Profile:
    profile_id: str
    domain: str
    source: str
    task: str
    known: ModelSpec | None
    anomaly: AnomalySpec
    calibration: str
    severity: str
    cards: str
    quality: QualitySpec = field(default_factory=QualitySpec)
    description: str = ""
    config_path: str = ""
    expected_hashes: Mapping[str, str] = field(default_factory=dict)

    @property
    def calibration_path(self) -> Path:
        return repo_path(self.calibration)

    @property
    def severity_path(self) -> Path:
        return repo_path(self.severity)

    @property
    def cards_path(self) -> Path:
        return repo_path(self.cards)

    def artifact_paths(self) -> dict[str, Path]:
        paths: dict[str, Path] = {
            "profile": Path(self.config_path),
            "calibration": self.calibration_path,
            "severity": self.severity_path,
            "cards": self.cards_path,
        }
        if self.known is not None:
            paths["known_weights"] = self.known.path
        if self.anomaly.enabled and self.anomaly.path is not None:
            paths["anomaly_weights"] = self.anomaly.path
        return paths

    def artifact_hashes(self) -> dict[str, str]:
        return {name: sha256_file(path) for name, path in self.artifact_paths().items()}

    def validate(self) -> None:
        if self.task not in _TASKS:
            raise ValueError(f"unsupported task: {self.task!r}")
        if self.task in _KNOWN_TASKS and self.known is None:
            raise ValueError(f"profile {self.profile_id!r} has no known-channel model")
        if self.task in _ANOMALY_TASKS and not self.anomaly.enabled:
            raise ValueError(f"profile {self.profile_id!r} requires the anomaly channel")
        if self.task not in _ANOMALY_TASKS and self.anomaly.enabled:
            raise ValueError(f"profile {self.profile_id!r} enables anomaly outside a dual task")
        if self.task == "anomaly" and self.known is not None:
            raise ValueError(f"profile {self.profile_id!r} has a known model but an anomaly-only task")
        if self.known is not None and self.known.kind not in ("detector", "segmenter"):
            raise ValueError(f"unsupported known-channel type: {self.known.kind!r}")
        if self.anomaly.backbone != "resnet18":
            raise ValueError(f"unsupported anomaly backbone: {self.anomaly.backbone!r}")
        if self.anomaly.image_layer not in ("layer2", "layer3"):
            raise ValueError(f"unsupported anomaly layer: {self.anomaly.image_layer!r}")
        if self.anomaly.pixel_layer not in ("layer2", "layer3"):
            raise ValueError(f"unsupported anomaly layer: {self.anomaly.pixel_layer!r}")

        paths = self.artifact_paths()
        for name, path in paths.items():
            if not path.exists():
                raise FileNotFoundError(f"profile {self.profile_id!r}: missing {name} at {path}")
        if self.expected_hashes:
            missing = sorted(set(paths) - set(self.expected_hashes))
            extra = sorted(set(self.expected_hashes) - set(paths))
            if missing or extra:
                raise ValueError(
                    f"profile {self.profile_id!r}: artifact manifest keys do not match "
                    f"(missing={missing}, extra={extra})"
                )
            for name, path in paths.items():
                actual = sha256_file(path)
                expected = self.expected_hashes[name]
                if actual != expected:
                    raise ValueError(
                        f"profile {self.profile_id!r}: {name} hash mismatch "
                        f"(expected {expected[:12]}, got {actual[:12]})"
                    )


def _known_spec(raw: dict[str, Any] | None) -> ModelSpec | None:
    if not raw:
        return None
    size = raw.get("input_size")
    if isinstance(size, list):
        size = tuple(int(v) for v in size)
    elif size is not None:
        size = int(size)
    return ModelSpec(
        kind=str(raw["type"]),
        weights=str(raw["weights"]),
        input_size=size,
        classes=tuple(str(v) for v in raw.get("classes", ())),
        conf_threshold=float(raw.get("conf_threshold", 0.01)),
        iou_threshold=float(raw.get("iou_threshold", 0.5)),
        score_threshold=(float(raw["score_threshold"]) if raw.get("score_threshold") is not None else None),
        report_threshold=float(raw.get("report_threshold", 0.25)),
        min_area_px=int(raw.get("min_area_px", 8)),
    )


def _expected_hashes(profile_id: str) -> dict[str, str]:
    if not ARTIFACTS_MANIFEST.exists():
        return {}
    payload = json.loads(ARTIFACTS_MANIFEST.read_text())
    entry = payload.get("profiles", {}).get(profile_id) or {}
    return {str(key): str(value) for key, value in entry.items()}


def load_profile(path: str | Path, *, validate: bool = True) -> Profile:
    config_path = Path(path).resolve()
    data = yaml.safe_load(config_path.read_text()) or {}
    anomaly = data.get("anomaly") or {}
    anomaly_layers = [str(value) for value in (anomaly.get("layers") or ["layer2", "layer3"])]
    quality = data.get("quality") or {}
    profile_id = str(data["id"])
    profile = Profile(
        profile_id=profile_id,
        domain=str(data.get("domain", profile_id)),
        source=str(data["source"]),
        task=str(data["task"]),
        known=_known_spec(data.get("known")),
        anomaly=AnomalySpec(
            enabled=bool(anomaly.get("enabled", False)),
            weights=(str(anomaly["weights"]) if anomaly.get("weights") else None),
            input_size=int(anomaly.get("input_size", 448)),
            backbone=str(anomaly.get("backbone", "resnet18")),
            image_layer=anomaly_layers[0],
            pixel_layer=anomaly_layers[1] if len(anomaly_layers) > 1 else anomaly_layers[0],
            bank_size=int(anomaly.get("bank_size", 4000)),
        ),
        calibration=str(data["calibration"]),
        severity=str(data["severity"]),
        cards=str(data["cards"]),
        quality=QualitySpec(
            min_width=int(quality.get("min_width", 64)),
            min_height=int(quality.get("min_height", 64)),
            blur_ratio_min=float(quality.get("blur_ratio_min", 0.02)),
            clipped_max=float(quality.get("clipped_max", 0.25)),
            max_pixels=int(quality.get("max_pixels", 25_000_000)),
        ),
        description=str(data.get("description", "")),
        config_path=str(config_path),
        expected_hashes=_expected_hashes(profile_id),
    )
    if validate:
        profile.validate()
    return profile


def list_profiles(directory: str | Path = REPO_ROOT / "configs" / "profiles") -> list[Path]:
    return sorted(Path(directory).glob("*.yaml"))
