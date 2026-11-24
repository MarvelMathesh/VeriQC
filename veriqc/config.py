"""Profile configuration and artifact integrity.

A profile is one inspection problem: a source, exactly one vision channel, one
calibration file, one severity policy and one advisory card set.  The
production runtime is ONNX Runtime on CPU; the ``.pt`` paths are training and
reference inputs used by the export and calibration commands.

Paths are repository-relative.  When ``configs/artifacts.json`` is present,
every runtime artifact is checked against its recorded SHA-256 before the
profile is accepted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from veriqc import REPO_ROOT, repo_path, sha256_file
from veriqc.quality import QualitySpec

__all__ = ["Profile", "load_profile", "list_profiles", "profile_by_source"]

ARTIFACTS_MANIFEST = REPO_ROOT / "configs" / "artifacts.json"
_TASKS = ("detect", "segment", "anomaly")


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
    weights: str
    input_size: int = 448
    backbone: str = "resnet18"
    image_layer: str = "layer2"
    pixel_layer: str = "layer3"
    bank_size: int = 4000

    @property
    def path(self) -> Path:
        return repo_path(self.weights)


@dataclass(frozen=True)
class Profile:
    profile_id: str
    domain: str
    source: str
    task: str
    known: ModelSpec | None
    anomaly: AnomalySpec | None
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

    @property
    def export_dir(self) -> Path:
        return REPO_ROOT / "models" / "profiles" / self.profile_id / "export"

    def runtime_artifact_paths(self) -> dict[str, Path]:
        """Files required by the deployed ONNX runtime."""

        paths: dict[str, Path] = {}
        if self.known is not None:
            name = "detector.onnx" if self.known.kind == "detector" else "segmenter.onnx"
            paths["known_runtime"] = self.export_dir / name
            external = self.export_dir / f"{name}.data"
            if external.exists():
                paths["known_runtime_data"] = external
        if self.anomaly is not None:
            paths["anomaly_backbone"] = self.export_dir / "anomaly_backbone.onnx"
            external = self.export_dir / "anomaly_backbone.onnx.data"
            if external.exists():
                paths["anomaly_backbone_data"] = external
            paths["anomaly_banks"] = self.export_dir / "anomaly_banks.npz"
        return paths

    def artifact_paths(self) -> dict[str, Path]:
        paths: dict[str, Path] = {
            "profile": Path(self.config_path),
            "calibration": self.calibration_path,
            "severity": self.severity_path,
            "cards": self.cards_path,
        }
        paths.update(self.runtime_artifact_paths())
        return paths

    def artifact_hashes(self) -> dict[str, str]:
        return {name: sha256_file(path) for name, path in self.artifact_paths().items()}

    def validate(self) -> None:
        if self.task not in _TASKS:
            raise ValueError(f"unsupported task: {self.task!r}")
        if self.task in ("detect", "segment"):
            if self.known is None:
                raise ValueError(f"profile {self.profile_id!r} has no known-channel model")
            if self.anomaly is not None:
                raise ValueError(
                    f"profile {self.profile_id!r} cannot enable the anomaly channel "
                    f"for task {self.task!r}"
                )
            expected_kind = "detector" if self.task == "detect" else "segmenter"
            if self.known.kind != expected_kind:
                raise ValueError(
                    f"profile {self.profile_id!r} task {self.task!r} requires a "
                    f"{expected_kind} model, not {self.known.kind!r}"
                )
            if not self.known.classes:
                raise ValueError(f"profile {self.profile_id!r} has no known classes")
        else:
            if self.anomaly is None:
                raise ValueError(f"profile {self.profile_id!r} requires the anomaly channel")
            if self.known is not None:
                raise ValueError(
                    f"profile {self.profile_id!r} cannot combine a known model with "
                    "the anomaly-only task"
                )
        if self.known is not None and self.known.kind not in ("detector", "segmenter"):
            raise ValueError(f"unsupported known-channel type: {self.known.kind!r}")
        if self.anomaly is not None:
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
        score_threshold=(
            float(raw["score_threshold"]) if raw.get("score_threshold") is not None else None
        ),
        report_threshold=float(raw.get("report_threshold", 0.25)),
        min_area_px=int(raw.get("min_area_px", 8)),
    )


def _anomaly_spec(raw: dict[str, Any] | None) -> AnomalySpec | None:
    if not raw or not bool(raw.get("enabled", True)):
        return None
    layers = [str(value) for value in (raw.get("layers") or ["layer2", "layer3"])]
    return AnomalySpec(
        weights=str(raw["weights"]),
        input_size=int(raw.get("input_size", 448)),
        backbone=str(raw.get("backbone", "resnet18")),
        image_layer=layers[0],
        pixel_layer=layers[1] if len(layers) > 1 else layers[0],
        bank_size=int(raw.get("bank_size", 4000)),
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
    quality = data.get("quality") or {}
    profile_id = str(data["id"])
    profile = Profile(
        profile_id=profile_id,
        domain=str(data.get("domain", profile_id)),
        source=str(data["source"]),
        task=str(data["task"]),
        known=_known_spec(data.get("known")),
        anomaly=_anomaly_spec(data.get("anomaly")),
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


def profile_by_source(source: str) -> Profile:
    """Resolve a profile by its declared source, without runtime validation."""

    for path in list_profiles():
        data = yaml.safe_load(path.read_text()) or {}
        if str(data.get("source")) == source:
            return load_profile(path, validate=False)
    raise KeyError(f"no profile declares source {source!r}")
