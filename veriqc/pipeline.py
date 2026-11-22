"""The single inspection pipeline used by the station and the evaluators."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from veriqc import REPO_ROOT
from veriqc.advisory.cards import load_cards
from veriqc.calibration import ProfileCalibration
from veriqc.config import Profile
from veriqc.decision import decide
from veriqc.inspection import Finding, InspectionResult, new_inspection_id
from veriqc.models import Detector, PatchCore, Segmenter
from veriqc.quality import QualityReport, assess_quality
from veriqc.severity import SeverityPolicy, zone_for_bbox

__all__ = ["Analysis", "Pipeline"]

_RULE_FOR_TASK = {
    "detect": "known_only",
    "segment": "known_only",
    "anomaly": "anomaly_only",
    "segment+anomaly": "dual",
}


@dataclass(frozen=True)
class Analysis:
    result: InspectionResult
    heatmap: np.ndarray | None = None
    known_mask: np.ndarray | None = None


class Pipeline:
    def __init__(self, profile: Profile, device: str = "cpu", overlay_root: Path | None = None) -> None:
        self.profile = profile
        self.device = device
        self.overlay_root = Path(overlay_root) if overlay_root is not None else REPO_ROOT / "artifacts"
        self.calibration = ProfileCalibration.load(profile.calibration_path)
        if self.calibration.profile_id != profile.profile_id:
            raise ValueError(
                f"calibration belongs to {self.calibration.profile_id!r}, not {profile.profile_id!r}"
            )
        expected_rule = _RULE_FOR_TASK[profile.task]
        if self.calibration.rule != expected_rule:
            raise ValueError(
                f"calibration rule {self.calibration.rule!r} does not match task "
                f"{profile.task!r} (expected {expected_rule!r})"
            )
        self.severity = SeverityPolicy.load(profile.severity_path)
        self.cards = load_cards(profile.cards_path)
        if self.cards.profile_id != profile.profile_id:
            raise ValueError(
                f"cards belong to {self.cards.profile_id!r}, not {profile.profile_id!r}"
            )

        self.detector: Detector | None = None
        self.segmenter: Segmenter | None = None
        self.anomaly: PatchCore | None = None
        if profile.known is not None:
            if profile.known.kind == "detector":
                self.detector = Detector.from_weights(
                    profile.known.path,
                    profile.known.classes,
                    input_size=int(profile.known.input_size or 640),
                    conf_threshold=profile.known.conf_threshold,
                    device=device,
                )
            elif profile.known.kind == "segmenter":
                self.segmenter = Segmenter.from_checkpoint(profile.known.path, device=device)
                self._validate_segmenter_geometry()
            else:
                raise ValueError(f"unsupported known-channel type: {profile.known.kind!r}")
        if profile.anomaly.enabled and profile.anomaly.path is not None:
            self.anomaly = PatchCore.load(profile.anomaly.path, device=device)
            self._validate_anomaly_geometry()
        self.artifact_hashes = profile.artifact_hashes()

    def _validate_segmenter_geometry(self) -> None:
        assert self.segmenter is not None
        expected = self.profile.known.input_size if self.profile.known is not None else None
        if expected is None:
            return
        expected_pair = tuple(expected) if isinstance(expected, tuple) else (int(expected), int(expected))
        actual = (self.segmenter.config.input_width, self.segmenter.config.input_height)
        if actual != expected_pair:
            raise ValueError(
                f"segmenter checkpoint geometry {actual} does not match profile {expected_pair}"
            )

    def _validate_anomaly_geometry(self) -> None:
        assert self.anomaly is not None
        spec = self.profile.anomaly
        actual = (
            self.anomaly.config.input_size,
            self.anomaly.config.backbone,
            self.anomaly.config.image_layer,
            self.anomaly.config.pixel_layer,
            self.anomaly.bank_size,
        )
        expected = (spec.input_size, spec.backbone, spec.image_layer, spec.pixel_layer, spec.bank_size)
        if actual != expected:
            raise ValueError(f"PatchCore checkpoint {actual} does not match profile {expected}")

    # -- channels ----------------------------------------------------------
    def _known(
        self, image: Image.Image
    ) -> tuple[list[dict[str, Any]], float | None, float, np.ndarray | None]:
        if self.profile.known is None:
            return [], None, 0.0, None
        if self.detector is not None:
            detections, latency = self.detector.predict(image)
            findings = []
            for detection in detections:
                if detection.class_name not in self.profile.known.classes:
                    continue
                x1, y1, x2, y2 = detection.bbox
                findings.append(
                    {
                        "class": detection.class_name,
                        "confidence": detection.confidence,
                        "bbox": (x1, y1, x2, y2),
                        "area": max(0.0, x2 - x1) * max(0.0, y2 - y1),
                    }
                )
            score = max((item["confidence"] for item in findings), default=0.0)
            return findings, score, latency, None
        if self.segmenter is not None:
            probability, latency = self.segmenter.predict_proba(image)
            threshold = float(self.profile.known.score_threshold or 0.5)
            mask = (probability >= threshold).astype(np.uint8)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            findings = []
            for index in range(1, count):
                area = float(stats[index, cv2.CC_STAT_AREA])
                if area < self.profile.known.min_area_px:
                    continue
                x = float(stats[index, cv2.CC_STAT_LEFT])
                y = float(stats[index, cv2.CC_STAT_TOP])
                w = float(stats[index, cv2.CC_STAT_WIDTH])
                h = float(stats[index, cv2.CC_STAT_HEIGHT])
                confidence = float(probability[labels == index].max())
                findings.append(
                    {
                        "class": self.profile.known.classes[0],
                        "confidence": confidence,
                        "bbox": (x, y, x + w, y + h),
                        "area": area,
                    }
                )
            score = float(probability.max())
            return findings, score, latency, probability
        return [], None, 0.0, None

    def _anomaly(self, image: Image.Image) -> tuple[float | None, np.ndarray | None, float]:
        if self.anomaly is None:
            return None, None, 0.0
        result = self.anomaly.score(image)
        return result.image_score, result.heatmap, result.latency_ms

    # -- inspection --------------------------------------------------------
    def analyze(self, image: Image.Image, part_id: str) -> Analysis:
        started = time.perf_counter()
        inspection_id = new_inspection_id()
        timestamp = time.time()
        image = image.convert("RGB")
        array = np.asarray(image)
        image_hash = hashlib.sha256(array.tobytes()).hexdigest()

        quality_started = time.perf_counter()
        quality = assess_quality(image, self.profile.quality)
        quality_ms = round((time.perf_counter() - quality_started) * 1000.0, 2)
        acquisition = {
            "qc_ok": quality.ok,
            "qc_flags": list(quality.flags),
            "statistics": quality.statistics,
            "hash_basis": "decoded_rgb_pixels",
        }
        if not quality.ok:
            return self._quality_review(
                image, part_id, inspection_id, timestamp, image_hash, acquisition, quality, quality_ms, started
            )

        latency: dict[str, float] = {"quality_ms": quality_ms}
        known_findings, known_score, known_latency, known_mask = self._known(image)
        latency["known_ms"] = known_latency
        anomaly_score, heatmap, anomaly_latency = self._anomaly(image)
        latency["anomaly_ms"] = anomaly_latency

        decision = decide(
            self.calibration,
            known_score=known_score,
            anomaly_score=anomaly_score,
            known_ok=known_score is not None,
            anomaly_ok=anomaly_score is not None,
        )

        findings: list[Finding] = []
        if decision.verdict != "PASS":
            if (
                not known_findings
                and known_mask is not None
                and known_score is not None
                and self.profile.known is not None
                and known_score >= float(self.profile.known.report_threshold)
            ):
                # A thin crack can score highly while every connected region
                # falls below the minimum component area.  Keep the result
                # localised rather than returning a bare verdict.
                y, x = np.unravel_index(int(np.argmax(known_mask)), known_mask.shape)
                half = 2
                x1 = max(0, x - half)
                y1 = max(0, y - half)
                x2 = min(known_mask.shape[1], x + half + 1)
                y2 = min(known_mask.shape[0], y + half + 1)
                known_findings = [
                    {
                        "class": self.profile.known.classes[0],
                        "confidence": float(known_score),
                        "bbox": (x1, y1, x2, y2),
                        "area": float((x2 - x1) * (y2 - y1)),
                    }
                ]
            if self.profile.known is not None:
                for index, item in enumerate(known_findings):
                    if float(item["confidence"]) < float(self.profile.known.report_threshold):
                        continue
                    bbox = tuple(float(value) for value in item["bbox"])
                    area = float(item["area"])
                    area_ratio = area / max(1.0, float(image.width * image.height))
                    zone = zone_for_bbox(bbox, image.size)
                    grade = self.severity.grade(item["class"], area_ratio, zone)
                    findings.append(
                        Finding(
                            finding_id=f"known-{index}",
                            source="known",
                            defect_class=str(item["class"]),
                            confidence=float(item["confidence"]),
                            score=float(item["confidence"]),
                            bbox=bbox,
                            area_px=area,
                            area_ratio=area_ratio,
                            zone=zone,
                            severity=grade.severity,
                            severity_status="policy",
                            severity_basis=grade.basis,
                            provenance={"channel": "known", "model": self.profile.known.kind},
                        )
                    )
            if heatmap is not None and anomaly_score is not None and decision.anomaly_p_value is not None:
                threshold = (
                    self.calibration.anomaly.threshold_for_alpha(self.calibration.alpha_review)
                    if self.calibration.anomaly is not None
                    else float(np.max(heatmap))
                )
                hot = (heatmap >= threshold).astype(np.uint8)
                count, labels, stats, _ = cv2.connectedComponentsWithStats(hot, connectivity=8)
                if count <= 1:
                    # The image score and the localization layer can disagree;
                    # a non-PASS anomaly must still carry a localised region.
                    cut = float(np.quantile(heatmap, 0.995))
                    hot = (heatmap >= cut).astype(np.uint8)
                    count, labels, stats, _ = cv2.connectedComponentsWithStats(hot, connectivity=8)
                if count > 1:
                    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                    area = float(stats[index, cv2.CC_STAT_AREA])
                    x = float(stats[index, cv2.CC_STAT_LEFT])
                    y = float(stats[index, cv2.CC_STAT_TOP])
                    w = float(stats[index, cv2.CC_STAT_WIDTH])
                    h = float(stats[index, cv2.CC_STAT_HEIGHT])
                    bbox = (x, y, x + w, y + h)
                    area_ratio = area / max(1.0, float(image.width * image.height))
                    zone = zone_for_bbox(bbox, image.size)
                    grade = self.severity.grade("unknown_anomaly", area_ratio, zone)
                    findings.append(
                        Finding(
                            finding_id="anomaly-0",
                            source="anomaly",
                            defect_class="unknown_anomaly",
                            score=float(anomaly_score),
                            bbox=bbox,
                            area_px=area,
                            area_ratio=area_ratio,
                            zone=zone,
                            severity=grade.severity,
                            severity_status="policy",
                            severity_basis=grade.basis,
                            provenance={"channel": "anomaly", "model": self.profile.anomaly.backbone},
                        )
                    )

        latency["total_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
        result = InspectionResult(
            inspection_id=inspection_id,
            timestamp=timestamp,
            part_id=part_id,
            profile_id=self.profile.profile_id,
            verdict=decision.verdict,
            findings=findings,
            anomaly_score=anomaly_score,
            anomaly_p_value=decision.anomaly_p_value,
            fused_p_value=decision.fused_p_value,
            known_score=known_score,
            known_p_value=decision.known_p_value,
            decision_rule=decision.rule,
            thresholds=decision.thresholds,
            acquisition=acquisition,
            latency_ms=latency,
            artifact_hashes=self.artifact_hashes,
            device=self.device,
            image_sha256=image_hash,
            reasons=list(decision.reasons),
        )
        return Analysis(result=result, heatmap=heatmap, known_mask=known_mask)

    def _quality_review(
        self,
        image: Image.Image,
        part_id: str,
        inspection_id: str,
        timestamp: float,
        image_hash: str,
        acquisition: dict[str, Any],
        quality: QualityReport,
        quality_ms: float,
        started: float,
    ) -> Analysis:
        result = InspectionResult(
            inspection_id=inspection_id,
            timestamp=timestamp,
            part_id=part_id,
            profile_id=self.profile.profile_id,
            verdict="REVIEW",
            acquisition=acquisition,
            latency_ms={
                "quality_ms": quality_ms,
                "total_ms": round((time.perf_counter() - started) * 1000.0, 2),
            },
            artifact_hashes=self.artifact_hashes,
            device=self.device,
            image_sha256=image_hash,
            reasons=[f"quality gate: {', '.join(quality.flags)}"],
        )
        return Analysis(result=result)

    def inspect(self, image: Image.Image, part_id: str, *, overlay: bool = True) -> InspectionResult:
        analysis = self.analyze(image, part_id)
        if overlay:
            analysis.result.overlay = self._save_overlay(
                image.convert("RGB"), analysis.result, analysis.heatmap, analysis.known_mask
            )
        return analysis.result

    # -- rendering ---------------------------------------------------------
    def _save_overlay(
        self,
        image: Image.Image,
        result: InspectionResult,
        heatmap: np.ndarray | None,
        known_mask: np.ndarray | None,
    ) -> str | None:
        canvas = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        if heatmap is not None and result.verdict != "PASS":
            normalized = heatmap.astype(np.float32)
            high = float(normalized.max())
            if high > 0:
                normalized = normalized / high
            colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            canvas = cv2.addWeighted(canvas, 0.65, colored, 0.35, 0)
        if known_mask is not None and result.verdict != "PASS" and self.profile.known is not None:
            threshold = float(self.profile.known.score_threshold or 0.5)
            binary = (known_mask >= threshold).astype(np.uint8)
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(canvas, contours, -1, (0, 255, 0), 2)
        for finding in result.findings:
            if finding.bbox is None:
                continue
            x1, y1, x2, y2 = (int(round(v)) for v in finding.bbox)
            color = (0, 200, 255) if finding.source == "known" else (0, 0, 255)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = f"{finding.defect_class} {finding.severity or ''}".strip()
            cv2.putText(
                canvas,
                label,
                (x1, max(14, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        verdict = result.verdict
        color = (0, 200, 0) if verdict == "PASS" else (0, 0, 255) if verdict == "FAIL" else (0, 200, 255)
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, 36), (20, 20, 20), -1)
        cv2.putText(
            canvas,
            f"{verdict} | {result.profile_id} | {result.part_id}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
        target = self.overlay_root / "overlays" / f"{result.inspection_id}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), canvas)
        try:
            return str(target.relative_to(REPO_ROOT))
        except ValueError:
            return str(target)
