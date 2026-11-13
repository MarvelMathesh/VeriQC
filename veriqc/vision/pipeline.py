"""Integrated Vision Pipeline for VeriQC.

Connects:
1. QC Image Gate (Laplacian blur, luminance, saturation)
2. Anomaly Gate (EfficientNet-B0 + Coreset-kNN memory bank)
3. Defect Detector (YOLO26 nano) via cascade (runs above review threshold)
4. Calibrated Triage Engine (Agreement rule + class floors)
5. Deterministic Severity Policy Engine (physical mm2, zone weights, class risk)

Outputs a typed, audit-hashed DefectRecord.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from veriqc.contract.qc import QualityCalibration, assess_quality
from veriqc.contract.record import DefectRecord, Severity, Verdict
from veriqc.contract.severity import SeverityPolicy, load_policy
from veriqc.contract.triage import TriageConfig, triage
from veriqc.vision.detector import YOLO26Detector
from veriqc.vision.gate import AnomalyGate

__all__ = ["VisionPipeline", "PipelineConfig"]


@dataclass
class PipelineConfig:
    """Configuration and thresholds for the end-to-end vision pipeline."""

    threshold_review: float = 0.35
    threshold_fail: float = 0.55
    triage_config: TriageConfig = field(default_factory=TriageConfig)
    always_run_detector: bool = False  # Set True for Hailo NPU, False for CPU cascade
    default_zone: str = "surface"


class VisionPipeline:
    """The unified on-device vision inspection pipeline."""

    def __init__(
        self,
        anomaly_gate: AnomalyGate,
        detector: YOLO26Detector,
        severity_policy: SeverityPolicy,
        quality_calibration: QualityCalibration | None = None,
        config: PipelineConfig | None = None,
    ):
        self.gate = anomaly_gate
        self.detector = detector
        self.severity_policy = severity_policy
        self.quality_calibration = quality_calibration
        self.config = config or PipelineConfig()

    def inspect(
        self,
        image_input: str | Path | Image.Image | np.ndarray,
        part_id: str,
    ) -> DefectRecord:
        """Inspect a single part and produce a typed DefectRecord."""
        t_start = time.perf_counter()
        latencies: dict[str, float] = {}

        # 1. Image loading
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        # 2. Image Quality Gate
        t0 = time.perf_counter()
        qc_ok = True
        qc_flags: list[str] = []
        if self.quality_calibration is not None:
            q_report = assess_quality(pil_img, self.quality_calibration)
            qc_ok = q_report.ok
            qc_flags = q_report.flags
        latencies["qc_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        # Early return if optical corruption detected (converts bad frame to REVIEW)
        if not qc_ok:
            latencies["total_ms"] = round((time.perf_counter() - t_start) * 1000.0, 2)
            return DefectRecord(
                part_id=part_id,
                verdict="REVIEW",
                qc_ok=False,
                qc_flags=qc_flags,
                reasons=[f"Image quality gate failed: {', '.join(qc_flags)}"],
                thresholds={
                    "review": self.config.threshold_review,
                    "fail": self.config.threshold_fail,
                },
                latency_ms=latencies,
                model_versions={
                    "gate": self.gate.config.backbone,
                    "detector": "yolo26n",
                    "severity_policy": str(self.severity_policy.version),
                },
            )

        # 3. Anomaly Gate
        t0 = time.perf_counter()
        gate_res = self.gate.score([pil_img])
        anomaly_score = float(gate_res.image_scores[0])
        heatmap = gate_res.heatmaps[0]
        latencies["gate_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        # 4. Detector with Cascade logic
        # On CPU, only run detector if score exceeds review threshold to save cycles
        t0 = time.perf_counter()
        run_detector = (
            self.config.always_run_detector
            or anomaly_score > self.config.threshold_review
        )

        det_dict: dict | None = None
        best_box: tuple[float, float, float, float] | None = None
        if run_detector:
            det_res = self.detector.predict([pil_img])
            det_out = det_res.detections[0]
            if len(det_out.scores) > 0:
                best_idx = int(np.argmax(det_out.scores))
                class_id = int(det_out.labels[best_idx])
                conf = float(det_out.scores[best_idx])
                box = det_out.boxes[best_idx]
                best_box = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))

                # Map class ID to class name
                class_names = getattr(self.detector.model, "names", {})
                class_name = class_names.get(class_id, f"defect_{class_id}")
                det_dict = {
                    "class": class_name,
                    "confidence": conf,
                    "bbox": list(best_box),
                }
        latencies["detector_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        # 5. Calibrated Triage
        t0 = time.perf_counter()
        thresholds = {
            "review": self.config.threshold_review,
            "fail": self.config.threshold_fail,
        }
        decision = triage(
            anomaly_score=anomaly_score,
            detection=det_dict,
            thresholds=thresholds,
            config=self.config.triage_config,
            qc_ok=True,
            qc_flags=[],
        )
        latencies["triage_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        # 6. Physical Area & Severity Engine
        defect_class = det_dict.get("class") if det_dict else None
        class_conf = det_dict.get("confidence") if det_dict else None
        zone = self.config.default_zone

        mask_area_px = None
        mask_area_mm2 = None
        severity: Severity = None
        severity_reasons: list[str] = []

        if decision.verdict != "PASS":
            # Estimate area from heatmap regions exceeding the threshold
            bin_mask = heatmap > self.config.threshold_review
            mask_area_px = float(np.sum(bin_mask))
            mm_per_px = self.severity_policy.mm_per_px
            mask_area_mm2 = mask_area_px * (mm_per_px**2)

            sev_outcome = self.severity_policy.severity(
                area_px=mask_area_px,
                defect_class=defect_class,
                zone=zone,
                score=anomaly_score,
                fail_threshold=self.config.threshold_fail,
            )
            severity = sev_outcome.severity
            severity_reasons = sev_outcome.reasons

        latencies["total_ms"] = round((time.perf_counter() - t_start) * 1000.0, 2)

        return DefectRecord(
            part_id=part_id,
            verdict=decision.verdict,
            defect_class=defect_class,
            class_confidence=class_conf,
            defect_bbox=best_box,
            zone=zone,
            anomaly_score=round(anomaly_score, 4),
            mask_area_px=mask_area_px,
            mask_area_mm2=round(mask_area_mm2, 2) if mask_area_mm2 is not None else None,
            severity=severity,
            severity_reasons=severity_reasons,
            thresholds=thresholds,
            reasons=decision.reasons,
            qc_ok=True,
            qc_flags=[],
            triage_rule=self.config.triage_config.rule,
            model_versions={
                "gate": self.gate.config.backbone,
                "detector": "yolo26n",
                "severity_policy": str(self.severity_policy.version),
            },
            latency_ms=latencies,
        )
