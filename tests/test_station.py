"""Unit tests for the end-to-end decoupled inspection station."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
import pytest
import numpy as np
from PIL import Image

from veriqc.contract.qc import QualityBand, QualityCalibration
from veriqc.contract.severity import SeverityPolicy
from veriqc.contract.triage import TriageConfig
from veriqc.station.engine import InspectionStation, StationConfig
from veriqc.storage.db import InspectionDatabase
from veriqc.vision.detector import YOLO26Detector
from veriqc.vision.gate import AnomalyGate, GateConfig
from veriqc.vision.pipeline import VisionPipeline, PipelineConfig


class _DummyModel:
    names = {0: "scratches", 1: "inclusion"}

    def predict(self, *args, **kwargs):
        class _Box:
            xyxy = torch.zeros((0, 4))
            conf = torch.zeros((0,))
            cls = torch.zeros((0,), dtype=torch.int64)

        class _Res:
            boxes = _Box()

        return [_Res()]


import torch


def test_station_end_to_end_inspection():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "station_audit.db"
        db = InspectionDatabase(db_path)

        # Vision pipeline components
        gate_cfg = GateConfig(backbone="mobilenet_v3_small", input_size=128, coreset_ratio=0.2, device="cpu")
        gate = AnomalyGate(gate_cfg)
        normals = [Image.fromarray(np.full((128, 128, 3), 100, dtype=np.uint8)) for _ in range(5)]
        gate.fit(normals)

        dummy_det = YOLO26Detector(_DummyModel(), device="cpu")
        sev_policy = SeverityPolicy()
        calib = QualityCalibration(
            blur=QualityBand(lo=0.0, hi=1000.0, name="blur"),
            luminance=QualityBand(lo=20.0, hi=240.0, name="luminance"),
            saturation=QualityBand(lo=0.0, hi=255.0, name="saturation"),
        )
        pipe_cfg = PipelineConfig(threshold_review=0.2, threshold_fail=0.5)
        pipeline = VisionPipeline(gate, dummy_det, sev_policy, calib, pipe_cfg)

        station = InspectionStation(
            vision_pipeline=pipeline,
            advisory_worker=None,  # Vision-only station test
            database=db,
            config=StationConfig(),
        )

        # Inspect a normal image
        normal_img = Image.fromarray(np.full((128, 128, 3), 100, dtype=np.uint8))
        rec = station.inspect_part(normal_img, part_id="PART-001")
        assert rec.part_id == "PART-001"
        assert rec.verdict == "PASS"

        # Verify DB logged it
        ok, n, err = db.verify_integrity()
        assert ok is True
        assert n == 1

        station.shutdown()

