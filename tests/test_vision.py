"""Unit tests for the production vision pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest
import numpy as np
import torch
from PIL import Image

from veriqc.contract.severity import SeverityPolicy
from veriqc.vision.backbone import PatchEmbedder
from veriqc.vision.gate import AnomalyGate, GateConfig
from veriqc.vision.detector import YOLO26Detector
from veriqc.vision.pipeline import VisionPipeline, PipelineConfig


def test_patch_embedder_shapes():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = PatchEmbedder(backbone="efficientnet_b0", input_size=224, device=device)
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    x = embedder.preprocess([img])
    emb = embedder.embed(x)
    assert emb.ndim == 4
    assert emb.shape[0] == 1
    # Check L2 normalization
    norms = torch.linalg.norm(emb, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_anomaly_gate_fit_score_save_load():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = GateConfig(backbone="mobilenet_v3_small", input_size=128, coreset_ratio=0.2, device=device)
    gate = AnomalyGate(cfg)

    # Synthetic normal images
    normals = [Image.fromarray(np.full((128, 128, 3), 100 + i * 5, dtype=np.uint8)) for i in range(5)]
    gate.fit(normals)
    assert gate.is_fitted
    assert gate.memory_bank is not None
    assert gate.memory_bytes() > 0

    # Score normal vs anomaly
    test_normal = Image.fromarray(np.full((128, 128, 3), 110, dtype=np.uint8))
    test_anomaly = Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))

    score_res = gate.score([test_normal, test_anomaly])
    assert len(score_res.image_scores) == 2
    assert score_res.heatmaps.shape == (2, 128, 128)
    # Anomaly should have higher distance than normal
    assert score_res.image_scores[1] > score_res.image_scores[0]

    # Save and reload
    with tempfile.TemporaryDirectory() as tmpdir:
        gate_path = Path(tmpdir) / "gate.pt"
        gate.save(gate_path)
        loaded_gate = AnomalyGate.load(gate_path, device=device)
        assert loaded_gate.is_fitted
        res_loaded = loaded_gate.score([test_normal, test_anomaly])
        assert np.allclose(res_loaded.image_scores, score_res.image_scores, atol=1e-5)
