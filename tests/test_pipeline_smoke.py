from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from veriqc.config import load_profile
from veriqc.pipeline import Pipeline

REPO = Path(__file__).resolve().parents[1]
STEEL_ONNX = REPO / "models" / "profiles" / "steel_surface" / "export" / "detector.onnx"
CARPET_ONNX = REPO / "models" / "profiles" / "textile_carpet" / "export" / "anomaly_backbone.onnx"
CARPET_BANKS = REPO / "models" / "profiles" / "textile_carpet" / "export" / "anomaly_banks.npz"
SEGMENTER_ONNX = REPO / "models" / "profiles" / "commutator" / "export" / "segmenter.onnx"
STEEL_IMAGE = REPO / "data" / "neu" / "NEU-DET" / "validation" / "images" / "scratches" / "scratches_241.jpg"
CARPET_IMAGE = REPO / "data" / "mvtec" / "carpet" / "test" / "hole" / "003.png"


@pytest.mark.slow
@pytest.mark.skipif(
    not STEEL_ONNX.exists() or not STEEL_IMAGE.exists(),
    reason="steel ONNX artifact or sample image is not present",
)
def test_steel_pipeline_end_to_end(tmp_path) -> None:
    profile = load_profile(REPO / "configs" / "profiles" / "steel_surface.yaml")
    pipeline = Pipeline(profile, device="cpu", overlay_root=tmp_path)
    image = Image.open(REPO / "data" / "neu" / "NEU-DET" / "validation" / "images" / "scratches" / "scratches_241.jpg")
    result = pipeline.inspect(image, "STEEL-1")
    assert result.verdict in ("PASS", "REVIEW", "FAIL")
    assert result.profile_id == "steel_surface"
    assert result.image_sha256
    assert result.overlay
    assert Path(result.overlay).exists()


@pytest.mark.skipif(
    not SEGMENTER_ONNX.exists(),
    reason="commutator ONNX artifact is not present",
)
def test_quality_gate_short_circuits_models(tmp_path) -> None:
    profile = load_profile(REPO / "configs" / "profiles" / "commutator.yaml")
    pipeline = Pipeline(profile, device="cpu", overlay_root=tmp_path)
    result = pipeline.inspect(Image.new("RGB", (32, 32), (128, 128, 128)), "GATE", overlay=False)
    assert result.verdict == "REVIEW"
    assert any("quality gate" in reason for reason in result.reasons)
    assert "known_ms" not in result.latency_ms


@pytest.mark.skipif(
    not CARPET_ONNX.exists() or not CARPET_BANKS.exists() or not CARPET_IMAGE.exists(),
    reason="carpet ONNX artifact or sample image is not present",
)
def test_carpet_pipeline_end_to_end(tmp_path) -> None:
    profile = load_profile(REPO / "configs" / "profiles" / "textile_carpet.yaml")
    pipeline = Pipeline(profile, device="cpu", overlay_root=tmp_path)
    image = Image.open(REPO / "data" / "mvtec" / "carpet" / "test" / "hole" / "003.png")
    result = pipeline.inspect(image, "CARPET-1")
    assert result.verdict in ("REVIEW", "FAIL")
    assert result.anomaly_p_value is not None
    assert result.findings
