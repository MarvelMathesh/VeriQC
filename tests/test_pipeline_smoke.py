from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from veriqc.config import load_profile
from veriqc.pipeline import Pipeline

REPO = Path(__file__).resolve().parents[1]
STEEL_WEIGHTS = REPO / "models" / "profiles" / "steel_surface" / "yolo26n_640.pt"
CARPET_WEIGHTS = REPO / "models" / "profiles" / "textile_carpet" / "patchcore.pt"
STEEL_IMAGE = REPO / "data" / "neu" / "NEU-DET" / "validation" / "images" / "scratches" / "scratches_241.jpg"
CARPET_IMAGE = REPO / "data" / "mvtec" / "carpet" / "test" / "hole" / "003.png"


@pytest.mark.slow
@pytest.mark.skipif(
    not STEEL_WEIGHTS.exists() or not STEEL_IMAGE.exists(),
    reason="steel model artifact or sample image is not present",
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


@pytest.mark.slow
@pytest.mark.skipif(
    not CARPET_WEIGHTS.exists() or not CARPET_IMAGE.exists(),
    reason="carpet model artifact or sample image is not present",
)
def test_quality_gate_short_circuits_models(tmp_path) -> None:
    profile = load_profile(REPO / "configs" / "profiles" / "commutator.yaml")
    pipeline = Pipeline(profile, device="cpu", overlay_root=tmp_path)
    result = pipeline.inspect(Image.new("RGB", (32, 32), (128, 128, 128)), "GATE", overlay=False)
    assert result.verdict == "REVIEW"
    assert any("quality gate" in reason for reason in result.reasons)
    assert "known_ms" not in result.latency_ms


def test_carpet_pipeline_end_to_end(tmp_path) -> None:
    profile = load_profile(REPO / "configs" / "profiles" / "textile_carpet.yaml")
    pipeline = Pipeline(profile, device="cpu", overlay_root=tmp_path)
    image = Image.open(REPO / "data" / "mvtec" / "carpet" / "test" / "hole" / "003.png")
    result = pipeline.inspect(image, "CARPET-1")
    assert result.verdict in ("REVIEW", "FAIL")
    assert result.anomaly_p_value is not None
    assert result.findings
