from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from veriqc.config import load_profile
from veriqc.models.onnx_runtime import ONNXDetector, ONNXPatchCore, ONNXSegmenter

REPO = Path(__file__).resolve().parents[1]
pytest.importorskip("onnxruntime")


def test_onnx_runtime_does_not_import_training_stack() -> None:
    code = (
        "import sys; import veriqc.models.onnx_runtime; "
        "assert 'torch' not in sys.modules, 'torch was imported'; "
        "assert 'ultralytics' not in sys.modules, 'ultralytics was imported'"
    )
    subprocess.run([sys.executable, "-c", code], cwd=REPO, check=True)


def test_patchcore_rejects_mismatched_bank_metadata(tmp_path) -> None:
    banks = tmp_path / "anomaly_banks.npz"
    np.savez(
        banks,
        image_bank=np.zeros((2, 4), dtype=np.float32),
        pixel_bank=np.zeros((2, 4), dtype=np.float32),
        image_layer="layer3",
        pixel_layer="layer2",
        input_size=448,
    )
    with pytest.raises(ValueError, match="bank layers"):
        ONNXPatchCore(tmp_path / "unused.onnx", banks, input_size=448)


def test_detector_onnx_matches_pytorch() -> None:
    profile = load_profile(REPO / "configs/profiles/steel_surface.yaml")
    model_path = REPO / "models/profiles/steel_surface/export/detector.onnx"
    if not model_path.exists():
        pytest.skip("detector ONNX export is not present")
    from veriqc.models.torch_detector import Detector

    reference = Detector.from_weights(
        profile.known.path,
        profile.known.classes,
        input_size=640,
        conf_threshold=0.01,
        device="cpu",
    )
    exported = ONNXDetector(
        model_path,
        profile.known.classes,
        input_size=640,
        conf_threshold=0.01,
    )
    manifest = json.loads((REPO / "data/prepared/neu/manifest.json").read_text())
    for item in manifest["items"]["test"][:3]:
        with Image.open(item["image"]) as opened:
            image = opened.convert("RGB")
        a, _ = reference.predict(image)
        b, _ = exported.predict(image)
        assert len(a) == len(b)
        if a:
            assert max(abs(x.confidence - y.confidence) for x, y in zip(a, b)) < 1e-4
            assert max(
                abs(u - v) for x, y in zip(a, b) for u, v in zip(x.bbox, y.bbox)
            ) < 0.01


def test_segmenter_onnx_matches_pytorch() -> None:
    profile = load_profile(REPO / "configs/profiles/commutator.yaml")
    model_path = REPO / "models/profiles/commutator/export/segmenter.onnx"
    if not model_path.exists():
        pytest.skip("segmenter ONNX export is not present")
    from veriqc.models.torch_segmenter import Segmenter

    reference = Segmenter.from_checkpoint(profile.known.path, device="cpu")
    exported = ONNXSegmenter(model_path, input_width=384, input_height=1024)
    manifest = json.loads((REPO / "data/prepared/kolektor/manifest.json").read_text())
    with Image.open(manifest["folds"][0]["test"][0]["image"]) as opened:
        image = opened.convert("RGB")
    a, _ = reference.predict_proba(image)
    b, _ = exported.predict_proba(image)
    assert a.shape == b.shape
    assert float(np.abs(a - b).max()) < 1e-4


def test_patchcore_onnx_matches_pytorch() -> None:
    profile = load_profile(REPO / "configs/profiles/textile_carpet.yaml")
    backbone = REPO / "models/profiles/textile_carpet/export/anomaly_backbone.onnx"
    banks = REPO / "models/profiles/textile_carpet/export/anomaly_banks.npz"
    if not backbone.exists() or not banks.exists():
        pytest.skip("anomaly ONNX export is not present")
    from veriqc.models.torch_patchcore import PatchCore

    reference = PatchCore.load(profile.anomaly.path, device="cpu")
    exported = ONNXPatchCore(backbone, banks, input_size=448)
    manifest = json.loads((REPO / "data/prepared/mvtec/manifest.json").read_text())
    item = next(
        item
        for item in manifest["categories"]["carpet"]["items"]
        if item["split"] == "test" and item["is_defect"]
    )
    with Image.open(item["image"]) as opened:
        image = opened.convert("RGB")
    a = reference.score(image)
    b = exported.score(image)
    assert abs(a.image_score - b.image_score) < 1e-4
    assert float(np.abs(a.heatmap - b.heatmap).max()) < 1e-3
