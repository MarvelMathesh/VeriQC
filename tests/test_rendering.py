from __future__ import annotations

from pathlib import Path

from PIL import Image

from veriqc.inspection import InspectionResult
from veriqc.rendering import save_overlay


def _result(inspection_id: str) -> InspectionResult:
    return InspectionResult(
        inspection_id=inspection_id,
        timestamp=1.0,
        part_id="P",
        profile_id="p",
        verdict="FAIL",
    )


def test_overlay_retention_bounds_the_runtime_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VERIQC_OVERLAY_KEEP", "2")
    image = Image.new("RGB", (64, 64), (128, 128, 128))
    for index in range(4):
        overlay = save_overlay(
            image,
            _result(f"i{index}"),
            None,
            None,
            overlay_root=tmp_path,
            known_threshold=0.5,
        )
        assert overlay is not None
    files = sorted(Path(tmp_path / "overlays").glob("*.png"))
    assert len(files) == 2
