from __future__ import annotations

import time
from pathlib import Path

import pytest

from veriqc.station.service import Station

REPO = Path(__file__).resolve().parents[1]
KOLEKTOR = REPO / "data" / "kolektor" / "kos02" / "Part0.jpg"
SEGMENTER_ONNX = REPO / "models" / "profiles" / "commutator" / "export" / "segmenter.onnx"
pytestmark = pytest.mark.skipif(
    not KOLEKTOR.exists() or not SEGMENTER_ONNX.exists(),
    reason="Kolektor sample image or ONNX artifact is not present",
)


def test_station_inspects_and_audits(tmp_path) -> None:
    station = Station(
        "commutator",
        enable_advisory=False,
        audit_path=tmp_path / "audit.db",
    )
    try:
        from PIL import Image

        with Image.open(KOLEKTOR) as opened:
            result = station.inspect(opened.convert("RGB"), "PART-1")
        assert result.profile_id == "commutator"
        assert result.verdict in ("PASS", "REVIEW", "FAIL")
        assert station.store.verify()["ok"] is True
    finally:
        station.close()


def test_station_capture_and_inspect(tmp_path) -> None:
    station = Station(
        "commutator",
        enable_advisory=False,
        audit_path=tmp_path / "audit.db",
    )
    try:
        station.start_capture(
            {
                "type": "folder",
                "path": str(KOLEKTOR.parent),
                "loop": True,
                "interval_s": 0.02,
            }
        )
        deadline = time.time() + 3.0
        while station.capture is not None and station.capture.latest() is None:
            if time.time() > deadline:
                raise AssertionError("capture did not produce a frame")
            time.sleep(0.02)
        result = station.capture_inspect("PART-2")
        assert result.part_id == "PART-2"
        assert station.store.verify()["ok"] is True
    finally:
        station.close()
