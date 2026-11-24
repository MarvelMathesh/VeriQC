from __future__ import annotations

import time

import numpy as np
import pytest
from PIL import Image

from veriqc.station.sources import CameraSource, FolderSource, VideoSource, build_source
from veriqc.station.capture import CaptureService


def _make_images(folder, count=3) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        Image.fromarray(np.full((48, 64, 3), index * 20, dtype=np.uint8)).save(
            folder / f"{index:02d}.png"
        )


def test_camera_config_propagates_settings() -> None:
    source = build_source(
        {
            "type": "camera",
            "camera": {
                "kind": "usb",
                "index": 2,
                "width": 640,
                "height": 480,
                "fps": 15,
            },
        }
    )
    assert isinstance(source, CameraSource)
    assert source.index == 2
    assert (source.width, source.height, source.fps) == (640, 480, 15)


def test_folder_source_is_deterministic_and_loops(tmp_path) -> None:
    _make_images(tmp_path)
    source = FolderSource(tmp_path, loop=True, interval_s=0, shuffle=False)
    iterator = iter(source)
    frames = [next(iterator) for _ in range(5)]
    assert [frame.image.size for frame in frames] == [(64, 48)] * 5
    assert frames[0].frame_index == 0
    assert frames[3].frame_index == 3


def test_capture_service_keeps_latest_frame(tmp_path) -> None:
    _make_images(tmp_path, count=2)
    service = CaptureService(
        {"type": "folder", "path": str(tmp_path), "loop": True, "interval_s": 0.02}
    )
    service.start()
    try:
        deadline = time.time() + 2.0
        while service.latest() is None and time.time() < deadline:
            time.sleep(0.01)
        frame = service.latest()
        assert frame is not None
        assert service.status()["running"] is True
        assert service.status()["frames_captured"] >= 1
    finally:
        service.stop()
    assert service.status()["running"] is False


def test_video_source_reads_and_loops(tmp_path) -> None:
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48)
    )
    assert writer.isOpened()
    for index in range(3):
        frame = np.full((48, 64, 3), index * 40, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    source = VideoSource(path, loop=True, interval_s=0)
    iterator = iter(source)
    frames = [next(iterator) for _ in range(5)]
    assert len(frames) == 5
    assert frames[0].image.size == (64, 48)
    assert frames[3].frame_index == 3
    source.close()
