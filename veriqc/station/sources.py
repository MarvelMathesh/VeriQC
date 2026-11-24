"""Frame acquisition for bench, conveyor and replay inspection.

A source is deliberately small and blocking: it yields frames and does no
inference or quality work.  The station's capture thread is the only consumer
and stores the newest frame in a single-slot buffer, so a slow inspection can
never make the camera queue grow.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
from PIL import Image

__all__ = ["Frame", "FolderSource", "VideoSource", "CameraSource", "build_source"]

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Frame:
    image: Image.Image
    source: str
    frame_index: int
    captured_at: float
    meta: dict[str, Any] = field(default_factory=dict)


class _BaseSource:
    def close(self) -> None:
        pass


class FolderSource(_BaseSource):
    def __init__(
        self,
        folder: str | Path,
        *,
        loop: bool = True,
        shuffle: bool = False,
        interval_s: float = 0.5,
        seed: int | None = None,
    ) -> None:
        self.folder = Path(folder)
        self.files = sorted(
            path for path in self.folder.rglob("*") if path.suffix.lower() in _IMAGE_SUFFIXES
        )
        if not self.files:
            raise FileNotFoundError(f"no images found under {self.folder}")
        self.loop = bool(loop)
        self.shuffle = bool(shuffle)
        self.interval_s = max(0.0, float(interval_s))
        self.rng = random.Random(seed)
        self._index = 0

    def __iter__(self) -> Iterator[Frame]:
        while True:
            order = list(self.files)
            if self.shuffle:
                self.rng.shuffle(order)
            for path in order:
                with Image.open(path) as opened:
                    image = opened.convert("RGB")
                yield Frame(
                    image=image,
                    source=f"folder:{self.folder}",
                    frame_index=self._index,
                    captured_at=time.time(),
                    meta={"path": str(path)},
                )
                self._index += 1
                if self.interval_s:
                    time.sleep(self.interval_s)
            if not self.loop:
                return


class VideoSource(_BaseSource):
    def __init__(self, path: str | Path, *, loop: bool = True, interval_s: float = 0.0) -> None:
        self.path = str(path)
        self.loop = bool(loop)
        self.interval_s = max(0.0, float(interval_s))
        self._cap: cv2.VideoCapture | None = None
        self._index = 0

    def _open(self) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(self.path)
        if not capture.isOpened():
            raise RuntimeError(f"cannot open video: {self.path}")
        return capture

    def __iter__(self) -> Iterator[Frame]:
        while True:
            capture = self._open()
            self._cap = capture
            try:
                while True:
                    ok, bgr = capture.read()
                    if not ok:
                        break
                    image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                    yield Frame(
                        image=image,
                        source=f"video:{self.path}",
                        frame_index=self._index,
                        captured_at=time.time(),
                        meta={"path": self.path},
                    )
                    self._index += 1
                    if self.interval_s:
                        time.sleep(self.interval_s)
            finally:
                capture.release()
                self._cap = None
            if not self.loop:
                return

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class CameraSource(_BaseSource):
    def __init__(
        self,
        kind: str = "usb",
        index: int = 0,
        *,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ) -> None:
        self.kind = kind.lower()
        self.index = int(index)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self._cap: cv2.VideoCapture | None = None
        self._picam = None
        self._index = 0

    def _open_usb(self) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not capture.isOpened():
            raise RuntimeError(f"cannot open USB camera index {self.index}")
        return capture

    def _open_csi(self):
        from picamera2 import Picamera2

        camera = Picamera2(self.index)
        config = camera.create_preview_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"}
        )
        camera.configure(config)
        camera.start()
        return camera

    def __iter__(self) -> Iterator[Frame]:
        if self.kind == "csi":
            self._picam = self._open_csi()
            try:
                while True:
                    array = self._picam.capture_array("main")
                    image = Image.fromarray(np.asarray(array, dtype=np.uint8)).convert("RGB")
                    yield Frame(
                        image=image,
                        source=f"camera:csi:{self.index}",
                        frame_index=self._index,
                        captured_at=time.time(),
                    )
                    self._index += 1
            finally:
                self._picam.stop()
                self._picam.close()
                self._picam = None
            return

        capture = self._open_usb()
        self._cap = capture
        try:
            while True:
                ok, bgr = capture.read()
                if not ok:
                    raise RuntimeError("camera read failed")
                image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                yield Frame(
                    image=image,
                    source=f"camera:usb:{self.index}",
                    frame_index=self._index,
                    captured_at=time.time(),
                )
                self._index += 1
        finally:
            capture.release()
            self._cap = None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._picam is not None:
            try:
                self._picam.stop()
                self._picam.close()
            finally:
                self._picam = None


def build_source(config: dict[str, Any] | None):
    if not config:
        return None
    kind = str(config.get("type", "folder")).lower()
    if kind in ("none", "off"):
        return None
    if kind == "folder":
        return FolderSource(
            config["path"],
            loop=bool(config.get("loop", True)),
            shuffle=bool(config.get("shuffle", False)),
            interval_s=float(config.get("interval_s", 0.5)),
            seed=config.get("seed"),
        )
    if kind == "video":
        return VideoSource(
            config["path"],
            loop=bool(config.get("loop", True)),
            interval_s=float(config.get("interval_s", 0.0)),
        )
    if kind in ("camera", "usb", "csi"):
        camera = config.get("camera") or {}
        return CameraSource(
            kind="csi" if kind == "csi" else str(camera.get("kind", "usb")),
            index=int(camera.get("index", config.get("index", 0))),
            width=int(camera.get("width", 1280)),
            height=int(camera.get("height", 720)),
            fps=int(camera.get("fps", 30)),
        )
    raise ValueError(f"unsupported source type: {kind!r}")
