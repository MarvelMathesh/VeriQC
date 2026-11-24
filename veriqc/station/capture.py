"""Decoupled camera capture with a single latest-frame slot."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

from veriqc.station.sources import Frame, build_source

__all__ = ["LatestFrameBuffer", "CaptureService"]


class LatestFrameBuffer:
    def __init__(self) -> None:
        self._frames: deque[Frame] = deque(maxlen=1)
        self._condition = threading.Condition()

    def put(self, frame: Frame) -> None:
        with self._condition:
            self._frames.append(frame)
            self._condition.notify_all()

    def latest(self) -> Frame | None:
        with self._condition:
            return self._frames[-1] if self._frames else None

    def clear(self) -> None:
        with self._condition:
            self._frames.clear()


class CaptureService:
    def __init__(self, source_config: dict[str, Any]) -> None:
        self.source_config = dict(source_config)
        self.source = build_source(self.source_config)
        if self.source is None:
            raise ValueError("capture requires an enabled source")
        self.buffer = LatestFrameBuffer()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = 0.0
        self._frames_captured = 0
        self._error: str | None = None
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def error(self) -> str | None:
        return self._error

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._error = None
        self._started_at = time.time()
        self._thread = threading.Thread(target=self._run, name="VeriQC-Capture", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            for frame in self.source:
                if self._stop.is_set():
                    break
                self.buffer.put(frame)
                with self._lock:
                    self._frames_captured += 1
        except Exception as exc:
            self._error = repr(exc)
        finally:
            try:
                self.source.close()
            except Exception:
                pass

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self._thread = None
        try:
            self.source.close()
        except Exception:
            pass
        self.buffer.clear()

    def latest(self) -> Frame | None:
        return self.buffer.latest()

    def status(self) -> dict[str, Any]:
        with self._lock:
            frames = self._frames_captured
        elapsed = max(1e-6, time.time() - self._started_at) if self._started_at else 0.0
        return {
            "running": self.running,
            "source": self.source_config,
            "frames_captured": frames,
            "fps": round(frames / elapsed, 2) if elapsed else 0.0,
            "error": self._error,
        }
