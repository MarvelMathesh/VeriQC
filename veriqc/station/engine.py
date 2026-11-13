"""Decoupled Inspection Station Engine.

Coordinates the two runtimes:
1. Fast Vision Pipeline (QC -> Gate -> Detector -> Triage -> Severity):
   Runs synchronously on the inspection path; returns verdict immediately to operator.
2. Asynchronous Advisory Worker (Llama 3.2 1B ExecuTorch):
   Runs in a background worker off the critical path; consumes DefectRecords,
   generates verified advisories, and logs tamper-evident entries into SQLite.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image

from veriqc.advisory.worker import AdvisoryOutcome, AdvisoryWorker
from veriqc.contract.record import DefectRecord
from veriqc.storage.db import AuditEntry, InspectionDatabase
from veriqc.vision.pipeline import VisionPipeline

__all__ = ["InspectionStation", "StationConfig"]


@dataclass
class StationConfig:
    max_advisory_queue: int = 100
    generate_advisory_for_pass: bool = False  # Only generate advisory for non-PASS parts


class InspectionStation:
    """The central edge inspection station coordinator."""

    def __init__(
        self,
        vision_pipeline: VisionPipeline,
        advisory_worker: AdvisoryWorker | None,
        database: InspectionDatabase,
        config: StationConfig | None = None,
        on_advisory_ready: Callable[[DefectRecord, AdvisoryOutcome], None] | None = None,
    ):
        self.vision = vision_pipeline
        self.advisor = advisory_worker
        self.db = database
        self.config = config or StationConfig()
        self.on_advisory_ready = on_advisory_ready

        self._advisory_queue: queue.Queue[DefectRecord] = queue.Queue(maxsize=self.config.max_advisory_queue)
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None

        if self.advisor is not None:
            self._start_advisory_worker()

    def _start_advisory_worker(self) -> None:
        def _worker_loop():
            while not self._stop_event.is_set():
                try:
                    record = self._advisory_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                try:
                    outcome = self.advisor.advise(record)
                    self.db.log_inspection(record, outcome)
                    if self.on_advisory_ready:
                        self.on_advisory_ready(record, outcome)
                except Exception as e:
                    # Fallback log in case of unexpected failure
                    print(f"[Station] Error generating advisory for {record.part_id}: {e}")
                finally:
                    self._advisory_queue.task_done()

        self._worker_thread = threading.Thread(target=_worker_loop, daemon=True, name="VeriQC-Advisor")
        self._worker_thread.start()

    def inspect_part(
        self,
        image: str | Path | Image.Image | np.ndarray,
        part_id: str,
    ) -> DefectRecord:
        """Synchronously inspect a part and asynchronously queue advisory if needed."""
        # 1. Immediate vision verdict
        record = self.vision.inspect(image, part_id=part_id)

        # 2. Queue for advisory if non-PASS (or configured for PASS)
        needs_advice = (record.verdict != "PASS") or self.config.generate_advisory_for_pass
        if self.advisor is not None and needs_advice:
            try:
                self._advisory_queue.put_nowait(record)
            except queue.Full:
                print(f"[Station] Advisory queue full, dropping background advice for {part_id}")
                self.db.log_inspection(record, None)
        else:
            self.db.log_inspection(record, None)

        return record

    def shutdown(self, wait_for_queue: bool = True) -> None:
        """Gracefully shut down station workers."""
        if wait_for_queue and self.advisor is not None:
            self._advisory_queue.join()
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)

