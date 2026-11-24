"""Station service: vision on the inspection path, advisory off it."""

from __future__ import annotations

import hashlib
import io
import json
import os
import queue
import threading
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from veriqc import REPO_ROOT, sha256_file
from veriqc.advisory.constants import ADVISORY_DEADLINE_S, ADVISORY_MAX_NEW_TOKENS
from veriqc.advisory.process_runner import AdvisorRuntimeSpec, ProcessAdvisorRunner
from veriqc.advisory.worker import AdvisoryOutcome, AdvisoryWorker
from veriqc.config import load_profile
from veriqc.inspection import InspectionResult
from veriqc.pipeline import Pipeline
from veriqc.audit import AuditStore
from veriqc.station.capture import CaptureService

__all__ = ["Station"]


def _verify_advisory_artifacts(pte: Path, tokenizer: Path, manifest: Path) -> dict[str, str]:
    if not manifest.exists():
        return {}
    expected = json.loads(manifest.read_text())
    model_hash = sha256_file(pte)
    tokenizer_hash = sha256_file(tokenizer / "tokenizer.json")
    if expected.get("sha256") and model_hash != expected["sha256"]:
        raise ValueError("advisory PTE hash mismatch")
    if expected.get("tokenizer_sha256") and tokenizer_hash != expected["tokenizer_sha256"]:
        raise ValueError("advisory tokenizer hash mismatch")
    hashes = {"advisory_model": model_hash, "advisory_tokenizer": tokenizer_hash}
    tokenizer_files = expected.get("tokenizer_files") or {}
    if tokenizer_files:
        for name, digest in sorted(tokenizer_files.items()):
            path = tokenizer / name
            if not path.is_file():
                raise ValueError(f"advisory tokenizer file is missing: {name}")
            if sha256_file(path) != digest:
                raise ValueError(f"advisory tokenizer file hash mismatch: {name}")
        bundle = hashlib.sha256(
            json.dumps(tokenizer_files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if expected.get("tokenizer_bundle_sha256") and bundle != expected["tokenizer_bundle_sha256"]:
            raise ValueError("advisory tokenizer bundle hash mismatch")
        hashes["advisory_tokenizer_bundle"] = bundle
    return hashes


class Station:
    def __init__(
        self,
        profile_id: str,
        *,
        device: str = "cpu",
        enable_advisory: bool = True,
        on_advisory: Callable[[InspectionResult, AdvisoryOutcome], None] | None = None,
        audit_path: Path | None = None,
        source_config: dict[str, Any] | None = None,
    ) -> None:
        self.profile_id = profile_id
        self.profile = load_profile(REPO_ROOT / "configs" / "profiles" / f"{profile_id}.yaml")
        self.pipeline = Pipeline(self.profile, device=device)
        self.store = AuditStore(audit_path or REPO_ROOT / "veriqc_audit.db")
        self.on_advisory = on_advisory
        self.capture: CaptureService | None = None
        if source_config and str(source_config.get("type", "none")).lower() not in ("none", "off"):
            self.capture = CaptureService(source_config)
        self.runner = None
        self.runner_error: str | None = None
        self.advisory_hashes: dict[str, str] = {}
        if enable_advisory:
            pte = REPO_ROOT / "models" / "advisory" / "llama32_1b_int8.pte"
            tokenizer = REPO_ROOT / "models" / "advisory" / "tokenizer"
            manifest = REPO_ROOT / "models" / "advisory" / "manifest.json"
            if pte.exists() and tokenizer.exists():
                try:
                    self.advisory_hashes = _verify_advisory_artifacts(pte, tokenizer, manifest)
                    self.runner = ProcessAdvisorRunner(
                        AdvisorRuntimeSpec(
                            module="veriqc.advisory.executorch",
                            factory="ExecuTorchAdvisor",
                            kwargs={
                                "pte_path": str(pte),
                                "tokenizer_path": str(tokenizer / "tokenizer.json"),
                                "max_new_tokens": ADVISORY_MAX_NEW_TOKENS,
                            },
                        ),
                        tokenizer_path=tokenizer / "tokenizer.json",
                    )
                except Exception as exc:
                    self.runner_error = repr(exc)
            else:
                self.runner_error = "advisory artefacts are not present"
        self.worker = AdvisoryWorker(
            self.pipeline.cards,
            self.runner,
            max_new_tokens=ADVISORY_MAX_NEW_TOKENS,
            deadline_s=float(
                os.environ.get("VERIQC_ADVISORY_DEADLINE_S", str(ADVISORY_DEADLINE_S))
            ),
            context_provider=self._advisory_context,
        )
        self._queue: queue.Queue[InspectionResult] = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._stats = {
            "inspections": 0,
            "advisories_completed": 0,
            "advisories_failed": 0,
            "advisories_dropped": 0,
            "advisories_timed_out": 0,
            "advisory_queue_high_water": 0,
        }
        self._stats_lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="VeriQC-Advisory")
        if self.runner is not None:
            self._thread.start()

    def _advisory_context(self) -> dict[str, Any]:
        return self.store.recent_context(50)

    def queue_depth(self) -> int:
        return self._queue.qsize()

    def inspect(self, image: Image.Image, part_id: str) -> InspectionResult:
        result = self.pipeline.inspect(image, part_id)
        if self.advisory_hashes:
            result.artifact_hashes.update(self.advisory_hashes)
        if result.verdict == "PASS":
            result.advisory = {
                "status": "none",
                "text": "Part accepted. Continue the line.",
                "template": "Part accepted. Continue the line.",
            }
        else:
            template, _card, _cause, _action = self.worker.template_for(result)
            result.advisory = {
                "status": "pending" if self.runner is not None else "template",
                "text": template,
                "template": template,
                "accepted": False,
            }
        self.store.append(result, event="verdict")
        with self._stats_lock:
            self._stats["inspections"] += 1
        if result.verdict != "PASS" and self.runner is not None:
            self._enqueue_advisory(result)
        return result

    def _enqueue_advisory(self, result: InspectionResult) -> None:
        """Keep advice fresh: one pending item; replace a stale one explicitly."""

        queued = InspectionResult.from_dict(result.to_dict())
        while True:
            try:
                self._queue.put_nowait(queued)
                break
            except queue.Full:
                try:
                    stale = self._queue.get_nowait()
                    self._queue.task_done()
                except queue.Empty:
                    continue
                template = self.worker.template_for(stale)[0]
                stale.advisory = {
                    "status": "template",
                    "text": template,
                    "template": template,
                    "path": "template",
                    "accepted": False,
                    "verification": {
                        "ok": False,
                        "violations": ["replaced by a newer part"],
                    },
                    "note": "replaced by a newer part",
                }
                self.store.append(stale, event="advisory", extra=stale.advisory)
                with self._stats_lock:
                    self._stats["advisories_dropped"] += 1
        with self._stats_lock:
            self._stats["advisory_queue_high_water"] = max(
                self._stats["advisory_queue_high_water"],
                self.queue_depth(),
            )

    def start_capture(self, source_config: dict[str, Any]) -> dict[str, Any]:
        previous = self.capture
        self.capture = None
        if previous is not None:
            previous.stop()
        capture = CaptureService(source_config)
        capture.start()
        self.capture = capture
        return capture.status()

    def stop_capture(self) -> dict[str, Any] | None:
        capture = self.capture
        if capture is None:
            return None
        self.capture = None
        capture.stop()
        return {"running": False}

    def capture_status(self) -> dict[str, Any] | None:
        capture = self.capture
        return capture.status() if capture is not None else None

    def capture_inspect(self, part_id: str) -> InspectionResult:
        capture = self.capture
        if capture is None:
            raise RuntimeError("no capture source is active")
        frame = capture.latest()
        if frame is None:
            raise RuntimeError("no frame has been captured yet")
        return self.inspect(frame.image, part_id)

    def latest_jpeg(self, quality: int = 80) -> bytes | None:
        capture = self.capture
        if capture is None:
            return None
        frame = capture.latest()
        if frame is None:
            return None
        buffer = io.BytesIO()
        frame.image.save(buffer, format="JPEG", quality=max(40, min(95, int(quality))))
        return buffer.getvalue()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                result = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                outcome = self.worker.advise_result(result)
                result.advisory = outcome.to_dict()
                self.store.append(result, event="advisory", extra=outcome.to_dict())
                with self._stats_lock:
                    self._stats["advisories_completed"] += 1
                    if outcome.timed_out:
                        self._stats["advisories_timed_out"] += 1
                if self.on_advisory is not None:
                    try:
                        self.on_advisory(result, outcome)
                    except Exception:
                        # Delivery to the UI must not turn a completed advisory
                        # into an advisory failure.
                        pass
            except Exception as exc:
                with self._stats_lock:
                    self._stats["advisories_failed"] += 1
                template = self.worker.template_for(result)[0]
                result.advisory = {
                    "status": "template",
                    "text": template,
                    "template": template,
                    "accepted": False,
                    "error": repr(exc),
                }
                self.store.append(result, event="advisory_error", extra={"error": repr(exc)})
            finally:
                self._queue.task_done()

    def review(self, inspection_id: str, label: str, note: str = "") -> dict[str, Any] | None:
        result = self.store.get(inspection_id)
        if result is None:
            return None
        inspection = InspectionResult.from_dict(result)
        inspection.operator = {"label": label, "note": note}
        self.store.append(inspection, event="review", extra={"label": label, "note": note})
        return inspection.to_dict()

    def metrics(self) -> dict[str, Any]:
        with self._stats_lock:
            stats = dict(self._stats)
        return {
            "profile_id": self.profile_id,
            "domain": self.profile.domain,
            "task": self.profile.task,
            "backend": self.pipeline.backend,
            "device": self.pipeline.device,
            "advisory": {
                "available": self.runner is not None,
                "error": self.runner_error,
                "queue": self.queue_depth(),
                "model": "Llama-3.2-1B-Instruct / ExecuTorch",
            },
            "stats": stats,
            "capture": self.capture_status(),
            "chain": self.store.verify(),
        }

    def close(self) -> None:
        if self.capture is not None:
            self.capture.stop()
            self.capture = None
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self.runner is not None:
            self.runner.close()
