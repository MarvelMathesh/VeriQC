"""Station engine: vision on the inspection path, advisory off it."""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from veriqc import REPO_ROOT, sha256_file
from veriqc.advisory.process_runner import AdvisorRuntimeSpec, ProcessAdvisorRunner
from veriqc.advisory.worker import AdvisoryOutcome, AdvisoryWorker
from veriqc.config import load_profile
from veriqc.inspection import InspectionResult
from veriqc.pipeline import Pipeline
from veriqc.audit import AuditStore

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
    return {"advisory_model": model_hash, "advisory_tokenizer": tokenizer_hash}


class Station:
    def __init__(
        self,
        profile_id: str,
        *,
        device: str = "cpu",
        enable_advisory: bool = True,
        on_advisory: Callable[[InspectionResult, AdvisoryOutcome], None] | None = None,
        audit_path: Path | None = None,
    ) -> None:
        self.profile_id = profile_id
        self.profile = load_profile(REPO_ROOT / "configs" / "profiles" / f"{profile_id}.yaml")
        self.pipeline = Pipeline(self.profile, device=device)
        self.store = AuditStore(audit_path or REPO_ROOT / "veriqc_audit.db")
        self.on_advisory = on_advisory
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
                                "max_new_tokens": 48,
                            },
                        ),
                        tokenizer_path=tokenizer / "tokenizer.json",
                    )
                except Exception as exc:
                    self.runner_error = repr(exc)
            else:
                self.runner_error = "advisory artefacts are not present"
        self.worker = AdvisoryWorker(self.pipeline.cards, self.runner)
        self._queue: queue.Queue[InspectionResult] = queue.Queue(maxsize=64)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="VeriQC-Advisory")
        if self.runner is not None:
            self._thread.start()
        self._stats = {
            "inspections": 0,
            "advisories_completed": 0,
            "advisories_failed": 0,
        }
        self._stats_lock = threading.Lock()

    def inspect(self, image: Image.Image, part_id: str) -> InspectionResult:
        result = self.pipeline.inspect(image, part_id)
        if self.advisory_hashes:
            result.artifact_hashes.update(self.advisory_hashes)
        template, _card, _cause, _action = self.worker.template_for(result)
        if result.verdict == "PASS":
            result.advisory = {
                "status": "none",
                "text": "Part accepted. Continue the line.",
                "template": "Part accepted. Continue the line.",
            }
        else:
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
            try:
                self._queue.put_nowait(InspectionResult.from_dict(result.to_dict()))
            except queue.Full:
                result.advisory["status"] = "template"
                result.advisory["note"] = "advisory queue full"
        return result

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
                if self.on_advisory is not None:
                    self.on_advisory(result, outcome)
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
            "device": self.pipeline.device,
            "advisory": {
                "available": self.runner is not None,
                "error": self.runner_error,
                "queue": self._queue.qsize(),
                "model": "Llama-3.2-1B-Instruct / ExecuTorch",
            },
            "stats": stats,
            "chain": self.store.verify(),
        }

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self.runner is not None:
            self.runner.close()
