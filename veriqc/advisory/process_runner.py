"""Process-isolated advisory runtime.

The vision station and the Llama/ExecuTorch runtime are separate CPU
pipelines.  This module keeps the native advisory runtime in a child process
so that a runtime crash, a native abort or an unexpectedly long generation
cannot take the inspection process with it.

The boundary is deliberately narrow:

* the parent builds the findings-only prompt;
* the child owns the model and performs generation only;
* the child returns untrusted model prose and runtime counters; and
* the parent verifies the prose and injects every identifier.

A multiprocessing pipe is used instead of a shell protocol because it gives
the parent a framed, private channel and a process handle that it can
terminate cleanly at shutdown.  The child is started with the ``spawn``
method so it does not inherit the vision process's torch, detector or bank
memory.
"""

from __future__ import annotations

import atexit
import importlib
import multiprocessing
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from veriqc.advisory.executorch import AdvisorResult

__all__ = [
    "AdvisorRuntimeSpec",
    "ProcessAdvisorError",
    "ProcessAdvisorRunner",
]


class ProcessAdvisorError(RuntimeError):
    """Raised when the isolated advisory runtime cannot be established."""


@dataclass(frozen=True)
class AdvisorRuntimeSpec:
    """A picklable description of the runner constructed in the child.

    ``module`` and ``factory`` identify a class by import path.  Keeping the
    specification declarative lets the child start under the ``spawn`` method
    without pickling an already-loaded model or a closure.
    """

    module: str
    factory: str
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    def create(self) -> Any:
        module = importlib.import_module(self.module)
        try:
            factory = getattr(module, self.factory)
        except AttributeError as exc:
            raise ProcessAdvisorError(
                f"the advisory runtime factory {self.module}:{self.factory} "
                "does not exist"
            ) from exc
        return factory(**dict(self.kwargs))


def _load_parent_tokenizer(tokenizer_path: str | Path | None) -> Any:
    """Load the chat tokenizer in the station process.

    Only the small tokenizer is shared.  The model and its native runtime
    stay in the child.  The tokenizer directory is profile-bound and is never
    replaced with another profile's tokenizer.
    """

    if tokenizer_path is None:
        return None
    path = Path(tokenizer_path)
    directory = path.parent if path.is_file() else path
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(str(directory))
    except Exception as exc:
        raise ProcessAdvisorError(
            f"the advisory chat tokenizer could not be loaded from {directory}: "
            f"{exc!r}"
        ) from exc


def _runtime_main(
    connection: Any,
    spec: AdvisorRuntimeSpec,
) -> None:
    """Own the model in the child and serve generation requests."""

    advisor = None
    try:
        advisor = spec.create()
        ready = {
            "pid": multiprocessing.current_process().pid,
            "path": str(getattr(advisor, "path", getattr(advisor, "pte_path", ""))),
            "load_seconds": float(getattr(advisor, "load_seconds", 0.0)),
            "resident_bytes": int(getattr(advisor, "resident_bytes", 0) or 0),
        }
        connection.send(("ready", ready))
    except BaseException as exc:
        try:
            connection.send(
                (
                    "ready_error",
                    {
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
            )
        except Exception:
            pass
        try:
            connection.close()
        except Exception:
            pass
        return

    try:
        while True:
            try:
                message = connection.recv()
            except EOFError:
                break
            if not isinstance(message, tuple) or not message:
                continue
            kind = message[0]
            if kind == "stop":
                break
            if kind != "generate" or len(message) != 4:
                connection.send(
                    (
                        "protocol_error",
                        str(message[0]) if message else "",
                        "invalid advisory runtime request",
                    )
                )
                continue
            _, request_id, prompt, max_new_tokens = message
            try:
                result = advisor.generate(
                    str(prompt),
                    max_new_tokens=int(max_new_tokens),
                )
                if not isinstance(result, AdvisorResult):
                    result = AdvisorResult(
                        text=str(getattr(result, "text", "")),
                        ok=bool(getattr(result, "ok", False)),
                        path=str(getattr(result, "path", "llm")),
                        prompt_tokens=int(getattr(result, "prompt_tokens", 0)),
                        generated_tokens=int(getattr(result, "generated_tokens", 0)),
                        ttft_ms=float(getattr(result, "ttft_ms", 0.0)),
                        inference_ms=float(getattr(result, "inference_ms", 0.0)),
                        decode_ms=float(getattr(result, "decode_ms", 0.0)),
                        tokens_per_second=float(
                            getattr(result, "tokens_per_second", 0.0)
                        ),
                        prefill_tokens_per_second=float(
                            getattr(result, "prefill_tokens_per_second", 0.0)
                        ),
                        resident_mb=float(getattr(result, "resident_mb", 0.0)),
                        error=getattr(result, "error", None),
                    )
            except BaseException as exc:
                result = AdvisorResult(
                    text="",
                    ok=False,
                    path="llm",
                    error=repr(exc),
                )
            try:
                connection.send(("result", request_id, result))
            except (BrokenPipeError, EOFError, OSError):
                break
    finally:
        closer = getattr(advisor, "close", None)
        if closer is not None:
            try:
                closer()
            except Exception:
                pass
        try:
            connection.close()
        except Exception:
            pass


class ProcessAdvisorRunner:
    """Parent-side proxy for a restartable advisory child process."""

    def __init__(
        self,
        spec: AdvisorRuntimeSpec,
        *,
        tokenizer_path: str | Path | None = None,
        start_timeout_s: float = 300.0,
    ):
        self.spec = spec
        self.tokenizer_path = str(tokenizer_path) if tokenizer_path is not None else None
        self.start_timeout_s = float(start_timeout_s)
        self.tokenizer = _load_parent_tokenizer(tokenizer_path)
        self.path = str(spec.kwargs.get("pte_path", ""))
        self.load_seconds = 0.0
        self.resident_bytes = 0
        self.process_id: int | None = None
        self._closed = True
        self._connection: Any | None = None
        self._process: Any | None = None
        self._request_lock = threading.Lock()
        self._lifecycle_lock = threading.RLock()

        self.start()
        atexit.register(self.close)

    @property
    def is_alive(self) -> bool:
        process = getattr(self, "_process", None)
        return bool(process is not None and process.is_alive())

    def start(self) -> None:
        """Start or replace the child runtime.

        ``close`` is a lifecycle boundary, not a permanent destruction of the
        proxy.  A FastAPI application can stop and start again in the same
        process, so the next start has to be able to load a fresh child rather
        than leave the station pointing at a closed pipe.
        """

        with self._lifecycle_lock:
            if self.is_alive:
                self._closed = False
                return
            self._closed = False
            self._spawn()

    def _spawn(self) -> None:
        context = multiprocessing.get_context("spawn")
        parent_connection, child_connection = context.Pipe(duplex=True)
        process = context.Process(
            target=_runtime_main,
            args=(child_connection, self.spec),
            name="VeriQC-Advisory-Runtime",
            daemon=False,
        )
        try:
            process.start()
        except Exception as exc:
            parent_connection.close()
            child_connection.close()
            raise ProcessAdvisorError(
                f"the isolated advisory runtime could not start: {exc!r}"
            ) from exc
        child_connection.close()

        if not parent_connection.poll(self.start_timeout_s):
            self._terminate_process(process)
            parent_connection.close()
            raise ProcessAdvisorError(
                "the isolated advisory runtime did not become ready within "
                f"{self.start_timeout_s:.0f}s"
            )
        try:
            kind, payload = parent_connection.recv()
        except EOFError as exc:
            self._terminate_process(process)
            parent_connection.close()
            raise ProcessAdvisorError(
                "the isolated advisory runtime exited before it was ready"
            ) from exc
        if kind == "ready_error":
            self._terminate_process(process)
            parent_connection.close()
            detail = payload.get("error", "unknown error")
            raise ProcessAdvisorError(
                f"the isolated advisory runtime failed to load: {detail}"
            )
        if kind != "ready" or not isinstance(payload, Mapping):
            self._terminate_process(process)
            parent_connection.close()
            raise ProcessAdvisorError(
                "the isolated advisory runtime returned an invalid ready message"
            )

        self._connection = parent_connection
        self._process = process
        self.process_id = int(payload.get("pid") or process.pid or 0)
        self.path = str(payload.get("path") or self.path)
        self.load_seconds = float(payload.get("load_seconds") or 0.0)
        self.resident_bytes = int(payload.get("resident_bytes") or 0)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
    ) -> AdvisorResult:
        """Send one prompt to the child and wait for its untrusted prose."""

        if self._closed:
            raise ProcessAdvisorError("the isolated advisory runtime is closed")
        if not self.is_alive:
            raise ProcessAdvisorError("the isolated advisory runtime is not alive")
        if not self._request_lock.acquire(blocking=False):
            raise ProcessAdvisorError(
                "the isolated advisory runtime already has a request in flight"
            )
        request_id = uuid.uuid4().hex
        connection = self._connection
        try:
            if connection is None:
                raise ProcessAdvisorError("the isolated advisory runtime has no pipe")
            connection.send(
                (
                    "generate",
                    request_id,
                    str(prompt),
                    int(max_new_tokens or 0),
                )
            )
            while True:
                try:
                    message = connection.recv()
                except EOFError as exc:
                    raise ProcessAdvisorError(
                        "the isolated advisory runtime exited during generation"
                    ) from exc
                if not isinstance(message, tuple) or not message:
                    continue
                if message[0] == "result" and len(message) == 3:
                    if message[1] != request_id:
                        continue
                    result = message[2]
                    if isinstance(result, AdvisorResult):
                        return result
                    raise ProcessAdvisorError(
                        "the isolated advisory runtime returned an invalid result"
                    )
                if message[0] == "protocol_error":
                    raise ProcessAdvisorError(str(message[2]))
        finally:
            self._request_lock.release()

    def close(self, timeout: float = 5.0) -> None:
        """Stop the child, terminating it if a generation will not yield."""

        with self._lifecycle_lock:
            process = getattr(self, "_process", None)
            connection = getattr(self, "_connection", None)
            self._closed = True
            if process is not None and process.is_alive():
                try:
                    if connection is not None:
                        connection.send(("stop",))
                except (BrokenPipeError, EOFError, OSError):
                    pass
                process.join(timeout=max(0.0, float(timeout)))
            self._terminate_process(process)
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass
            self._connection = None
            self._process = None
            self.process_id = None

    @staticmethod
    def _terminate_process(process: Any | None) -> None:
        if process is None:
            return
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
        if process.is_alive():
            try:
                process.kill()
            except (AttributeError, OSError):
                pass
            process.join(timeout=2.0)
