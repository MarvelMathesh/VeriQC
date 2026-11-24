"""The ExecuTorch advisor runtime.

One long-lived ``TextLLMRunner`` object; the model is loaded once (the KV
cache for a 1B model at a 1-2k context is allocated in full at load time, so
reloading per request would dominate the latency budget).

The runner gives us exactly the numbers the report needs: prompt tokens,
time to first token, generated tokens and the inference window, so TTFT and
decode rate are measured rather than estimated.
"""

from __future__ import annotations

import atexit
import json
import re
import resource
import sys
import time
import weakref
from dataclasses import dataclass, field
from pathlib import Path

from veriqc.advisory.constants import ADVISORY_MAX_NEW_TOKENS

__all__ = ["AdvisorResult", "ExecuTorchAdvisor", "ADVISORY_TOKENIZER_DIR"]

#: Where the advisory's tokenizer lives.  The station ships the tokenizer
#: next to the artefact it belongs to, because the two are a pair: the
#: artefact was exported for that vocabulary and that chat template, and a
#: tokenizer from anywhere else would change the prompt the model sees.
ADVISORY_TOKENIZER_DIR = (
    Path(__file__).resolve().parents[2] / "models" / "advisory" / "tokenizer"
)

#: The tokenizers file the ExecuTorch runtime reads.  It is the same
#: vocabulary the HF tokenizer is built from, in the format the native
#: runtime expects.
ADVISORY_TOKENIZER_FILE = ADVISORY_TOKENIZER_DIR / "tokenizer.json"


def _peak_rss_bytes() -> int:
    """The peak resident set size of this process, in bytes.

    On Linux ``ru_maxrss`` is in kilobytes.  The advisory artefact is the
    largest thing the station holds, so the peak after loading it is the
    number the deployment budget is built from -- and it is the number that
    decides whether the advisory fits on a Raspberry Pi 5 at all.
    """
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024

#: Every advisor that is currently loaded. The native runtime has to be
#: released while the interpreter is still alive: a runner that is
#: destroyed during interpreter shutdown aborts the process, because its
#: worker thread is still joinable.
_LIVE_ADVISORS: "weakref.WeakSet[ExecuTorchAdvisor]" = weakref.WeakSet()


@dataclass
class AdvisorResult:
    text: str
    ok: bool
    path: str = "llm"  # llm | template | failed
    prompt_tokens: int = 0
    generated_tokens: int = 0
    ttft_ms: float = 0.0
    inference_ms: float = 0.0
    decode_ms: float = 0.0
    tokens_per_second: float = 0.0
    #: The prefill rate, as the runtime reports it.  Prefill and decode
    #: are different stages with very different speeds, so the advisory's
    #: latency budget is built from both.
    prefill_tokens_per_second: float = 0.0
    resident_mb: float = 0.0
    error: str | None = None
    raw_stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "ok": self.ok,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "ttft_ms": round(self.ttft_ms, 1),
            "inference_ms": round(self.inference_ms, 1),
            "decode_ms": round(self.decode_ms, 1),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "prefill_tokens_per_second": round(self.prefill_tokens_per_second, 2),
            "resident_mb": round(self.resident_mb, 1),
            "error": self.error,
        }


class ExecuTorchAdvisor:
    """Thin wrapper around ExecuTorch's LLM runner."""

    def __init__(
        self,
        pte_path: str | Path,
        tokenizer_path: str | Path | None = None,
        max_new_tokens: int = ADVISORY_MAX_NEW_TOKENS,
        temperature: float = 0.0,
        use_chat_template: bool = True,
    ):
        """Load the artefact and the tokenizer.

        The ExecuTorch runner uses four worker threads and the Python API
        does not expose the setting. Four is also the Raspberry Pi 5's core
        count, so the measured latency is taken on the right amount of
        silicon.
        """
        from executorch.extension.llm.runner import TextLLMRunner

        self.pte_path = str(pte_path)
        self.path = self.pte_path
        self.tokenizer_path = str(tokenizer_path or ADVISORY_TOKENIZER_FILE)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.load_seconds = 0.0
        self.resident_bytes = 0
        self.resident_mb = 0.0
        self.reset_error: str | None = None
        self.tokenizer = None
        if use_chat_template:
            self.tokenizer = self._load_hf_tokenizer()
        t0 = time.perf_counter()
        self.runner = TextLLMRunner(self.pte_path, self.tokenizer_path)
        self.load_seconds = time.perf_counter() - t0
        self.resident_bytes = _peak_rss_bytes()
        _LIVE_ADVISORS.add(self)

    def close(self) -> None:
        """Release the native runtime.

        The runner owns worker threads. If it is destroyed while the
        interpreter is shutting down, the process aborts with
        ``terminate called without an active exception``. Releasing it
        explicitly, while the interpreter is still alive, avoids that.

        A failing reset is reported rather than swallowed: the runner keeps
        its prompt state and its KV cache, so a reset that did not happen
        means the runtime is released with that state still live, which is
        exactly the situation that aborts the process.
        """
        runner = getattr(self, "runner", None)
        if runner is None:
            return
        try:
            runner.reset()
        except Exception as exc:
            self.reset_error = repr(exc)
            print(
                f"advisory runner reset failed during close: {exc!r}; the runtime "
                "is being released with its prompt state still live",
                file=sys.stderr,
            )
        self.runner = None

    def _load_hf_tokenizer(self):
        """The HF fast tokenizer, used only to apply the chat template.

        A failure here is loud. The chat template is the prompt shape the
        artefact was exported and smoke-tested against: without it the model
        is fed raw text, answers it markedly worse, and the station would
        silently be running a different advisory from the one that was
        measured. Failing to load the tokenizer is therefore a deployment
        error, not a degraded mode.

        The tokenizer ships with the artefact, in one directory, with the
        chat template in its configuration. The template is what makes the
        prompt a chat prompt, so a tokenizer without it is not the same
        tokenizer.
        """
        from transformers import AutoTokenizer

        tokenizer_file = Path(self.tokenizer_path)
        path = tokenizer_file.parent if tokenizer_file.is_file() else tokenizer_file
        if not (path / "tokenizer.json").exists():
            raise RuntimeError(
                f"the chat tokenizer is missing at {path}. The advisory is run in the "
                "Llama chat format; without the tokenizer the prompt shape changes "
                "and the artefact no longer answers the prompt it was built for."
            )
        return AutoTokenizer.from_pretrained(str(path))

    # -- generation --------------------------------------------------------
    def generate(self, prompt: str, max_new_tokens: int | None = None) -> AdvisorResult:
        from executorch.extension.llm.runner import GenerationConfig

        cfg = GenerationConfig(
            echo=False,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            temperature=self.temperature,
        )
        chunks: list[str] = []
        stats_holder: dict = {}

        # The runner keeps its KV cache and prompt state between calls, so a
        # second advisory would continue the first one's context (and, once the
        # context is full, return nothing at all).  Every advisory therefore
        # starts from a reset runner.
        #
        # A failing reset is an error, not something to ignore: the next
        # generation would otherwise be answered in the context of the
        # previous part, and the advisory would be about the wrong component.
        try:
            self.runner.reset()
        except Exception as exc:
            return AdvisorResult(
                text="",
                ok=False,
                path="llm",
                error=f"the advisory runner could not be reset before generation: {exc!r}",
            )

        def on_token(text: str) -> None:
            chunks.append(text)

        def on_stats(stats) -> None:
            stats_holder["stats"] = stats

        try:
            self.runner.generate(prompt, cfg, on_token, on_stats)
        except Exception as exc:  # keep the station alive
            return AdvisorResult(text="", ok=False, path="llm", error=repr(exc))

        text = "".join(chunks)
        # The artefact is memory-mapped, so the pages are faulted in as the
        # model runs and the resident set only reaches its peak after a
        # generation.  The peak is re-read here, because it is the number the
        # deployment budget is built from.
        self.resident_bytes = max(self.resident_bytes, _peak_rss_bytes())
        self.resident_mb = round(self.resident_bytes / (1024.0 * 1024.0), 1)
        # the runner can emit the end-of-turn marker as text; keep it out of
        # the operator UI and the audit log
        text = re.sub(r"<\|[^|]*\|>", "", text).strip()
        stats = stats_holder.get("stats")
        out = AdvisorResult(text=text, ok=True, path="llm", resident_mb=self.resident_mb)
        if stats is not None:
            out.prompt_tokens = int(stats.num_prompt_tokens)
            out.generated_tokens = int(stats.num_generated_tokens)
            out.ttft_ms = float(stats.first_token_ms - stats.inference_start_ms)
            out.inference_ms = float(stats.inference_end_ms - stats.inference_start_ms)
            out.decode_ms = float(stats.inference_end_ms - stats.first_token_ms)
            if out.decode_ms > 0 and out.generated_tokens > 0:
                out.tokens_per_second = 1000.0 * out.generated_tokens / out.decode_ms
            try:
                out.raw_stats = json.loads(stats.to_json_string())
            except Exception:
                out.raw_stats = {"repr": repr(stats)}
            # The runtime reports the prefill rate itself; the two stages
            # are far apart in speed, so the budget needs both.
            try:
                out.prefill_tokens_per_second = float(
                    out.raw_stats.get("prefill_token_per_sec", 0.0)
                )
            except (TypeError, ValueError):
                out.prefill_tokens_per_second = 0.0
        return out

@atexit.register
def _release_all_advisors() -> None:
    """Release every loaded runner before the interpreter shuts down."""
    for advisor in list(_LIVE_ADVISORS):
        try:
            advisor.close()
        except Exception:
            pass
