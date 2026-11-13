"""The ExecuTorch advisor runtime.

One long-lived ``TextLLMRunner`` object; the model is loaded once (the KV
cache for a 1B model at a 1-2k context is allocated in full at load time, so
reloading per request would dominate the latency budget).

The runner gives us exactly the numbers the report needs: prompt tokens,
time to first token, generated tokens and the inference window, so TTFT and
decode rate are measured rather than estimated.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["AdvisorResult", "ExecuTorchAdvisor"]


@dataclass
class AdvisorResult:
    text: str
    ok: bool
    path: str = "llm"  # llm | retry | template
    prompt_tokens: int = 0
    generated_tokens: int = 0
    ttft_ms: float = 0.0
    inference_ms: float = 0.0
    decode_ms: float = 0.0
    tokens_per_second: float = 0.0
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
            "error": self.error,
        }


class ExecuTorchAdvisor:
    """Thin wrapper around ExecuTorch's LLM runner."""

    def __init__(
        self,
        pte_path: str | Path,
        tokenizer_path: str | Path,
        max_new_tokens: int = 96,
        temperature: float = 0.0,
        threads: int = 4,
        max_seq_length: int = 1024,
        use_chat_template: bool = True,
    ):
        from executorch.extension.llm.runner import GenerationConfig, TextLLMRunner

        self.pte_path = str(pte_path)
        self.tokenizer_path = str(tokenizer_path)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.threads = threads
        self.max_seq_length = max_seq_length
        self.load_seconds = 0.0
        self.tokenizer = None
        if use_chat_template:
            self.tokenizer = self._load_hf_tokenizer()
        t0 = time.perf_counter()
        self.runner = TextLLMRunner(self.pte_path, self.tokenizer_path)
        self.load_seconds = time.perf_counter() - t0

    @staticmethod
    def _load_hf_tokenizer():
        """The HF fast tokenizer, used only to apply the chat template."""
        try:
            from transformers import AutoTokenizer

            path = Path(__file__).resolve().parents[2] / "models" / "llama32_1b" / "hf"
            if (path / "tokenizer.json").exists():
                return AutoTokenizer.from_pretrained(str(path))
        except Exception:
            return None
        return None

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
        try:
            self.runner.reset()
        except Exception:
            pass

        def on_token(text: str) -> None:
            chunks.append(text)

        def on_stats(stats) -> None:
            stats_holder["stats"] = stats

        try:
            self.runner.generate(prompt, cfg, on_token, on_stats)
        except Exception as exc:  # keep the station alive
            return AdvisorResult(text="", ok=False, path="llm", error=repr(exc))

        text = "".join(chunks)
        # the runner can emit the end-of-turn marker as text; keep it out of
        # the operator UI and the audit log
        text = re.sub(r"<\|[^|]*\|>", "", text).strip()
        stats = stats_holder.get("stats")
        out = AdvisorResult(text=text, ok=True, path="llm")
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
        return out

    def reset(self) -> None:
        try:
            self.runner.reset()
        except Exception:
            pass
