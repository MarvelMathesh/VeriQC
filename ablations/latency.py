"""Latency measurement helpers for the edge budget.

Three things are measured here, all on a 4-thread CPU slice so that the
numbers line up with the Raspberry Pi 5's four Cortex-A76 cores:

* ONNX Runtime latency (fp32 and dynamically quantised int8);
* the ExecuTorch + XNNPACK latency of the same model, which is the runtime
  Arm used for the published Raspberry Pi 5 numbers;
* the conversion between the two, so that a measurement taken in one runtime
  can be reported for the other.

The published anchors (Arm AI Model Optimizer, ExecuTorch v1.1.0 + XNNPACK
INT8, Raspberry Pi 5 16 GB) are:

    mobilenet_v3_small   2.87 ms      resnet18          9.36 ms
    wideresnet50        49.43 ms      ssd_mobilenet_v2  39.76 ms
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "LatencyResult",
    "measure_onnxruntime",
    "quantize_dynamic_int8",
    "measure_torch_module",
]


@dataclass
class LatencyResult:
    runtime: str
    precision: str
    p50_ms: float
    p95_ms: float
    mean_ms: float
    runs: int
    threads: int
    artefact_bytes: int = 0

    def to_dict(self) -> dict:
        return {
            "runtime": self.runtime,
            "precision": self.precision,
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "mean_ms": round(self.mean_ms, 3),
            "runs": self.runs,
            "threads": self.threads,
            "artefact_bytes": self.artefact_bytes,
        }


def measure_onnxruntime(
    model_path: str | Path,
    input_shape: tuple[int, int, int, int] = (1, 3, 224, 224),
    runs: int = 50,
    warmup: int = 10,
    threads: int = 4,
    precision: str = "fp32",
) -> LatencyResult:
    """Measure an ONNX model with ONNX Runtime on a fixed thread count."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
    x = np.random.rand(*input_shape).astype(np.float32)
    inp = sess.get_inputs()[0].name
    for _ in range(warmup):
        sess.run(None, {inp: x})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {inp: x})
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return LatencyResult(
        runtime="onnxruntime",
        precision=precision,
        p50_ms=statistics.median(times),
        p95_ms=times[int(0.95 * (len(times) - 1))],
        mean_ms=statistics.fmean(times),
        runs=runs,
        threads=threads,
        artefact_bytes=Path(model_path).stat().st_size,
    )


def quantize_dynamic_int8(fp32_path: str | Path, int8_path: str | Path) -> Path:
    """Dynamic int8 quantisation with ONNX Runtime (the Pi-class path)."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out = Path(int8_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(out),
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
    )
    return out


def measure_torch_module(
    module,
    example,
    runs: int = 50,
    warmup: int = 10,
    threads: int = 4,
    precision: str = "fp32",
) -> LatencyResult:
    """Measure a torch module (used for the torchvision / effdet candidates)."""
    import torch

    torch.set_num_threads(threads)
    module = module.eval()
    with torch.no_grad():
        for _ in range(warmup):
            module(example)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            module(example)
            times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return LatencyResult(
        runtime="pytorch",
        precision=precision,
        p50_ms=statistics.median(times),
        p95_ms=times[int(0.95 * (len(times) - 1))],
        mean_ms=statistics.fmean(times),
        runs=runs,
        threads=threads,
    )
