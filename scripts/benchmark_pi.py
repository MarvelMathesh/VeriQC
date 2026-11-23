#!/usr/bin/env python3
"""Measure the Raspberry Pi deployment path.

Three modes are deliberately separate processes so one mode cannot inflate the
next one's resident memory:

``vision``
    Quality gate, model, decision and findings on the inspection path.
``advisory``
    The isolated ExecuTorch child on real non-PASS findings.  Vision is used
    only to construct the findings and is not included in the advisory timing.
``resident``
    The station with vision and the advisory child alive together, driven at a
    fixed arrival interval.  Reports vision percentiles, process-tree memory,
    queue high-water mark, queue drops and generation timeouts.

Temperature and throttle flags are sampled once a second.  The script never
claims a result it did not measure: missing sensors stay null.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from veriqc.advisory.cards import load_cards  # noqa: E402
from veriqc.advisory.process_runner import AdvisorRuntimeSpec, ProcessAdvisorRunner  # noqa: E402
from veriqc.advisory.worker import AdvisoryWorker  # noqa: E402
from veriqc.config import load_profile  # noqa: E402
from veriqc.datasets import load_manifest  # noqa: E402
from veriqc.pipeline import Pipeline  # noqa: E402
from veriqc.station.engine import Station  # noqa: E402

MVTEC_PROFILES = {
    "textile_carpet": "carpet",
    "packaging_bottle": "bottle",
    "pharma_capsule": "capsule",
    "automotive_metal_nut": "metal_nut",
    "electronics_transistor": "transistor",
}
CASE_PROFILES = ("commutator", "steel_surface", "textile_carpet", "packaging_bottle")
THROTTLE_BITS = {
    0: "under_voltage_now",
    1: "arm_frequency_capped_now",
    2: "throttled_now",
    3: "soft_temperature_limit_now",
    16: "under_voltage_occurred",
    17: "arm_frequency_capped_occurred",
    18: "throttled_occurred",
    19: "soft_temperature_limit_occurred",
}


def package(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def read_temperature() -> float | None:
    try:
        return round(int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000.0, 1)
    except (OSError, ValueError):
        pass
    try:
        output = subprocess.run(
            ["vcgencmd", "measure_temp"], check=True, capture_output=True, text=True
        ).stdout
        return round(float(output.split("=")[1].split("'")[0]), 1)
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return None


def read_throttled() -> dict[str, Any] | None:
    try:
        output = subprocess.run(
            ["vcgencmd", "get_throttled"], check=True, capture_output=True, text=True
        ).stdout.strip()
        value = int(output.split("=")[1], 16)
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return None
    return {
        "raw": f"0x{value:x}",
        "flags": [name for bit, name in THROTTLE_BITS.items() if value & (1 << bit)],
    }


class Monitor(threading.Thread):
    def __init__(self, interval_s: float = 1.0) -> None:
        super().__init__(daemon=True, name="PiMonitor")
        self.interval_s = interval_s
        self._stop = threading.Event()
        self.temperature: list[float] = []
        self.throttle: list[dict[str, Any]] = []
        self.parent_rss_mb: list[float] = []
        self.tree_rss_mb: list[float] = []

    def run(self) -> None:
        import psutil

        root = psutil.Process(os.getpid())
        while not self._stop.is_set():
            temperature = read_temperature()
            if temperature is not None:
                self.temperature.append(temperature)
            throttle = read_throttled()
            if throttle is not None:
                self.throttle.append(throttle)
            try:
                processes = [root, *root.children(recursive=True)]
                parent_mb = root.memory_info().rss / (1024.0 * 1024.0)
                total_mb = parent_mb + sum(
                    child.memory_info().rss / (1024.0 * 1024.0)
                    for child in processes[1:]
                    if child.is_running()
                )
                self.parent_rss_mb.append(round(parent_mb, 1))
                self.tree_rss_mb.append(round(total_mb, 1))
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def stop(self) -> None:
        self._stop.set()
        self.join(timeout=5.0)

    def summary(self) -> dict[str, Any]:
        temperatures = self.temperature
        flags = []
        for sample in self.throttle:
            for flag in sample["flags"]:
                if flag.endswith("_occurred") and flag not in flags:
                    flags.append(flag)
        return {
            "temperature_c": {
                "start": temperatures[0] if temperatures else None,
                "median": round(statistics.median(temperatures), 1) if temperatures else None,
                "max": max(temperatures) if temperatures else None,
                "end": temperatures[-1] if temperatures else None,
                "samples": len(temperatures),
            },
            "throttle": {
                "before": self.throttle[0] if self.throttle else None,
                "after": self.throttle[-1] if self.throttle else None,
                "occurred_flags": flags,
            },
            "rss_mb": {
                "parent_max": max(self.parent_rss_mb) if self.parent_rss_mb else None,
                "tree_max": max(self.tree_rss_mb) if self.tree_rss_mb else None,
                "samples": len(self.tree_rss_mb),
            },
        }


def environment() -> dict[str, Any]:
    governor = None
    try:
        governor = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text().strip()
    except OSError:
        pass
    model = None
    try:
        model = Path("/proc/device-tree/model").read_text().strip("\x00").strip()
    except OSError:
        pass
    return {
        "hostname": platform.node(),
        "model": model,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_governor": governor,
        "packages": {
            "torch": package("torch"),
            "torchvision": package("torchvision"),
            "ultralytics": package("ultralytics"),
            "numpy": package("numpy"),
            "executorch": package("executorch"),
            "transformers": package("transformers"),
            "opencv": package("opencv-python"),
        },
        "memory_total_mb": round(
            os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024.0 * 1024.0), 1
        ),
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def percentiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "median": round(float(np.quantile(array, 0.50)), 2),
        "p95": round(float(np.quantile(array, 0.95)), 2),
        "p99": round(float(np.quantile(array, 0.99)), 2),
        "min": round(float(array.min()), 2),
        "max": round(float(array.max()), 2),
    }


def sample_items(profile: str, n: int, *, defects_only: bool = False) -> list[dict]:
    if profile == "steel_surface":
        manifest = load_manifest(REPO / "data" / "prepared" / "neu" / "manifest.json")
        items = manifest["items"]["test"]
        return [item for item in items if item["is_defect"]][:n] if defects_only else items[:n]
    if profile == "commutator":
        manifest = load_manifest(REPO / "data" / "prepared" / "kolektor" / "manifest.json")
        items = manifest["folds"][0]["test"]
    else:
        category = MVTEC_PROFILES[profile]
        manifest = load_manifest(REPO / "data" / "prepared" / "mvtec" / "manifest.json")
        items = [
            item
            for item in manifest["categories"][category]["items"]
            if item["split"] == "test"
        ]
    if defects_only:
        items = [item for item in items if item["is_defect"]]
    return items[:n]


def load_images(items: list[dict]) -> list[Image.Image]:
    images = []
    for item in items:
        with Image.open(item["image"]) as opened:
            images.append(opened.convert("RGB"))
    return images


def run_vision(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    torch.set_num_threads(args.threads)
    profile = load_profile(REPO / "configs" / "profiles" / f"{args.profile}.yaml")
    pipeline = Pipeline(profile, device="cpu", overlay_root=Path(args.overlay_root))
    items = sample_items(args.profile, args.n + args.warmup)
    if not items:
        raise SystemExit("no images for the requested profile")
    images = load_images(items)
    while len(images) < args.n + args.warmup:
        images.extend(load_images(items))
    for image in images[: args.warmup]:
        pipeline.inspect(image, "PI-WARMUP", overlay=False)
    monitor = Monitor()
    monitor.start()
    started = time.perf_counter()
    latencies = []
    wall = []
    try:
        for index, image in enumerate(images[args.warmup :]):
            began = time.perf_counter()
            result = pipeline.inspect(image, f"PI-{index:04d}", overlay=False)
            wall.append(round((time.perf_counter() - began) * 1000.0, 2))
            latencies.append(float(result.latency_ms["total_ms"]))
    finally:
        monitor.stop()
    elapsed = time.perf_counter() - started
    return {
        "mode": "vision",
        "profile": args.profile,
        "device": "cpu",
        "threads": args.threads,
        "images": len(latencies),
        "warmup_images": args.warmup,
        "total_ms": percentiles(latencies),
        "wall_ms": percentiles(wall),
        "throughput_images_per_second": round(len(latencies) / elapsed, 3),
        "elapsed_s": round(elapsed, 1),
        "monitor": monitor.summary(),
    }


def build_advisory_cases(args: argparse.Namespace) -> list[tuple[str, Any]]:
    cases: list[tuple[str, Any]] = []
    for profile_id in CASE_PROFILES:
        profile = load_profile(REPO / "configs" / "profiles" / f"{profile_id}.yaml")
        pipeline = Pipeline(profile, device="cpu", overlay_root=Path(args.overlay_root))
        items = sample_items(profile_id, args.advisory_cases, defects_only=True)
        for item in items:
            with Image.open(item["image"]) as opened:
                result = pipeline.inspect(
                    opened.convert("RGB"), item["part"], overlay=False
                )
            if result.verdict != "PASS":
                cases.append((profile_id, result))
        del pipeline
        gc.collect()
    return cases


def run_advisory(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    cases = build_advisory_cases(args)
    build_seconds = time.perf_counter() - started
    pte = REPO / "models" / "advisory" / "llama32_1b_int8.pte"
    tokenizer = REPO / "models" / "advisory" / "tokenizer"
    monitor = Monitor()
    monitor.start()
    load_started = time.perf_counter()
    runner = ProcessAdvisorRunner(
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
        start_timeout_s=args.start_timeout,
    )
    startup_seconds = time.perf_counter() - load_started
    workers = {}
    for profile_id in CASE_PROFILES:
        profile = load_profile(REPO / "configs" / "profiles" / f"{profile_id}.yaml")
        workers[profile_id] = AdvisoryWorker(load_cards(profile.cards_path), runner)
    rows = []
    try:
        for profile_id, result in cases:
            outcome = workers[profile_id].advise_result(result)
            rows.append(
                {
                    "profile_id": profile_id,
                    "verdict": result.verdict,
                    "defect_class": result.findings[0].defect_class if result.findings else None,
                    "accepted": outcome.accepted,
                    "path": outcome.path,
                    "latency_ms": round(outcome.latency_ms, 1),
                    "ttft_ms": outcome.model.get("ttft_ms"),
                    "tokens_per_second": outcome.model.get("tokens_per_second"),
                    "generated_tokens": outcome.model.get("generated_tokens"),
                    "resident_mb": outcome.model.get("resident_mb"),
                    "violations": outcome.verification.get("violations", []),
                }
            )
            print(
                profile_id,
                result.verdict,
                outcome.accepted,
                round(outcome.latency_ms, 0),
                flush=True,
            )
    finally:
        runner.close()
        monitor.stop()
    latencies = [row["latency_ms"] for row in rows]
    ttfts = [row["ttft_ms"] for row in rows if row["ttft_ms"] is not None]
    rates = [
        row["tokens_per_second"]
        for row in rows
        if row["tokens_per_second"] is not None
    ]
    resident = [row["resident_mb"] for row in rows if row["resident_mb"]]
    return {
        "mode": "advisory",
        "device": "cpu",
        "cases": len(rows),
        "accepted": sum(1 for row in rows if row["accepted"]),
        "acceptance_rate": round(
            sum(1 for row in rows if row["accepted"]) / max(1, len(rows)), 4
        ),
        "case_build_seconds": round(build_seconds, 1),
        "model_startup_seconds": round(startup_seconds, 1),
        "latency_ms": percentiles(latencies) if latencies else None,
        "ttft_ms": percentiles(ttfts) if ttfts else None,
        "tokens_per_second_median": round(statistics.median(rates), 2) if rates else None,
        "resident_mb_max": round(max(resident), 1) if resident else None,
        "monitor": monitor.summary(),
        "rows": rows,
    }


def run_resident(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    station = Station(
        args.profile,
        device="cpu",
        enable_advisory=True,
        audit_path=Path(args.audit_path),
    )
    startup_seconds = time.perf_counter() - started
    if station.runner is None:
        raise SystemExit(f"advisory runtime unavailable: {station.runner_error}")
    items = sample_items(args.profile, args.defects, defects_only=True)
    if not items:
        raise SystemExit("no non-PASS source images for the resident run")
    images = load_images(items)
    monitor = Monitor()
    monitor.start()
    latencies = []
    walls = []
    queue_sizes = []
    drops = 0
    began = time.perf_counter()
    try:
        for index in range(args.n):
            image = images[index % len(images)]
            call_started = time.perf_counter()
            result = station.inspect(image, f"PI-RES-{index:04d}")
            wall = (time.perf_counter() - call_started) * 1000.0
            walls.append(round(wall, 2))
            latencies.append(float(result.latency_ms["total_ms"]))
            if result.advisory and result.advisory.get("note") == "advisory queue full":
                drops += 1
            queue_sizes.append(station._queue.qsize())
            remaining = args.interval - (time.perf_counter() - call_started)
            if remaining > 0:
                time.sleep(remaining)
        submit_seconds = time.perf_counter() - began
        drain_started = time.perf_counter()
        while (
            station._queue.qsize() > 0
            and time.perf_counter() - drain_started < args.drain
        ):
            time.sleep(1.0)
        remaining = station._queue.qsize()
        drained = remaining == 0
        drain_seconds = time.perf_counter() - drain_started
        metrics = station.metrics()
        outcomes = station.store.advisory_outcomes()
        verdict_counts = station.store.verdict_counts()
        chain = station.store.verify()
    finally:
        station.close()
        monitor.stop()
    return {
        "mode": "resident",
        "profile": args.profile,
        "device": "cpu",
        "startup_seconds": round(startup_seconds, 1),
        "inspections": len(latencies),
        "arrival_interval_s": args.interval,
        "submit_seconds": round(submit_seconds, 1),
        "drain_seconds": round(drain_seconds, 1),
        "drained": drained,
        "vision_ms": percentiles(latencies),
        "wall_ms": percentiles(walls),
        "qsize_max_observed": max(queue_sizes) if queue_sizes else 0,
        "queue_drops_reported": drops,
        "stats": metrics["stats"],
        "advisory_outcomes": outcomes,
        "verdict_counts": verdict_counts,
        "queue_remaining_at_end": remaining,
        "audit_chain": chain,
        "monitor": monitor.summary(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["vision", "advisory", "resident"], required=True)
    parser.add_argument("--profile", default="commutator")
    parser.add_argument("--images", dest="n", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--advisory-cases", type=int, default=10)
    parser.add_argument("--defects", type=int, default=18)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--drain", type=float, default=120.0)
    parser.add_argument("--start-timeout", type=float, default=600.0)
    parser.add_argument("--overlay-root", default="/tmp/veriqc-pi-overlays")
    parser.add_argument("--audit-path", default="/tmp/veriqc-pi-audit.db")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    mode: Callable[[argparse.Namespace], dict[str, Any]] = {
        "vision": run_vision,
        "advisory": run_advisory,
        "resident": run_resident,
    }[args.mode]
    payload = {
        "schema": "veriqc.pi-benchmark/1",
        "environment": environment(),
        **mode(args),
    }
    output = REPO / (args.out or f"reports/tables/pi_{args.mode}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
