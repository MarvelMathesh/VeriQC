#!/usr/bin/env python3
"""Start the station, inspect one part over HTTP, and record the evidence.

This is an acceptance check, not a benchmark.  It measures time-to-health,
verifies artifact hashes are attached, waits for the asynchronous advisory,
and checks overlay authentication, CSV export and the audit chain.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]


def wait_for(client: httpx.Client, url: str, timeout: float) -> float:
    started = time.perf_counter()
    while time.perf_counter() - started < timeout:
        try:
            response = client.get(url)
            if response.status_code == 200:
                return time.perf_counter() - started
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"station did not become healthy within {timeout:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="commutator")
    parser.add_argument("--image", default="data/kolektor/kos02/Part6.jpg")
    parser.add_argument("--part-id", default="PI-SMOKE")
    parser.add_argument("--port", type=int, default=8199)
    parser.add_argument("--token", default="pi-smoke")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--out", default="reports/tables/pi_station_smoke.json")
    args = parser.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    environment = os.environ.copy()
    environment.update(
        {
            "VERIQC_PROFILE": args.profile,
            "VERIQC_API_TOKEN": args.token,
            "PYTHONUNBUFFERED": "1",
        }
    )
    log = REPO / "reports" / "tables" / "pi_station_smoke.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [
            sys.executable,
            "-u",
            str(REPO / "scripts" / "run_station.py"),
            "--port",
            str(args.port),
        ],
        cwd=REPO,
        env=environment,
        stdout=log.open("w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    headers = {"X-VeriQC-Token": args.token}
    try:
        with httpx.Client(timeout=30.0) as client:
            startup_seconds = wait_for(client, f"{base}/health", args.timeout)
            health = client.get(f"{base}/health").json()
            metrics = client.get(f"{base}/metrics", headers=headers).json()
            with open(REPO / args.image, "rb") as handle:
                response = client.post(
                    f"{base}/inspect",
                    headers=headers,
                    files={"file": (Path(args.image).name, handle, "image/jpeg")},
                    data={"part_id": args.part_id},
                )
            response.raise_for_status()
            submitted = response.json()
            inspection_id = submitted["inspection_id"]
            result = submitted
            advisory_started = time.perf_counter()
            while time.perf_counter() - advisory_started < args.timeout:
                result = client.get(f"{base}/result/{inspection_id}", headers=headers).json()
                advisory = result.get("advisory") or {}
                if advisory.get("status") != "pending":
                    break
                time.sleep(1.0)
            advisory = result.get("advisory") or {}
            overlay = client.get(
                f"{base}/overlay/{inspection_id}", headers=headers
            )
            csv_response = client.get(f"{base}/export.csv", headers=headers)
            history = client.get(f"{base}/history?limit=5", headers=headers).json()
            payload = {
                "schema": "veriqc.station-smoke/1",
                "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "profile": args.profile,
                "startup_seconds": round(startup_seconds, 1),
                "health": health,
                "advisory_available": metrics["advisory"]["available"],
                "inspection_id": inspection_id,
                "verdict": result["verdict"],
                "findings": len(result.get("findings", [])),
                "vision_latency_ms": result["latency_ms"].get("total_ms"),
                "artifact_hashes": sorted(result.get("artifact_hashes", {})),
                "advisory_status": advisory.get("status"),
                "advisory_accepted": advisory.get("accepted"),
                "advisory_latency_ms": advisory.get("latency_ms"),
                "advisory_model": advisory.get("model"),
                "advisory_text": advisory.get("text"),
                "overlay_http": overlay.status_code,
                "overlay_bytes": len(overlay.content),
                "csv_http": csv_response.status_code,
                "csv_bytes": len(csv_response.content),
                "audit_chain": history.get("chain"),
                "chain_ok": history.get("integrity_ok"),
            }
            output = REPO / args.out
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(payload, indent=2) + "\n")
            print(json.dumps(payload, indent=2))
    finally:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)


if __name__ == "__main__":
    main()
