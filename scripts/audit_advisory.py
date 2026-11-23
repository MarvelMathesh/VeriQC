#!/usr/bin/env python3
"""Audit the advisory path on real inspection findings.

The audit measures acceptance, grounding violations and latency.  Usefulness
requires an independent human rating and is reported as not measured unless a
rating sheet is supplied.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from veriqc import sha256_file  # noqa: E402
from veriqc.advisory.process_runner import AdvisorRuntimeSpec, ProcessAdvisorRunner  # noqa: E402
from veriqc.advisory.worker import AdvisoryWorker  # noqa: E402
from veriqc.config import load_profile  # noqa: E402
from veriqc.datasets import load_manifest  # noqa: E402
from veriqc.pipeline import Pipeline  # noqa: E402

MVTEC_PROFILES = {
    "textile_carpet": "carpet",
    "packaging_bottle": "bottle",
    "pharma_capsule": "capsule",
    "automotive_metal_nut": "metal_nut",
}


def sample_items(profile: str, n: int) -> list[dict]:
    if profile == "steel_surface":
        manifest = load_manifest(REPO / "data" / "prepared" / "neu" / "manifest.json")
        return manifest["items"]["test"][:n]
    if profile == "commutator":
        manifest = load_manifest(REPO / "data" / "prepared" / "kolektor" / "manifest.json")
        defects = [item for item in manifest["folds"][0]["test"] if item["is_defect"]]
        return defects[:n]
    category = MVTEC_PROFILES[profile]
    manifest = load_manifest(REPO / "data" / "prepared" / "mvtec" / "manifest.json")
    defects = [item for item in manifest["categories"][category]["items"] if item["split"] == "test" and item["is_defect"]]
    return defects[:n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=10, help="cases per profile")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="reports/tables/advisory_audit.json")
    args = parser.parse_args()
    pte = REPO / "models" / "advisory" / "llama32_1b_int8.pte"
    tokenizer = REPO / "models" / "advisory" / "tokenizer"
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
    )
    rows = []
    started = time.time()
    try:
        for profile_id in ("commutator", "steel_surface", "textile_carpet", "packaging_bottle"):
            profile = load_profile(REPO / "configs" / "profiles" / f"{profile_id}.yaml")
            pipeline = Pipeline(profile, device=args.device)
            worker = AdvisoryWorker(pipeline.cards, runner, max_new_tokens=48)
            items = sample_items(profile_id, args.cases)
            for item in items:
                with Image.open(item["image"]) as image:
                    result = pipeline.inspect(image.convert("RGB"), item["part"], overlay=False)
                if result.verdict == "PASS":
                    continue
                outcome = worker.advise_result(result)
                rows.append(
                    {
                        "profile_id": profile_id,
                        "part_id": item["part"],
                        "verdict": result.verdict,
                        "defect_class": result.findings[0].defect_class if result.findings else None,
                        "severity": result.findings[0].severity if result.findings else None,
                        "path": outcome.path,
                        "accepted": outcome.accepted,
                        "violations": outcome.verification.get("violations", []),
                        "verification": outcome.verification,
                        "model": outcome.model,
                        "text": outcome.text,
                        "template": outcome.template,
                        "latency_ms": round(outcome.latency_ms, 1),
                        "ttft_ms": outcome.model.get("ttft_ms"),
                        "tokens_per_second": outcome.model.get("tokens_per_second"),
                        "generated_tokens": outcome.model.get("generated_tokens"),
                    }
                )
                print(profile_id, item["part"], result.verdict, outcome.accepted, round(outcome.latency_ms, 0), flush=True)
    finally:
        runner.close()
    accepted = [row for row in rows if row["accepted"]]
    rejected = [row for row in rows if not row["accepted"]]
    format_rejected = [
        row for row in rejected
        if any("valid three-line" in violation for violation in row["violations"])
    ]
    accepted_with_violation = [row for row in accepted if row["violations"]]
    latencies = [row["latency_ms"] for row in rows if row.get("latency_ms") is not None]
    accepted_latencies = [
        row["latency_ms"] for row in accepted if row.get("latency_ms") is not None
    ]
    ttfts = [row["ttft_ms"] for row in rows if row.get("ttft_ms") is not None]
    accepted_ttfts = [
        row["ttft_ms"] for row in accepted if row.get("ttft_ms") is not None
    ]
    decode_rates = [
        row["tokens_per_second"] for row in rows if row.get("tokens_per_second") is not None
    ]
    generated_tokens = [
        row["generated_tokens"] for row in rows if row.get("generated_tokens") is not None
    ]
    resident = [
        row["model"].get("resident_mb")
        for row in rows
        if row.get("model") and row["model"].get("resident_mb")
    ]
    rejection_reasons: dict[str, int] = {}
    for row in rejected:
        for violation in row["violations"]:
            rejection_reasons[violation] = rejection_reasons.get(violation, 0) + 1
    report = {
        "cases": len(rows),
        "accepted": len(accepted),
        "acceptance_rate": round(len(accepted) / max(1, len(rows)), 4),
        "rejected": len(rejected),
        "rejected_format": len(format_rejected),
        "accepted_grounding_violations": len(accepted_with_violation),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "latency_ms_median": round(statistics.median(latencies), 1) if latencies else None,
        "latency_ms_median_accepted": (
            round(statistics.median(accepted_latencies), 1) if accepted_latencies else None
        ),
        "latency_ms_min": round(min(latencies), 1) if latencies else None,
        "latency_ms_max": round(max(latencies), 1) if latencies else None,
        "ttft_ms_median": round(statistics.median(ttfts), 1) if ttfts else None,
        "ttft_ms_median_accepted": (
            round(statistics.median(accepted_ttfts), 1) if accepted_ttfts else None
        ),
        "ttft_ms_min": round(min(ttfts), 1) if ttfts else None,
        "ttft_ms_max": round(max(ttfts), 1) if ttfts else None,
        "tokens_per_second_median": (
            round(statistics.median(decode_rates), 2) if decode_rates else None
        ),
        "generated_tokens_min": min(generated_tokens) if generated_tokens else None,
        "generated_tokens_max": max(generated_tokens) if generated_tokens else None,
        "resident_mb_median": round(statistics.median(resident), 1) if resident else None,
        "resident_mb_max": round(max(resident), 1) if resident else None,
        "human_usefulness": "not measured; blind sheet prepared, awaiting independent rater",
        "model_sha256": sha256_file(pte),
        "tokenizer_sha256": sha256_file(tokenizer / "tokenizer.json"),
        "wall_seconds": round(time.time() - started, 1),
        "rows": rows,
    }
    output = REPO / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
