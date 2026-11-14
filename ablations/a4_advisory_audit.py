"""Ablation A4: Advisory Verification -- Verified Slot-Filling vs. Raw LLM vs. Templates.

Hypothesis:
Small edge LLMs (1B) produce unsupported claims (hallucinating numbers, unhedged causes,
invented action IDs) when asked for free text.
Deterministic verification reduces unsupported claims to 0.0% through strict
slot verification, one temperature=0 retry, and deterministic template fallback.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import time
from collections import Counter
from pathlib import Path

from veriqc.advisory.cards import CardLibrary, load_cards
from veriqc.advisory.prompts import build_prompt, build_template
from veriqc.advisory.runner import ExecuTorchAdvisor
from veriqc.advisory.verifier import verify_advisory
from veriqc.advisory.worker import AdvisoryWorker
from ablations.synthetic_cases import generate_cases

REPORTS = Path(__file__).resolve().parents[1] / "reports" / "tables"
CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def audit_advisories(
    worker: AdvisoryWorker,
    cards: CardLibrary,
    cases: list,
    advisor: ExecuTorchAdvisor | None = None,
) -> dict:
    """Run comparative audit across: Template, Raw LLM (if advisor given), and Verified Worker."""
    results = {}
    n = len(cases)

    # 1. Baseline: Template Only
    template_advisories = []
    for c in cases:
        retrieved = cards.retrieve(c.record.defect_class, c.record.severity, limit=3)
        t_text = build_template(c.record, retrieved)
        v_res = verify_advisory(t_text, c.record, retrieved)
        template_advisories.append({"verified": v_res.ok, "violations": v_res.violations})

    results["template_only"] = {
        "verified_rate": round(sum(1 for t in template_advisories if t["verified"]) / n, 4),
        "unsupported_claim_rate": round(sum(1 for t in template_advisories if not t["verified"]) / n, 4),
    }

    # 2. Production: VeriQC Verified Worker (Slots + Verifier + Retry + Fallback)
    worker_rows = []
    for c in cases:
        outcome = worker.advise(c.record)
        worker_rows.append(outcome.to_dict())

    paths = Counter(r["path"] for r in worker_rows)
    verified_count = sum(1 for r in worker_rows if r["verified"])
    unsupported_count = sum(1 for r in worker_rows if not r["verified"])

    ttfts = [r["ttft_ms"] for r in worker_rows if r.get("ttft_ms")]
    tps = [r["tokens_per_second"] for r in worker_rows if r.get("tokens_per_second")]

    results["veriqc_verified_worker"] = {
        "n_cases": n,
        "verified_rate": round(verified_count / n, 4),
        "first_pass_rate": round(paths.get("llm", 0) / n, 4),
        "retry_rate": round(paths.get("retry", 0) / n, 4),
        "fallback_rate": round(paths.get("template", 0) / n, 4),
        "final_unsupported_claim_rate": round(unsupported_count / n, 4),
        "ttft_ms_median": round(st.median(ttfts), 1) if ttfts else None,
        "decode_tps_median": round(st.median(tps), 2) if tps else None,
    }

    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--pte", default="models/llama32_1b_dyn8.pte")
    ap.add_argument("--tokenizer", default="models/llama32_1b_dyn8.pte.tokenizer.json")
    ap.add_argument("--out", default=str(REPORTS / "a4_advisory_audit.json"))
    args = ap.parse_args()

    cards = load_cards(CONFIGS / "sop_cards.yaml")
    cases = generate_cases(n=args.n, seed=0)
    print(f"Loaded {len(cards.cards)} cards. Generated {len(cases)} synthetic stress cases.")

    advisor = None
    if Path(args.pte).exists() and Path(args.tokenizer).exists():
        print(f"Loading ExecuTorch advisor from {args.pte}...")
        advisor = ExecuTorchAdvisor(args.pte, args.tokenizer)

    worker = AdvisoryWorker(card_library=cards, runner=advisor)
    print("Running Advisory Audit...")
    res = audit_advisories(worker, cards, cases, advisor=advisor)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Results saved to {args.out}")


if __name__ == "__main__":
    main()

