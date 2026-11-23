from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXCLUDED = {".git", "venv", ".venv", "data", "models", "__pycache__", ".pytest_cache", "artifacts"}


def test_exactly_two_markdown_documents() -> None:
    documents = sorted(
        path.relative_to(REPO)
        for path in REPO.rglob("*.md")
        if not any(part in EXCLUDED for part in path.parts)
    )
    assert documents == [Path("README.md"), Path("RESEARCH.md")]


def test_results_evidence_paths_exist() -> None:
    results = REPO / "reports" / "results.json"
    if not results.exists():
        return
    payload = json.loads(results.read_text())
    for record in payload["profiles"].values():
        for evidence in record["evidence"]:
            assert (REPO / evidence).exists(), evidence
    for key in ("advisory", "export_parity"):
        evidence = payload[key].get("evidence")
        if evidence:
            assert (REPO / evidence).exists(), evidence


def test_pi_benchmarks_are_labeled_as_pi() -> None:
    for mode in ("vision", "advisory", "resident"):
        path = REPO / "reports" / "tables" / f"pi_{mode}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        assert payload["mode"] == mode
        assert payload["environment"]["machine"] == "aarch64"
        assert "Raspberry Pi 5" in (payload["environment"]["model"] or "")
        assert payload["environment"]["memory_total_mb"] > 7000


def test_pi_operating_point_matches_host() -> None:
    host = REPO / "reports" / "tables" / "evaluation_commutator.json"
    pi = REPO / "reports" / "tables" / "pi_evaluation_commutator.json"
    if not host.exists() or not pi.exists():
        return
    host_payload = json.loads(host.read_text())
    pi_payload = json.loads(pi.read_text())
    assert host_payload["verdict_counts"] == pi_payload["verdict_counts"]
    assert host_payload["confusion"] == pi_payload["confusion"]


def test_usefulness_instrument_has_no_fabricated_scores() -> None:
    sheet = REPO / "reports" / "tables" / "advisory_usefulness_sheet.csv"
    if not sheet.exists():
        return
    import csv

    rows = list(csv.DictReader(sheet.open(newline="")))
    assert len(rows) == 37
    assert all(row["rater_id"] == "" for row in rows)
    assert all(row["usefulness_a_1_5"] == "" for row in rows)


def test_runtime_artifact_manifest_is_complete() -> None:
    manifest = REPO / "configs" / "artifacts.json"
    if not manifest.exists():
        return
    payload = json.loads(manifest.read_text())
    for profile in payload["profiles"].values():
        assert {"profile", "calibration", "severity", "cards"} <= set(profile)
