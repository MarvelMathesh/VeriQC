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


def test_runtime_artifact_manifest_is_complete() -> None:
    manifest = REPO / "configs" / "artifacts.json"
    if not manifest.exists():
        return
    payload = json.loads(manifest.read_text())
    for profile in payload["profiles"].values():
        assert {"profile", "calibration", "severity", "cards"} <= set(profile)
