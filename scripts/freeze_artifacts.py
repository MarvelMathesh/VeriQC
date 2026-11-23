#!/usr/bin/env python3
"""Freeze expected runtime artifact hashes for the station.

The profile YAML declares paths; this manifest records the SHA-256 of every
runtime artifact.  ``load_profile(validate=True)`` refuses a missing or
modified artifact.  Re-run this command after any deliberate retrain or
calibration update.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from veriqc import sha256_file  # noqa: E402
from veriqc.config import list_profiles, load_profile  # noqa: E402

OUTPUT = REPO / "configs" / "artifacts.json"
ADVISORY_MANIFEST = REPO / "models" / "advisory" / "manifest.json"
ADVISORY_PTE = REPO / "models" / "advisory" / "llama32_1b_int8.pte"
ADVISORY_TOKENIZER = REPO / "models" / "advisory" / "tokenizer" / "tokenizer.json"
PUBLISHED_ADVISORY = REPO / "reports" / "models" / "advisory.json"


def main() -> None:
    profiles = {}
    for path in list_profiles():
        profile = load_profile(path, validate=False)
        profiles[profile.profile_id] = {
            name: sha256_file(artifact_path)
            for name, artifact_path in profile.artifact_paths().items()
        }
    payload = {
        "schema": "veriqc.artifacts/1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "note": "Runtime artifact hashes. Re-run scripts/freeze_artifacts.py after a deliberate update.",
        "profiles": profiles,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(OUTPUT.relative_to(REPO))

    if ADVISORY_MANIFEST.exists() and ADVISORY_PTE.exists() and ADVISORY_TOKENIZER.exists():
        source = json.loads(ADVISORY_MANIFEST.read_text())
        published = {
            "artefact": str(ADVISORY_PTE.relative_to(REPO)),
            "bytes": ADVISORY_PTE.stat().st_size,
            "sha256": sha256_file(ADVISORY_PTE),
            "quant": source.get("quant"),
            "dtype_override": source.get("dtype_override"),
            "max_seq_length": source.get("max_seq_length"),
            "tokenizer": str(ADVISORY_TOKENIZER.relative_to(REPO)),
            "tokenizer_sha256": sha256_file(ADVISORY_TOKENIZER),
            "tokenizer_files": source.get("tokenizer_files"),
            "tokenizer_bundle_sha256": source.get("tokenizer_bundle_sha256"),
            "source_checkpoint": source.get("source_checkpoint"),
            "runtime": source.get("runtime"),
            "note": "The advisory runs in an isolated child process and cannot change the verdict.",
        }
        PUBLISHED_ADVISORY.parent.mkdir(parents=True, exist_ok=True)
        PUBLISHED_ADVISORY.write_text(json.dumps(published, indent=2) + "\n")
        print(PUBLISHED_ADVISORY.relative_to(REPO))


if __name__ == "__main__":
    main()
