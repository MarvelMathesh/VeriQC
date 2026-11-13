"""Dataset adapters and deterministic split manifests.

Every experiment in this project is reproducible because the split is written
to a manifest together with a content hash of each file.  If a file changes,
the hash changes and the run is no longer comparable.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

__all__ = [
    "file_sha1",
    "write_manifest",
    "read_manifest",
    "split_list",
]

DATA_ROOT = Path(os.environ.get("VERIQC_DATA", Path(__file__).resolve().parents[2] / "data"))


def file_sha1(path: str | Path, chunk: int = 1 << 20) -> str:
    """SHA-1 of a file (used to detect silent dataset changes)."""
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def split_list(items: list, fractions: tuple[float, ...], seed: int = 0) -> list[list]:
    """Deterministic split of a list into disjoint parts.

    Uses a seeded permutation, so the same input list and seed always give the
    same split, and the parts never overlap.
    """
    import random

    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1")
    idx = list(range(len(items)))
    random.Random(seed).shuffle(idx)
    out, start = [], 0
    for i, f in enumerate(fractions):
        if i == len(fractions) - 1:
            end = len(items)
        else:
            end = start + int(round(f * len(items)))
        out.append([items[j] for j in sorted(idx[start:end])])
        start = end
    return out


def write_manifest(path: str | Path, payload: dict, hashes: dict[str, str] | None = None) -> Path:
    """Write a JSON manifest, optionally with a per-file sha1 map."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    if hashes:
        body["_file_sha1"] = hashes
    p.write_text(json.dumps(body, indent=2, sort_keys=True, default=str))
    return p


def read_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
