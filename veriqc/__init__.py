"""VeriQC: risk-calibrated edge inspection with a verified on-device advisory."""

from __future__ import annotations

from pathlib import Path

__version__ = "2.0.0"

REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(value: str | Path) -> Path:
    """Resolve a repository-relative path without accepting escapes."""
    path = Path(value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def sha256_file(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
