"""Unit tests for the tamper-evident SQLite storage."""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest

from veriqc.contract.record import DefectRecord
from veriqc.storage.db import InspectionDatabase


def test_storage_cryptographic_chain():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_audit.db"
        db = InspectionDatabase(db_path)

        # Integrity on empty DB
        ok, n, err = db.verify_integrity()
        assert ok is True
        assert n == 0

        # Log entry 1
        r1 = DefectRecord(part_id="P-101", verdict="PASS", timestamp=1.0)
        e1 = db.log_inspection(r1, None)
        assert e1.prev_hash == "0" * 64
        assert len(e1.curr_hash) == 64

        # Log entry 2
        r2 = DefectRecord(part_id="P-102", verdict="FAIL", timestamp=2.0)
        e2 = db.log_inspection(r2, None)
        assert e2.prev_hash == e1.curr_hash

        # Verify integrity
        ok, count, err = db.verify_integrity()
        assert ok is True
        assert count == 2

        # Export JSONL
        jsonl_path = Path(tmpdir) / "audit.jsonl"
        exported = db.export_jsonl(jsonl_path)
        assert exported == 2
        assert jsonl_path.exists()

