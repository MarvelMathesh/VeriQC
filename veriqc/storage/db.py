"""Tamper-evident SQLite audit storage with rolling SHA-256 hash chains.

Stores inspection decisions, advisory outputs, model metadata, and latency metrics
in an on-device SQLite database configured in WAL mode.
Provides tamper evidence by linking each audit log entry to the hash of the
previous entry in a cryptographic hash chain. Supports complete JSONL audit export.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from veriqc.advisory.worker import AdvisoryOutcome
from veriqc.contract.record import DefectRecord

__all__ = ["InspectionDatabase", "AuditEntry"]

GENESIS_HASH = "0" * 64


@dataclass
class AuditEntry:
    entry_id: int
    part_id: str
    verdict: str
    timestamp: float
    record_hash: str
    record_json: str
    advisory_text: str
    advisory_path: str
    advisory_verified: bool
    prev_hash: str
    curr_hash: str
    latency_ms_json: str


class InspectionDatabase:
    """Production SQLite storage with WAL mode and cryptographic chain."""

    def __init__(self, db_path: str | Path = "veriqc_audit.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    part_id TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    record_hash TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    advisory_text TEXT,
                    advisory_path TEXT,
                    advisory_verified INTEGER,
                    prev_hash TEXT NOT NULL,
                    curr_hash TEXT NOT NULL,
                    latency_ms_json TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_part_id ON audit_log (part_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_verdict ON audit_log (verdict)")

    def get_last_hash(self) -> str:
        """Fetch the cryptographic hash of the most recent audit entry."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT curr_hash FROM audit_log ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            return row["curr_hash"] if row else GENESIS_HASH

    def log_inspection(
        self,
        record: DefectRecord,
        advisory: AdvisoryOutcome | None = None,
    ) -> AuditEntry:
        """Log an inspection record and optional advisory into the tamper-evident chain."""
        with self._get_connection() as conn:
            prev_hash = self.get_last_hash()
            record_json = record.to_json()
            record_hash = record.content_hash()

            adv_text = advisory.text if advisory else ""
            adv_path = advisory.path if advisory else "none"
            adv_verified = 1 if (advisory and advisory.verified) else 0

            # Cryptographic block hash: H(prev_hash || record_hash || advisory_text)
            chain_input = f"{prev_hash}:{record_hash}:{adv_text}".encode("utf-8")
            curr_hash = hashlib.sha256(chain_input).hexdigest()

            cursor = conn.execute(
                """
                INSERT INTO audit_log (
                    part_id, verdict, timestamp, record_hash, record_json,
                    advisory_text, advisory_path, advisory_verified,
                    prev_hash, curr_hash, latency_ms_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.part_id,
                    record.verdict,
                    record.timestamp,
                    record_hash,
                    record_json,
                    adv_text,
                    adv_path,
                    adv_verified,
                    prev_hash,
                    curr_hash,
                    json.dumps(record.latency_ms),
                ),
            )
            entry_id = cursor.lastrowid

            return AuditEntry(
                entry_id=entry_id,
                part_id=record.part_id,
                verdict=record.verdict,
                timestamp=record.timestamp,
                record_hash=record_hash,
                record_json=record_json,
                advisory_text=adv_text,
                advisory_path=adv_path,
                advisory_verified=bool(adv_verified),
                prev_hash=prev_hash,
                curr_hash=curr_hash,
                latency_ms_json=json.dumps(record.latency_ms),
            )

    def verify_integrity(self) -> tuple[bool, int, str | None]:
        """Verify the integrity of the entire cryptographic chain."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM audit_log ORDER BY id ASC")
            rows = cursor.fetchall()

            expected_prev = GENESIS_HASH
            for row in rows:
                if row["prev_hash"] != expected_prev:
                    return False, row["id"], f"Prev hash mismatch at row {row['id']}"

                chain_input = f"{row['prev_hash']}:{row['record_hash']}:{row['advisory_text']}".encode("utf-8")
                computed_curr = hashlib.sha256(chain_input).hexdigest()
                if row["curr_hash"] != computed_curr:
                    return False, row["id"], f"Curr hash corruption at row {row['id']}"

                expected_prev = row["curr_hash"]

            return True, len(rows), None

    def export_jsonl(self, out_path: str | Path) -> int:
        """Export all audit records to JSONL."""
        path = Path(out_path)
        count = 0
        with self._get_connection() as conn, open(path, "w", encoding="utf-8") as f:
            cursor = conn.execute("SELECT * FROM audit_log ORDER BY id ASC")
            for row in cursor:
                f.write(json.dumps(dict(row)) + "\n")
                count += 1
        return count

