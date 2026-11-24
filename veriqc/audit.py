"""Append-only SQLite audit log with a SHA-256 hash chain."""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import threading
import time
from contextlib import closing
from pathlib import Path
from typing import Any

from veriqc.inspection import InspectionResult

__all__ = ["AuditStore", "GENESIS"]


GENESIS = "0" * 64


class AuditStore:
    def __init__(self, path: str | Path = "veriqc_audit.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.anchor = self.path.with_suffix(".head.json")
        self._lock = threading.Lock()
        self._init()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        connection.row_factory = sqlite3.Row
        return connection

    def _init(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    inspection_id TEXT NOT NULL,
                    part_id TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    event TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    payload_json TEXT NOT NULL,
                    prev_hash TEXT NOT NULL,
                    curr_hash TEXT NOT NULL
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS ix_events_inspection ON events(inspection_id)")

    def _head(self, connection: sqlite3.Connection) -> tuple[str, int]:
        row = connection.execute(
            "SELECT id, curr_hash FROM events ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return GENESIS, 0
        # The table is append-only, so the last row id is the event count.
        return str(row["curr_hash"]), int(row["id"])

    def append(
        self,
        result: InspectionResult,
        event: str = "verdict",
        extra: dict[str, Any] | None = None,
    ) -> str:
        payload = {"result": result.to_dict(include_record_hash=True), "extra": extra or {}}
        payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        with self._lock:
            with closing(self._connect()) as connection:
                prev_hash, _ = self._head(connection)
                timestamp = time.time()
                material = (
                    f"{prev_hash}|{event}|{result.inspection_id}|{timestamp:.6f}|{payload_json}"
                )
                current = hashlib.sha256(material.encode("utf-8")).hexdigest()
                connection.execute("BEGIN IMMEDIATE")
                cursor = connection.execute(
                    """
                    INSERT INTO events
                        (inspection_id, part_id, verdict, event, timestamp, payload_json, prev_hash, curr_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        result.inspection_id,
                        result.part_id,
                        result.verdict,
                        event,
                        timestamp,
                        payload_json,
                        prev_hash,
                        current,
                    ),
                )
                row_id = int(cursor.lastrowid)
                connection.execute("COMMIT")
            # Keep the lock through the anchor write so two appends cannot
            # publish an older head after a newer one.
            self._write_anchor({"head": current, "count": row_id})
        return current

    def _write_anchor(self, payload: dict[str, Any]) -> None:
        temporary = self.anchor.with_suffix(self.anchor.suffix + ".tmp")
        temporary.write_text(json.dumps(payload) + "\n")
        temporary.replace(self.anchor)

    def history(self, limit: int = 25) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT id, inspection_id, part_id, verdict, event, timestamp, curr_hash "
                "FROM events ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [dict(row) for row in rows]

    def get(self, inspection_id: str) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT payload_json FROM events WHERE inspection_id = ? AND event = 'verdict' "
                "ORDER BY id ASC LIMIT 1",
                (inspection_id,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row["payload_json"])["result"]

    def latest_event(self, inspection_id: str, event: str) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT payload_json FROM events WHERE inspection_id = ? AND event = ? "
                "ORDER BY id DESC LIMIT 1",
                (inspection_id, event),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row["payload_json"])

    def verdict_counts(self) -> dict[str, int]:
        """Count stored verdict events by label."""

        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT verdict, COUNT(*) FROM events WHERE event = 'verdict' "
                "GROUP BY verdict ORDER BY verdict"
            ).fetchall()
        return {str(row[0]): int(row[1]) for row in rows}

    def advisory_outcomes(self) -> dict[str, Any]:
        """Count advisory event outcomes without exposing prompt text."""

        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT payload_json FROM events WHERE event = 'advisory' ORDER BY id ASC"
            ).fetchall()
        paths: dict[str, int] = {}
        accepted = {True: 0, False: 0}
        violations: dict[str, int] = {}
        for row in rows:
            payload = json.loads(row["payload_json"])
            extra = payload.get("extra") or {}
            path = str(extra.get("path") or "unknown")
            paths[path] = paths.get(path, 0) + 1
            accepted[bool(extra.get("accepted"))] = accepted.get(bool(extra.get("accepted")), 0) + 1
            for violation in (extra.get("verification") or {}).get("violations", []):
                violations[str(violation)] = violations.get(str(violation), 0) + 1
        return {
            "events": len(rows),
            "paths": dict(sorted(paths.items())),
            "accepted": {
                "true": accepted.get(True, 0),
                "false": accepted.get(False, 0),
            },
            "violations": dict(sorted(violations.items())),
        }

    def recent_context(self, window: int = 50) -> dict[str, Any]:
        """Aggregate recent verdicts for the advisory, without part identifiers."""

        limit = max(1, min(int(window), 500))
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT verdict, payload_json FROM events WHERE event = 'verdict' "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        verdicts = {"PASS": 0, "REVIEW": 0, "FAIL": 0}
        defects: dict[str, int] = {}
        for row in rows:
            verdict = str(row["verdict"])
            verdicts[verdict] = verdicts.get(verdict, 0) + 1
            try:
                payload = json.loads(row["payload_json"])["result"]
            except Exception:
                continue
            for finding in payload.get("findings", []):
                name = str(finding.get("defect_class") or "unknown")
                defects[name] = defects.get(name, 0) + 1
        return {
            "window": len(rows),
            "verdicts": verdicts,
            "defects": dict(sorted(defects.items(), key=lambda item: item[1], reverse=True)),
        }

    def verify(self) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            rows = connection.execute("SELECT * FROM events ORDER BY id ASC").fetchall()
        prev = GENESIS
        for row in rows:
            material = (
                f"{prev}|{row['event']}|{row['inspection_id']}|{row['timestamp']:.6f}|"
                f"{row['payload_json']}"
            )
            expected = hashlib.sha256(material.encode("utf-8")).hexdigest()
            if row["prev_hash"] != prev or row["curr_hash"] != expected:
                return {
                    "ok": False,
                    "chain_ok": False,
                    "anchor_ok": False,
                    "rows": len(rows),
                    "first_bad": int(row["id"]),
                }
            try:
                payload = json.loads(row["payload_json"])
                record = InspectionResult.from_dict(payload["result"])
                stored_record_hash = str(payload["result"]["record_hash"])
            except Exception:
                return {
                    "ok": False,
                    "chain_ok": False,
                    "anchor_ok": False,
                    "rows": len(rows),
                    "first_bad": int(row["id"]),
                }
            if record.inspection_id != row["inspection_id"] or record.record_hash() != stored_record_hash:
                return {
                    "ok": False,
                    "chain_ok": False,
                    "anchor_ok": False,
                    "rows": len(rows),
                    "first_bad": int(row["id"]),
                }
            prev = row["curr_hash"]

        anchor_ok = True
        if rows:
            if not self.anchor.exists():
                anchor_ok = False
            else:
                try:
                    anchor = json.loads(self.anchor.read_text())
                    anchor_ok = (
                        anchor.get("head") == prev
                        and int(anchor.get("count", len(rows))) == len(rows)
                    )
                except Exception:
                    anchor_ok = False
        return {
            "ok": anchor_ok,
            "chain_ok": True,
            "anchor_ok": anchor_ok,
            "rows": len(rows),
            "first_bad": None,
        }

    def export_csv(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            rows = connection.execute("SELECT * FROM events ORDER BY id ASC").fetchall()
        with open(target, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["id", "inspection_id", "part_id", "verdict", "event", "timestamp", "curr_hash"]
            )
            for row in rows:
                writer.writerow(
                    [
                        row["id"],
                        row["inspection_id"],
                        row["part_id"],
                        row["verdict"],
                        row["event"],
                        f"{row['timestamp']:.6f}",
                        row["curr_hash"],
                    ]
                )
        return target
