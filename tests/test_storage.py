from __future__ import annotations

import json
import sqlite3

from veriqc.inspection import InspectionResult
from veriqc.audit import AuditStore


def result(identifier: str = "a") -> InspectionResult:
    return InspectionResult(
        inspection_id=identifier,
        timestamp=1.0,
        part_id="P1",
        profile_id="p",
        verdict="FAIL",
    )


def test_chain_verifies_and_detects_tampering(tmp_path) -> None:
    store = AuditStore(tmp_path / "audit.db")
    store.append(result("a"))
    store.append(result("b"), event="review", extra={"label": "PASS"})
    assert store.verify()["ok"] is True
    assert len(store.history()) == 2
    assert store.get("a") is not None

    connection = sqlite3.connect(tmp_path / "audit.db")
    connection.execute("UPDATE events SET payload_json='{}' WHERE id=1")
    connection.commit()
    connection.close()
    assert store.verify()["ok"] is False


def test_missing_anchor_is_reported(tmp_path) -> None:
    store = AuditStore(tmp_path / "audit.db")
    store.append(result())
    store.anchor.unlink()
    report = store.verify()
    assert report["ok"] is False
    assert report["anchor_ok"] is False


def test_export(tmp_path) -> None:
    store = AuditStore(tmp_path / "audit.db")
    store.append(result())
    target = store.export_csv(tmp_path / "out.csv")
    assert target.exists()
    assert "inspection_id" in target.read_text()


def test_anchor_records_head(tmp_path) -> None:
    store = AuditStore(tmp_path / "audit.db")
    store.append(result())
    anchor = json.loads((tmp_path / "audit.head.json").read_text())
    assert anchor["count"] == 1
    assert len(anchor["head"]) == 64
