from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

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


def test_concurrent_appends_keep_anchor_current(tmp_path) -> None:
    store = AuditStore(tmp_path / "audit.db")

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda index: store.append(result(f"c{index}")), range(32)))

    report = store.verify()
    assert report["ok"] is True
    assert report["chain_ok"] is True
    assert report["anchor_ok"] is True
    assert report["rows"] == 32
    anchor = json.loads(store.anchor.read_text())
    assert anchor["count"] == 32


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


def test_advisory_outcomes_are_counted(tmp_path) -> None:
    store = AuditStore(tmp_path / "audit.db")
    store.append(
        result(),
        event="advisory",
        extra={"path": "llm", "accepted": True, "verification": {"violations": []}},
    )
    store.append(
        result("b"),
        event="advisory",
        extra={
            "path": "template",
            "accepted": False,
            "verification": {"violations": ["generation timeout or busy"]},
        },
    )
    assert store.verdict_counts() == {}
    outcomes = store.advisory_outcomes()
    assert outcomes["events"] == 2
    assert outcomes["paths"] == {"llm": 1, "template": 1}
    assert outcomes["accepted"] == {"true": 1, "false": 1}
    assert outcomes["violations"] == {"generation timeout or busy": 1}


def test_anchor_records_head(tmp_path) -> None:
    store = AuditStore(tmp_path / "audit.db")
    store.append(result())
    anchor = json.loads((tmp_path / "audit.head.json").read_text())
    assert anchor["count"] == 1
    assert len(anchor["head"]) == 64


def test_recent_context_counts_verdicts_and_findings(tmp_path) -> None:
    store = AuditStore(tmp_path / "audit.db")
    item = result("a")
    item.verdict = "FAIL"
    from veriqc.inspection import Finding

    item.findings = [
        Finding(
            finding_id="f",
            source="known",
            defect_class="crack",
            severity="major",
        )
    ]
    store.append(item)
    context = store.recent_context(10)
    assert context["window"] == 1
    assert context["verdicts"]["FAIL"] == 1
    assert context["defects"]["crack"] == 1
