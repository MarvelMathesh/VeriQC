#!/usr/bin/env python3
"""Build a blind, paired rating sheet for advisory usefulness.

Each case shows the deterministic template and the verified advisory text as
anonymous options A and B.  The assignment is randomised per case and stored
in a separate key, so a rater cannot tell which arm is the model.  The rater
scores both options on usefulness and actionability, records any grounding
error, and states a preference.

The sheet is an instrument, not a result.  Use ``analyze_usefulness.py`` on a
completed sheet to produce the measured report.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / "reports" / "tables" / "advisory_audit.json"
SHEET = REPO / "reports" / "tables" / "advisory_usefulness_sheet.csv"
KEY = REPO / "reports" / "tables" / "advisory_usefulness_key.csv"

COLUMNS = [
    "case_id",
    "profile",
    "verdict",
    "defect_class",
    "severity",
    "option_a",
    "option_b",
    "rater_id",
    "usefulness_a_1_5",
    "usefulness_b_1_5",
    "actionability_a_1_5",
    "actionability_b_1_5",
    "preference",
    "grounding_error",
    "comments",
]


def _one_line(text: str) -> str:
    return " ".join(str(text).split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", default=str(AUDIT))
    parser.add_argument("--sheet", default=str(SHEET))
    parser.add_argument("--key", default=str(KEY))
    parser.add_argument("--seed", type=int, default=20261122)
    args = parser.parse_args()

    audit = json.loads(Path(args.audit).read_text())
    rows = [row for row in audit["rows"] if row.get("accepted")]
    if not rows:
        raise SystemExit("the audit has no accepted advisories to rate")
    rng = random.Random(args.seed)
    sheet_rows = []
    key_rows = []
    for index, row in enumerate(rows, start=1):
        case_id = f"case-{index:02d}"
        advisory_first = rng.random() < 0.5
        advisory = _one_line(row["text"])
        template = _one_line(row["template"])
        option_a = advisory if advisory_first else template
        option_b = template if advisory_first else advisory
        sheet_rows.append(
            {
                "case_id": case_id,
                "profile": row["profile_id"],
                "verdict": row["verdict"],
                "defect_class": row.get("defect_class") or "",
                "severity": row.get("severity") or "",
                "option_a": option_a,
                "option_b": option_b,
                "rater_id": "",
                "usefulness_a_1_5": "",
                "usefulness_b_1_5": "",
                "actionability_a_1_5": "",
                "actionability_b_1_5": "",
                "preference": "",
                "grounding_error": "",
                "comments": "",
            }
        )
        key_rows.append(
            {
                "case_id": case_id,
                "option_a_source": "advisory" if advisory_first else "template",
                "option_b_source": "template" if advisory_first else "advisory",
            }
        )

    sheet_path = Path(args.sheet)
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    with sheet_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(sheet_rows)
    key_path = Path(args.key)
    with key_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["case_id", "option_a_source", "option_b_source"]
        )
        writer.writeheader()
        writer.writerows(key_rows)
    print(f"{len(sheet_rows)} cases -> {sheet_path}")
    print(f"blind assignment -> {key_path}")
    print("Give the rater the sheet only; keep the key away from the rater.")


if __name__ == "__main__":
    main()
