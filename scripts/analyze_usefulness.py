#!/usr/bin/env python3
"""Analyse a completed blind usefulness sheet.

The unit is one case, pooled over raters.  The script refuses incomplete
sheets rather than treating blanks as zero.  It reports the paired difference
(advisory minus template), a paired t interval, a Wilcoxon signed-rank test
and the preference counts.  Only an independent human rater can supply the
input; the script never invents ratings.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path

from scipy import stats

REPO = Path(__file__).resolve().parents[1]
SHEET = REPO / "reports" / "tables" / "advisory_usefulness_sheet.csv"
KEY = REPO / "reports" / "tables" / "advisory_usefulness_key.csv"
OUT = REPO / "reports" / "tables" / "advisory_usefulness.json"
SCORE_COLUMNS = (
    "usefulness_a_1_5",
    "usefulness_b_1_5",
    "actionability_a_1_5",
    "actionability_b_1_5",
)


def _score(value: str, column: str, line: int) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"line {line}: {column} must be an integer 1-5, got {value!r}") from exc
    if not 1 <= score <= 5:
        raise SystemExit(f"line {line}: {column} must be 1-5, got {score}")
    return score


def _mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 3) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sheet", default=str(SHEET))
    parser.add_argument("--key", default=str(KEY))
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    sheet_path = Path(args.sheet)
    key_path = Path(args.key)
    if not sheet_path.exists() or not key_path.exists():
        raise SystemExit("build the blind sheet first with scripts/build_usefulness_sheet.py")
    key = {row["case_id"]: row for row in csv.DictReader(key_path.open(newline=""))}
    rows = list(csv.DictReader(sheet_path.open(newline="")))
    if not rows:
        raise SystemExit("the sheet is empty")

    raters = {row["rater_id"].strip() for row in rows}
    raters.discard("")
    if not raters:
        raise SystemExit(
            "the sheet is unrated: fill rater_id and every score, then rerun. "
            "No usefulness result is recorded without an independent rater."
        )

    paired = []
    for line, row in enumerate(rows, start=2):
        if not row["rater_id"].strip():
            raise SystemExit(f"line {line}: rater_id is blank")
        mapping = key[row["case_id"]]
        scores = {
            column: _score(row[column], column, line)
            for column in SCORE_COLUMNS
        }
        usefulness = {
            "template": scores[
                "usefulness_a_1_5"
                if mapping["option_a_source"] == "template"
                else "usefulness_b_1_5"
            ],
            "advisory": scores[
                "usefulness_a_1_5"
                if mapping["option_a_source"] == "advisory"
                else "usefulness_b_1_5"
            ],
        }
        actionability = {
            "template": scores[
                "actionability_a_1_5"
                if mapping["option_a_source"] == "template"
                else "actionability_b_1_5"
            ],
            "advisory": scores[
                "actionability_a_1_5"
                if mapping["option_a_source"] == "advisory"
                else "actionability_b_1_5"
            ],
        }
        preference = row["preference"].strip().lower()
        if preference not in ("a", "b", "equal"):
            raise SystemExit(
                f"line {line}: preference must be A, B or equal, got {row['preference']!r}"
            )
        if preference == "equal":
            preferred = "equal"
        else:
            preferred = (
                mapping["option_a_source"] if preference == "a" else mapping["option_b_source"]
            )
        paired.append(
            {
                "case_id": row["case_id"],
                "rater_id": row["rater_id"].strip(),
                "profile": row["profile"],
                "usefulness_template": usefulness["template"],
                "usefulness_advisory": usefulness["advisory"],
                "actionability_template": actionability["template"],
                "actionability_advisory": actionability["advisory"],
                "preferred": preferred,
                "grounding_error": row["grounding_error"].strip().lower() or "not recorded",
                "comments": row["comments"].strip(),
            }
        )

    usefulness_delta = [
        row["usefulness_advisory"] - row["usefulness_template"] for row in paired
    ]
    action_delta = [
        row["actionability_advisory"] - row["actionability_template"] for row in paired
    ]
    n = len(paired)
    mean_delta = statistics.mean(usefulness_delta)
    if n >= 2:
        standard_error = statistics.stdev(usefulness_delta) / math.sqrt(n)
    else:
        standard_error = 0.0
    if n >= 2 and standard_error > 0:
        critical = float(stats.t.ppf(0.975, n - 1))
        interval = [
            round(mean_delta - critical * standard_error, 3),
            round(mean_delta + critical * standard_error, 3),
        ]
    else:
        interval = [round(mean_delta, 3), round(mean_delta, 3)]
    if any(usefulness_delta) and n >= 5:
        wilcoxon = stats.wilcoxon(
            [row["usefulness_advisory"] for row in paired],
            [row["usefulness_template"] for row in paired],
            zero_method="wilcox",
        )
        p_value = round(float(wilcoxon.pvalue), 6)
    else:
        p_value = None
    preference_counts = Counter(row["preferred"] for row in paired)
    grounding_counts = Counter(row["grounding_error"] for row in paired)
    payload = {
        "schema": "veriqc.advisory-usefulness/1",
        "measured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "measured",
        "raters": sorted(raters),
        "cases": n,
        "scoring": "1-5 usefulness and actionability, blind paired template vs verified advisory",
        "usefulness": {
            "template_mean": _mean([row["usefulness_template"] for row in paired]),
            "advisory_mean": _mean([row["usefulness_advisory"] for row in paired]),
            "paired_delta_mean": round(mean_delta, 3),
            "paired_delta_ci95": interval,
            "paired_delta_median": round(statistics.median(usefulness_delta), 3),
            "wilcoxon_p": p_value,
            "advisory_better": sum(delta > 0 for delta in usefulness_delta),
            "equal": sum(delta == 0 for delta in usefulness_delta),
            "template_better": sum(delta < 0 for delta in usefulness_delta),
        },
        "actionability": {
            "template_mean": _mean([row["actionability_template"] for row in paired]),
            "advisory_mean": _mean([row["actionability_advisory"] for row in paired]),
            "paired_delta_mean": _mean(action_delta),
        },
        "preference": dict(sorted(preference_counts.items())),
        "grounding_error": dict(sorted(grounding_counts.items())),
        "rows": paired,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
