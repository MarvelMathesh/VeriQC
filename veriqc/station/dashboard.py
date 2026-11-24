"""Operator dashboard served locally with no external assets."""

from __future__ import annotations

from pathlib import Path

__all__ = ["DASHBOARD_HTML"]

DASHBOARD_HTML = Path(__file__).with_name("dashboard.html").read_text(encoding="utf-8")
