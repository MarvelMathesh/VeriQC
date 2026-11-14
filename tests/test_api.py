"""Integration tests for the VeriQC Station FastAPI application."""

from __future__ import annotations

import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np

from veriqc.station.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_operator_dashboard_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "VeriQC Edge Station" in response.text
    assert "Station Verdict" in response.text
    assert "On-Device Advisory" in response.text


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_inspections" in data
    assert "pass_count" in data


def test_history_endpoint(client):
    response = client.get("/history?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert "integrity_verified" in data
    assert "entries" in data
    assert data["integrity_verified"] is True


def test_inspect_endpoint(client):
    # Create synthetic test image
    img = Image.fromarray(np.full((128, 128, 3), 120, dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/inspect?part_id=P-TEST-001",
        files={"file": ("test.jpg", buf, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["part_id"] == "P-TEST-001"
    assert data["verdict"] in ("PASS", "REVIEW", "FAIL")
    assert "latency_ms" in data
    assert "total_ms" in data["latency_ms"]

