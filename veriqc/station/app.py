"""FastAPI station: one vision path, one isolated advisory runtime, one audit log."""

from __future__ import annotations

import asyncio
import hmac
import io
import json
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from typing import Any

import yaml
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from PIL import Image
from starlette.concurrency import run_in_threadpool

from veriqc import REPO_ROOT, __version__
from veriqc.advisory.worker import AdvisoryOutcome
from veriqc.inspection import InspectionResult
from veriqc.station.dashboard import DASHBOARD_HTML
from veriqc.station.service import Station

__all__ = ["app"]

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
PART_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
PROFILE_ID_RE = re.compile(r"^[a-z0-9_]{1,64}$")
DEFAULT_PROFILE = os.environ.get("VERIQC_PROFILE", "commutator")
DISABLE_ADVISORY = os.environ.get("VERIQC_DISABLE_ADVISORY", "0") == "1"


class Manager:
    def __init__(self) -> None:
        self.station: Station | None = None
        self.error: str | None = None
        self.lock = threading.Lock()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.connections: list[WebSocket] = []

    def activate(self, profile_id: str) -> Station:
        with self.lock:
            if self.station is not None:
                self.station.close()
                self.station = None
            self.error = None
            try:
                self.station = Station(
                    profile_id,
                    enable_advisory=not DISABLE_ADVISORY,
                    on_advisory=self.broadcast_advisory,
                )
            except Exception as exc:
                self.error = repr(exc)
                raise
            return self.station

    def broadcast_advisory(self, result: InspectionResult, outcome: AdvisoryOutcome) -> None:
        if self.loop is None:
            return
        message = {"type": "advisory", "result": result.to_dict()}
        try:
            asyncio.run_coroutine_threadsafe(self.broadcast(message), self.loop)
        except RuntimeError:
            # The event loop can close while the advisory worker is finishing.
            self.loop = None

    async def broadcast(self, message: dict[str, Any]) -> None:
        for connection in list(self.connections):
            try:
                await connection.send_json(message)
            except Exception:
                if connection in self.connections:
                    self.connections.remove(connection)

    def close(self) -> None:
        with self.lock:
            if self.station is not None:
                self.station.close()
                self.station = None


manager = Manager()


def _require_token(x_veriqc_token: str | None = Header(default=None)) -> None:
    expected = os.environ.get("VERIQC_API_TOKEN")
    if not expected:
        return
    if not hmac.compare_digest(x_veriqc_token or "", expected):
        raise HTTPException(status_code=401, detail="invalid station token")


def _token_ok(token: str | None) -> bool:
    expected = os.environ.get("VERIQC_API_TOKEN")
    if not expected:
        return True
    return hmac.compare_digest(token or "", expected)


@asynccontextmanager
async def lifespan(_: FastAPI):
    manager.loop = asyncio.get_running_loop()
    try:
        await run_in_threadpool(manager.activate, DEFAULT_PROFILE)
    except Exception as exc:
        manager.error = repr(exc)
    source_config = os.environ.get("VERIQC_SOURCE_CONFIG")
    if source_config and manager.station is not None:
        try:
            await run_in_threadpool(manager.station.start_capture, json.loads(source_config))
        except Exception as exc:
            manager.error = f"source startup failed: {exc!r}"
    yield
    manager.close()


app = FastAPI(title="VeriQC Edge Station", version=__version__, lifespan=lifespan)


def _station() -> Station:
    if manager.station is None:
        raise HTTPException(status_code=503, detail=manager.error or "station is not available")
    return manager.station


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return DASHBOARD_HTML


@app.get("/health")
def health() -> dict[str, Any]:
    station = manager.station
    return {
        "ok": station is not None,
        "error": manager.error,
        "profile": station.profile_id if station is not None else None,
        "capture": station.capture_status() if station is not None else None,
    }


@app.get("/profiles")
def profiles(_: None = Depends(_require_token)) -> dict[str, Any]:
    rows = []
    for path in sorted((REPO_ROOT / "configs" / "profiles").glob("*.yaml")):
        data = yaml.safe_load(path.read_text()) or {}
        rows.append(
            {
                "id": data.get("id", path.stem),
                "domain": data.get("domain", ""),
                "task": data.get("task", ""),
                "description": data.get("description", ""),
            }
        )
    return {"active": manager.station.profile_id if manager.station else None, "profiles": rows}


@app.post("/profiles/{profile_id}/activate")
async def activate(profile_id: str, _: None = Depends(_require_token)) -> dict[str, Any]:
    if not PROFILE_ID_RE.fullmatch(profile_id):
        raise HTTPException(status_code=404, detail="unknown profile")
    profile_path = (REPO_ROOT / "configs" / "profiles" / f"{profile_id}.yaml").resolve()
    if profile_path.parent != (REPO_ROOT / "configs" / "profiles").resolve() or not profile_path.exists():
        raise HTTPException(status_code=404, detail="unknown profile")
    try:
        station = await run_in_threadpool(manager.activate, profile_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=repr(exc)) from exc
    return station.metrics()


@app.post("/inspect")
async def inspect(
    file: UploadFile = File(...),
    part_id: str = Form("PART-0001"),
    _: None = Depends(_require_token),
) -> JSONResponse:
    if not PART_ID_RE.fullmatch(part_id):
        raise HTTPException(status_code=422, detail="part_id must be 1-64 letters, digits, _ or -")
    payload = await file.read()
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="image exceeds 8 MiB")
    try:
        with Image.open(io.BytesIO(payload)) as opened:
            width, height = opened.size
            if width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=422, detail="image exceeds the pixel limit")
            image = opened.convert("RGB")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"unreadable image: {exc!r}") from exc
    station = _station()
    result = await run_in_threadpool(station.inspect, image, part_id)
    await manager.broadcast({"type": "verdict", "result": result.to_dict()})
    return JSONResponse(result.to_dict())


@app.post("/source/start")
async def start_source(payload: dict[str, Any], _: None = Depends(_require_token)) -> dict[str, Any]:
    station = _station()
    source_type = str(payload.get("type", "")).lower()
    if source_type in ("", "none", "off"):
        return await run_in_threadpool(station.stop_capture) or {"running": False}
    if source_type not in ("folder", "video", "camera", "usb", "csi"):
        raise HTTPException(status_code=422, detail="unsupported source type")
    source_config = dict(payload)
    source_config["type"] = source_type
    try:
        status = await run_in_threadpool(station.start_capture, source_config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=repr(exc)) from exc
    return status


@app.post("/source/stop")
async def stop_source(_: None = Depends(_require_token)) -> dict[str, Any]:
    station = _station()
    return await run_in_threadpool(station.stop_capture) or {"running": False}


@app.get("/source/status")
def source_status(_: None = Depends(_require_token)) -> dict[str, Any] | None:
    return _station().capture_status()


@app.post("/capture")
async def capture(part_id: str = Form("PART-0001"), _: None = Depends(_require_token)) -> JSONResponse:
    if not PART_ID_RE.fullmatch(part_id):
        raise HTTPException(status_code=422, detail="part_id must be 1-64 letters, digits, _ or -")
    station = _station()
    try:
        result = await run_in_threadpool(station.capture_inspect, part_id)
    except Exception as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    await manager.broadcast({"type": "verdict", "result": result.to_dict()})
    return JSONResponse(result.to_dict())


def _mjpeg_frames(station: Station):
    while station.capture is not None:
        frame = station.latest_jpeg()
        if frame is not None:
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                + str(len(frame)).encode("ascii")
                + b"\r\n\r\n"
                + frame
                + b"\r\n"
            )
        else:
            time.sleep(0.1)
        time.sleep(0.05)


@app.get("/stream.mjpg")
def stream_mjpeg(token: str | None = None, x_veriqc_token: str | None = Header(default=None)):
    if not (_token_ok(token) or _token_ok(x_veriqc_token)):
        raise HTTPException(status_code=401, detail="invalid station token")
    station = _station()
    if station.capture is None:
        raise HTTPException(status_code=409, detail="no capture source is active")
    return StreamingResponse(
        _mjpeg_frames(station),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/result/{inspection_id}")
def result(inspection_id: str, _: None = Depends(_require_token)) -> dict[str, Any]:
    station = _station()
    record = station.store.get(inspection_id)
    if record is None:
        raise HTTPException(status_code=404, detail="inspection not found")
    advisory = station.store.latest_event(inspection_id, "advisory")
    if advisory is not None and advisory.get("extra"):
        record["advisory"] = advisory["extra"]
    review = station.store.latest_event(inspection_id, "review")
    if review is not None and review.get("result", {}).get("operator"):
        record["operator"] = review["result"]["operator"]
    return record


@app.get("/overlay/{inspection_id}")
def overlay(
    inspection_id: str,
    token: str | None = None,
    x_veriqc_token: str | None = Header(default=None),
) -> FileResponse:
    if not (_token_ok(token) or _token_ok(x_veriqc_token)):
        raise HTTPException(status_code=401, detail="invalid station token")
    station = _station()
    record = station.store.get(inspection_id)
    if record is None or not record.get("overlay"):
        raise HTTPException(status_code=404, detail="overlay not found")
    path = (REPO_ROOT / record["overlay"]).resolve()
    overlay_root = (REPO_ROOT / "artifacts" / "overlays").resolve()
    if not path.is_relative_to(overlay_root) or not path.exists():
        raise HTTPException(status_code=404, detail="overlay not found")
    return FileResponse(path, media_type="image/png")


@app.get("/history")
def history(limit: int = 25, _: None = Depends(_require_token)) -> dict[str, Any]:
    limit = max(1, min(int(limit), 200))
    station = _station()
    chain = station.store.verify()
    return {"entries": station.store.history(limit=limit), "integrity_ok": chain["ok"], "chain": chain}


@app.get("/metrics")
def metrics(_: None = Depends(_require_token)) -> dict[str, Any]:
    return _station().metrics()


@app.post("/review/{inspection_id}")
async def review(inspection_id: str, payload: dict[str, Any], _: None = Depends(_require_token)) -> dict[str, Any]:
    label = str(payload.get("label", "")).upper()
    if label not in ("PASS", "FAIL"):
        raise HTTPException(status_code=422, detail="label must be PASS or FAIL")
    station = _station()
    record = await run_in_threadpool(station.review, inspection_id, label, str(payload.get("note", "")))
    if record is None:
        raise HTTPException(status_code=404, detail="inspection not found")
    return record


@app.get("/export.csv")
def export_csv(
    token: str | None = None,
    x_veriqc_token: str | None = Header(default=None),
) -> FileResponse:
    if not (_token_ok(token) or _token_ok(x_veriqc_token)):
        raise HTTPException(status_code=401, detail="invalid station token")
    station = _station()
    target = REPO_ROOT / "reports" / "audit_export.csv"
    station.store.export_csv(target)
    return FileResponse(target, media_type="text/csv", filename="veriqc_audit.csv")


@app.websocket("/ws")
async def websocket(websocket: WebSocket, token: str | None = None) -> None:
    if not _token_ok(token):
        await websocket.close(code=4401)
        return
    await websocket.accept()
    manager.connections.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in manager.connections:
            manager.connections.remove(websocket)
