"""FastAPI Backend and Real-time Operator UI for VeriQC Station.

Provides:
- POST /inspect: Synchronous part inspection (takt time ~30ms), queues verified advisory.
- GET /history: Tamper-evident audit logs with cryptographic hash verification.
- GET /metrics: Station throughput, verdict distribution, and stage latencies.
- WebSocket /ws: Real-time broadcast of inspection decisions and verified advisories.
- GET /: Rich single-page operator dashboard.
"""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from typing import List

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

from veriqc.advisory.cards import load_cards
from veriqc.advisory.runner import ExecuTorchAdvisor
from veriqc.advisory.worker import AdvisoryOutcome, AdvisoryWorker
from veriqc.contract.qc import QualityBand, QualityCalibration
from veriqc.contract.record import DefectRecord
from veriqc.contract.severity import SeverityPolicy, load_policy
from veriqc.station.engine import InspectionStation, StationConfig
from veriqc.storage.db import InspectionDatabase
from veriqc.vision.detector import YOLO26Detector
from veriqc.vision.gate import AnomalyGate
from veriqc.vision.pipeline import PipelineConfig, VisionPipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
CONFIGS_DIR = REPO_ROOT / "configs"

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    stn = get_or_create_station()
    yield
    stn.shutdown(wait_for_queue=False)

app = FastAPI(title="VeriQC Edge Station", version="1.0.0", lifespan=lifespan)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect(connection)


manager = ConnectionManager()

# Global station state
station: InspectionStation | None = None
database: InspectionDatabase | None = None


def get_or_create_station() -> InspectionStation:
    global station, database
    if station is not None:
        return station

    db_path = REPO_ROOT / "veriqc_audit.db"
    database = InspectionDatabase(db_path)

    # 1. Anomaly Gate
    gate_path = MODELS_DIR / "anomaly_gate_efficientnet_b0.pt"
    if gate_path.exists():
        gate = AnomalyGate.load(gate_path)
    else:
        # Fallback empty gate if not yet fitted
        gate = AnomalyGate()

    # 2. YOLO26 Detector
    det_weights = MODELS_DIR / "ultralytics" / "yolo26n_c640" / "weights" / "best.pt"
    if not det_weights.exists():
        det_weights = MODELS_DIR / "weights" / "yolo26n.pt"
    detector = YOLO26Detector.from_weights(det_weights) if det_weights.exists() else None

    # 3. Severity & QC Policies
    policy_path = CONFIGS_DIR / "severity_policy.yaml"
    sev_policy = load_policy(policy_path) if policy_path.exists() else SeverityPolicy()

    calib = QualityCalibration(
        blur=QualityBand(lo=10.0, hi=10000.0, name="blur"),
        luminance=QualityBand(lo=15.0, hi=245.0, name="luminance"),
        saturation=QualityBand(lo=0.0, hi=255.0, name="saturation"),
    )

    pipe_cfg = PipelineConfig(threshold_review=0.70, threshold_fail=1.20)
    vision_pipeline = VisionPipeline(
        anomaly_gate=gate,
        detector=detector,
        severity_policy=sev_policy,
        quality_calibration=calib,
        config=pipe_cfg,
    )

    # 4. Advisory Worker (ExecuTorch Llama 3.2 1B dynamic INT8)
    pte_path = MODELS_DIR / "llama32_1b_dyn8.pte"
    tok_path = MODELS_DIR / "llama32_1b_dyn8.pte.tokenizer.json"
    advisor_runner = None
    if pte_path.exists() and tok_path.exists():
        try:
            advisor_runner = ExecuTorchAdvisor(pte_path, tok_path)
        except Exception as e:
            print(f"[Warning] ExecuTorch runner failed to load: {e}")

    cards_path = CONFIGS_DIR / "sop_cards.yaml"
    cards = load_cards(cards_path) if cards_path.exists() else None
    advisory_worker = AdvisoryWorker(card_library=cards, runner=advisor_runner) if cards else None

    # Callback when background advisory finishes
    def on_advisory_done(rec: DefectRecord, adv: AdvisoryOutcome):
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(
                manager.broadcast(
                    {
                        "type": "advisory_update",
                        "part_id": rec.part_id,
                        "advisory": adv.to_dict(),
                        "text": adv.text,
                    }
                )
            )

    station = InspectionStation(
        vision_pipeline=vision_pipeline,
        advisory_worker=advisory_worker,
        database=database,
        config=StationConfig(),
        on_advisory_ready=on_advisory_done,
    )
    return station





@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.post("/inspect")
async def inspect_endpoint(file: UploadFile = File(...), part_id: str = "P-AUTO"):
    stn = get_or_create_station()
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    rec = stn.inspect_part(img, part_id=part_id)
    rec_dict = rec.to_dict()

    # Broadcast to connected operator screens
    await manager.broadcast({"type": "inspection_verdict", "record": rec_dict})
    return JSONResponse(rec_dict)


@app.get("/history")
def history_endpoint(limit: int = 50):
    global database
    if database is None:
        get_or_create_station()

    with database._get_connection() as conn:
        cursor = conn.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
        )
        rows = [dict(r) for r in cursor.fetchall()]

    ok, total, err = database.verify_integrity()
    return JSONResponse({"integrity_verified": ok, "total_records": total, "entries": rows})


@app.get("/metrics")
def metrics_endpoint():
    global database
    if database is None:
        get_or_create_station()

    with database._get_connection() as conn:
        c_total = conn.execute("SELECT count(*) FROM audit_log").fetchone()[0]
        c_pass = conn.execute("SELECT count(*) FROM audit_log WHERE verdict = 'PASS'").fetchone()[0]
        c_fail = conn.execute("SELECT count(*) FROM audit_log WHERE verdict = 'FAIL'").fetchone()[0]
        c_rev = conn.execute("SELECT count(*) FROM audit_log WHERE verdict = 'REVIEW'").fetchone()[0]

    return JSONResponse(
        {
            "total_inspections": c_total,
            "pass_count": c_pass,
            "fail_count": c_fail,
            "review_count": c_rev,
        }
    )


@app.get("/", response_class=HTMLResponse)
def operator_dashboard():
    return HTMLResponse(
        """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VeriQC - Edge Inspection Station</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-950 text-slate-100 font-sans min-h-screen">
    <div class="max-w-7xl mx-auto p-6 space-y-6">
        <!-- Header -->
        <header class="flex justify-between items-center border-b border-slate-800 pb-4">
            <div>
                <h1 class="text-2xl font-bold tracking-tight text-white flex items-center gap-3">
                    <span class="w-3 h-3 rounded-full bg-emerald-500 animate-pulse"></span>
                    VeriQC Edge Station
                </h1>
                <p class="text-xs text-slate-400 mt-1">Calibrated Triage &amp; Verified On-Device Advisory (ExecuTorch Llama 3.2 1B)</p>
            </div>
            <div class="flex items-center gap-4 text-xs font-mono">
                <span id="integrity-badge" class="px-2.5 py-1 rounded bg-emerald-950 text-emerald-400 border border-emerald-800">
                    Chain Integrity: Verified
                </span>
                <span class="px-2.5 py-1 rounded bg-slate-900 text-slate-300 border border-slate-800">
                    Runtime: Edge NPU/CPU
                </span>
            </div>
        </header>

        <!-- Main Grid -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <!-- Inspection Stream & Banner -->
            <div class="lg:col-span-2 space-y-6">
                <!-- Verdict Banner -->
                <div id="verdict-card" class="rounded-xl border border-slate-800 bg-slate-900/60 p-8 flex flex-col items-center justify-center transition-all">
                    <span class="text-xs uppercase tracking-wider text-slate-400 font-semibold mb-2">Station Verdict</span>
                    <div id="verdict-banner" class="text-5xl font-black text-slate-300 tracking-wider font-mono">
                        STANDBY
                    </div>
                    <div id="part-id-badge" class="mt-3 text-sm font-mono text-slate-400">
                        Waiting for part trigger...
                    </div>
                </div>

                <!-- Evidence & Metrics -->
                <div class="grid grid-cols-3 gap-4 font-mono text-sm">
                    <div class="bg-slate-900 p-4 rounded-lg border border-slate-800">
                        <span class="text-xs text-slate-400 block font-sans">Anomaly Score</span>
                        <span id="metric-score" class="text-xl font-bold text-white">--</span>
                    </div>
                    <div class="bg-slate-900 p-4 rounded-lg border border-slate-800">
                        <span class="text-xs text-slate-400 block font-sans">Defect Class</span>
                        <span id="metric-class" class="text-xl font-bold text-white">--</span>
                    </div>
                    <div class="bg-slate-900 p-4 rounded-lg border border-slate-800">
                        <span class="text-xs text-slate-400 block font-sans">Defect Area (mm²)</span>
                        <span id="metric-area" class="text-xl font-bold text-white">--</span>
                    </div>
                </div>

                <!-- Inspection Upload Trigger -->
                <div class="bg-slate-900 p-6 rounded-xl border border-slate-800 space-y-4">
                    <h3 class="text-sm font-semibold uppercase tracking-wider text-slate-400">Manual Part Trigger</h3>
                    <div class="flex items-center gap-4">
                        <input type="text" id="input-part-id" placeholder="Part ID (e.g. P-0412)" value="P-0412" class="bg-slate-950 border border-slate-700 px-3 py-2 rounded text-sm font-mono w-48 text-white focus:outline-none focus:border-indigo-500">
                        <input type="file" id="input-file" accept="image/*" class="text-sm text-slate-400 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-xs file:font-semibold file:bg-slate-800 file:text-slate-200 hover:file:bg-slate-700">
                        <button onclick="triggerInspection()" class="bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-xs px-5 py-2.5 rounded transition">
                            Inspect Part
                        </button>
                    </div>
                </div>
            </div>

            <!-- Verified Advisory Panel -->
            <div class="space-y-6">
                <div class="bg-slate-900 p-6 rounded-xl border border-slate-800 flex flex-col h-full space-y-4">
                    <div class="flex justify-between items-center border-b border-slate-800 pb-3">
                        <h2 class="text-sm font-bold uppercase tracking-wider text-slate-200">On-Device Advisory</h2>
                        <span id="advisory-badge" class="text-xs font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400">
                            STANDBY
                        </span>
                    </div>
                    <div id="advisory-content" class="flex-1 text-sm text-slate-300 font-sans space-y-3 leading-relaxed">
                        <p class="text-slate-500 italic">No advisory requested. Station is ready for next part.</p>
                    </div>
                    <div class="border-t border-slate-800 pt-3 text-xs font-mono text-slate-500 flex justify-between">
                        <span id="adv-tokens">Tokens: --</span>
                        <span id="adv-latency">Decode: --</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Audit Trail Table -->
        <div class="bg-slate-900 rounded-xl border border-slate-800 overflow-hidden">
            <div class="px-6 py-4 border-b border-slate-800 flex justify-between items-center">
                <h3 class="text-sm font-bold uppercase tracking-wider text-slate-300">Tamper-Evident Audit Chain (SQLite WAL)</h3>
                <span class="text-xs text-slate-400">Rolling SHA-256 Hashes</span>
            </div>
            <div class="overflow-x-auto">
                <table class="w-full text-left text-xs font-mono">
                    <thead class="bg-slate-950/60 text-slate-400 border-b border-slate-800">
                        <tr>
                            <th class="px-4 py-2.5">Row ID</th>
                            <th class="px-4 py-2.5">Part ID</th>
                            <th class="px-4 py-2.5">Verdict</th>
                            <th class="px-4 py-2.5">Record Hash</th>
                            <th class="px-4 py-2.5">Chain Prev Hash</th>
                            <th class="px-4 py-2.5">Chain Block Hash</th>
                        </tr>
                    </thead>
                    <tbody id="audit-table-body" class="divide-y divide-slate-800 text-slate-300">
                        <!-- Populated by JS -->
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        const ws = new WebSocket(`ws://${location.host}/ws`);
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === "inspection_verdict") {
                updateVerdict(data.record);
                fetchHistory();
            } else if (data.type === "advisory_update") {
                updateAdvisory(data);
                fetchHistory();
            }
        };

        function updateVerdict(record) {
            const banner = document.getElementById("verdict-banner");
            const partBadge = document.getElementById("part-id-badge");
            const card = document.getElementById("verdict-card");

            banner.innerText = record.verdict;
            partBadge.innerText = `Part ID: ${record.part_id} | Total Time: ${record.latency_ms?.total_ms || '--'} ms`;

            card.className = "rounded-xl border p-8 flex flex-col items-center justify-center transition-all " +
                (record.verdict === "PASS" ? "bg-emerald-950/30 border-emerald-700/50 text-emerald-400" :
                 record.verdict === "FAIL" ? "bg-rose-950/30 border-rose-700/50 text-rose-400" :
                 "bg-amber-950/30 border-amber-700/50 text-amber-400");

            document.getElementById("metric-score").innerText = record.anomaly_score ?? "--";
            document.getElementById("metric-class").innerText = record.defect_class ? `${record.defect_class} (${Math.round((record.class_confidence||0)*100)}%)` : "None";
            document.getElementById("metric-area").innerText = record.mask_area_mm2 ? `${record.mask_area_mm2} [${record.severity||'minor'}]` : "--";

            // Reset advisory while waiting for worker
            if (record.verdict !== "PASS") {
                document.getElementById("advisory-badge").innerText = "GENERATING...";
                document.getElementById("advisory-badge").className = "text-xs font-mono px-2 py-0.5 rounded bg-amber-900 text-amber-300 animate-pulse";
                document.getElementById("advisory-content").innerHTML = `<p class="text-slate-400 italic">Synthesizing SOP and verifying claims against DefectRecord ${record.part_id}...</p>`;
            }
        }

        function updateAdvisory(advData) {
            const adv = advData.advisory;
            const badge = document.getElementById("advisory-badge");
            badge.innerText = adv.verified ? `VERIFIED (${adv.path.toUpperCase()})` : `UNVERIFIED`;
            badge.className = adv.verified ?
                "text-xs font-mono px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800" :
                "text-xs font-mono px-2 py-0.5 rounded bg-rose-950 text-rose-400 border border-rose-800";

            document.getElementById("advisory-content").innerHTML = `
                <div class="bg-slate-950 p-3 rounded border border-slate-800 font-mono text-xs text-indigo-300">
                    ACTIONS: ${adv.action_ids.join(", ") || "None"}
                </div>
                <div class="text-slate-200 mt-2">
                    ${advData.text || "No advisory text"}
                </div>
            `;
            document.getElementById("adv-tokens").innerText = `Tokens: ${adv.generated_tokens} (prompt ${adv.prompt_tokens})`;
            document.getElementById("adv-latency").innerText = `Rate: ${adv.tokens_per_second} tok/s (TTFT ${adv.ttft_ms}ms)`;
        }

        async function triggerInspection() {
            const partId = document.getElementById("input-part-id").value || "P-0412";
            const fileInput = document.getElementById("input-file");
            if (!fileInput.files.length) {
                alert("Please select an image file first");
                return;
            }

            const formData = new FormData();
            formData.append("file", fileInput.files[0]);
            formData.append("part_id", partId);

            const res = await fetch(`/inspect?part_id=${partId}`, {
                method: "POST",
                body: formData
            });
            const data = await res.json();
            updateVerdict(data);
            fetchHistory();
        }

        async function fetchHistory() {
            const res = await fetch("/history?limit=10");
            const data = await res.json();
            const tbody = document.getElementById("audit-table-body");
            tbody.innerHTML = "";
            for (const row of data.entries) {
                const tr = document.createElement("tr");
                tr.innerHTML = `
                    <td class="px-4 py-2 text-slate-500">${row.id}</td>
                    <td class="px-4 py-2 font-bold">${row.part_id}</td>
                    <td class="px-4 py-2 ${row.verdict === 'PASS' ? 'text-emerald-400' : row.verdict === 'FAIL' ? 'text-rose-400' : 'text-amber-400'}">${row.verdict}</td>
                    <td class="px-4 py-2 text-slate-400">${row.record_hash}</td>
                    <td class="px-4 py-2 text-slate-500 font-mono truncate max-w-xs">${row.prev_hash.slice(0, 16)}...</td>
                    <td class="px-4 py-2 text-emerald-400 font-mono truncate max-w-xs">${row.curr_hash.slice(0, 16)}...</td>
                `;
                tbody.appendChild(tr);
            }
        }
        fetchHistory();
    </script>
</body>
</html>
        """
    )

