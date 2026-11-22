"""Operator dashboard, served locally with no external assets."""

from __future__ import annotations

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VeriQC Edge Station</title>
<style>
  :root { color-scheme: dark; --bg:#0a0f1c; --panel:#111827; --line:#1f2937; --text:#e5e7eb; --muted:#94a3b8; --pass:#34d399; --fail:#f87171; --review:#fbbf24; --accent:#60a5fa; }
  * { box-sizing: border-box; }
  body { margin:0; font:14px/1.5 ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; background:var(--bg); color:var(--text); }
  header { display:flex; align-items:center; gap:16px; padding:14px 22px; border-bottom:1px solid var(--line); background:#0d1424; }
  header h1 { font-size:18px; margin:0; letter-spacing:.02em; }
  header .spacer { flex:1; }
  header select, header button, input, button { background:#0b1220; color:var(--text); border:1px solid var(--line); border-radius:6px; padding:8px 10px; font:inherit; }
  button { cursor:pointer; }
  button.primary { border-color:var(--accent); color:#dbeafe; }
  main { display:grid; grid-template-columns: minmax(0,1.55fr) minmax(340px,.9fr); gap:16px; padding:18px 22px 40px; }
  section { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px; }
  h2 { font-size:12px; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); margin:0 0 10px; }
  .verdict { font-size:44px; font-weight:800; letter-spacing:.04em; padding:10px 0 4px; }
  .pass { color:var(--pass); } .fail { color:var(--fail); } .review { color:var(--review); }
  .muted { color:var(--muted); }
  .row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
  .imagebox { margin-top:12px; background:#050914; border:1px solid var(--line); border-radius:8px; min-height:260px; display:flex; align-items:center; justify-content:center; overflow:hidden; }
  .imagebox img { max-width:100%; max-height:520px; display:block; }
  table { width:100%; border-collapse:collapse; margin-top:6px; }
  th, td { text-align:left; padding:7px 8px; border-bottom:1px solid var(--line); font-size:12px; }
  th { color:var(--muted); font-weight:600; }
  .advice { white-space:pre-wrap; line-height:1.55; font-size:14px; }
  .chips { display:flex; gap:6px; flex-wrap:wrap; margin:8px 0; }
  .chip { border:1px solid var(--line); border-radius:999px; padding:3px 9px; font-size:11px; color:var(--muted); }
  .metric { display:grid; grid-template-columns:1fr auto; gap:6px 14px; font-size:12px; }
  .metric div:nth-child(odd) { color:var(--muted); }
  .history tr { cursor:pointer; }
  .history tr:hover td { background:#0b1220; }
  .ok { color:var(--pass); } .bad { color:var(--fail); }
  @media (max-width: 1000px) { main { grid-template-columns:1fr; } }
</style>
</head>
<body>
<header>
  <h1>VeriQC <span class="muted">| edge inspection</span></h1>
  <select id="profile" aria-label="profile"></select>
  <button id="activate">activate</button>
  <span class="spacer"></span>
  <span id="chain" class="muted">chain: --</span>
  <span id="device" class="muted">device: --</span>
</header>
<main>
  <div>
    <section>
      <div class="row">
        <div>
          <h2>verdict</h2>
          <div id="verdict" class="verdict muted">STANDBY</div>
          <div id="part" class="muted">waiting for a part</div>
        </div>
        <span class="spacer" style="flex:1"></span>
        <div class="row">
          <input id="part-id" value="PART-0001" aria-label="part id" style="width:150px">
          <input id="file" type="file" accept="image/*" aria-label="image">
          <button class="primary" id="inspect">inspect</button>
        </div>
      </div>
      <div class="chips" id="chips"></div>
      <div class="imagebox"><img id="overlay" alt="" style="display:none"></div>
    </section>
    <section style="margin-top:16px">
      <h2>findings</h2>
      <table><thead><tr><th>source</th><th>class</th><th>confidence</th><th>score</th><th>area %</th><th>zone</th><th>severity</th></tr></thead><tbody id="findings"></tbody></table>
    </section>
    <section style="margin-top:16px">
      <h2>audit history</h2>
      <table class="history"><thead><tr><th>id</th><th>part</th><th>verdict</th><th>event</th></tr></thead><tbody id="history"></tbody></table>
      <div class="row" style="margin-top:10px"><a id="export" href="/export.csv" class="muted">export csv</a></div>
    </section>
  </div>
  <div>
    <section>
      <h2>on-device advisory</h2>
      <div id="advisory" class="advice muted">no advisory requested</div>
      <div class="chips" id="advisory-meta"></div>
      <div class="row" id="review-actions" style="display:none; margin-top:10px">
        <button id="review-pass">confirm pass</button>
        <button id="review-fail">confirm fail</button>
      </div>
    </section>
    <section style="margin-top:16px">
      <h2>station</h2>
      <div class="metric" id="metrics"></div>
    </section>
    <section style="margin-top:16px">
      <h2>runtime separation</h2>
      <div class="advice muted" id="runtime"></div>
    </section>
  </div>
</main>
<script>
const TOKEN = new URLSearchParams(location.search).get("token") || "";
const headers = () => TOKEN ? {"X-VeriQC-Token": TOKEN} : {};
let currentId = null;

function setText(id, value) {
  const node = document.getElementById(id);
  node.textContent = value === null || value === undefined ? "--" : String(value);
}
function cell(row, value) {
  const td = document.createElement("td");
  td.textContent = value === null || value === undefined ? "--" : String(value);
  row.appendChild(td);
}
function render(result) {
  currentId = result.inspection_id;
  const verdict = document.getElementById("verdict");
  verdict.textContent = result.verdict;
  verdict.className = "verdict " + (result.verdict === "PASS" ? "pass" : result.verdict === "FAIL" ? "fail" : "review");
  setText("part", (result.part_id || "--") + " | " + (result.latency_ms?.total_ms ?? "--") + " ms");
  const overlay = document.getElementById("overlay");
  if (result.overlay) { overlay.src = "/overlay/" + result.inspection_id + (TOKEN ? "?token=" + encodeURIComponent(TOKEN) : ""); overlay.style.display = "block"; }
  else { overlay.style.display = "none"; }
  const findings = document.getElementById("findings");
  findings.textContent = "";
  for (const f of result.findings || []) {
    const tr = document.createElement("tr");
    cell(tr, f.source); cell(tr, f.defect_class); cell(tr, f.confidence); cell(tr, f.score);
    cell(tr, f.area_ratio === null || f.area_ratio === undefined ? "--" : (100 * f.area_ratio).toFixed(3));
    cell(tr, f.zone); cell(tr, f.severity);
    findings.appendChild(tr);
  }
  const chips = document.getElementById("chips");
  chips.textContent = "";
  for (const [key, value] of Object.entries(result.thresholds || {})) {
    const span = document.createElement("span"); span.className = "chip"; span.textContent = key + "=" + value; chips.appendChild(span);
  }
  if (result.anomaly_p_value !== null && result.anomaly_p_value !== undefined) {
    const span = document.createElement("span"); span.className = "chip"; span.textContent = "anomaly p=" + Number(result.anomaly_p_value).toFixed(4); chips.appendChild(span);
  }
  const adv = result.advisory || {};
  setText("advisory", adv.text || (result.verdict === "PASS" ? "Part accepted." : "no advisory"));
  const meta = document.getElementById("advisory-meta");
  meta.textContent = "";
  for (const text of [adv.status, adv.accepted ? "verified model" : "template floor", adv.latency_ms ? Math.round(adv.latency_ms) + " ms" : null]) {
    if (!text) continue;
    const span = document.createElement("span"); span.className = "chip"; span.textContent = text; meta.appendChild(span);
  }
  document.getElementById("review-actions").style.display = result.verdict === "REVIEW" ? "flex" : "none";
  fetchHistory();
}
async function inspect() {
  const file = document.getElementById("file").files[0];
  if (!file) { setText("advisory", "select an image first"); return; }
  const part = document.getElementById("part-id").value || "PART-0001";
  const form = new FormData(); form.append("file", file); form.append("part_id", part);
  const response = await fetch("/inspect", {method:"POST", body:form, headers:headers()});
  if (!response.ok) { setText("advisory", "inspection failed: " + await response.text()); return; }
  render(await response.json());
}
async function fetchHistory() {
  const response = await fetch("/history?limit=12", {headers:headers()});
  const data = await response.json();
  setText("chain", "chain: " + (data.integrity_ok ? "verified" : "BROKEN"));
  const body = document.getElementById("history"); body.textContent = "";
  for (const row of data.entries || []) {
    const tr = document.createElement("tr");
    cell(tr, row.id); cell(tr, row.part_id); cell(tr, row.verdict); cell(tr, row.event);
    tr.onclick = async () => {
      const r = await fetch("/result/" + row.inspection_id, {headers:headers()});
      if (r.ok) render(await r.json());
    };
    body.appendChild(tr);
  }
}
async function fetchMetrics() {
  const response = await fetch("/metrics", {headers:headers()});
  const data = await response.json();
  setText("device", "device: " + (data.device || "--"));
  const box = document.getElementById("metrics"); box.textContent = "";
  const entries = {
    profile: data.profile_id, domain: data.domain, task: data.task,
    advisory: data.advisory?.available ? "available" : "unavailable",
    queue: data.advisory?.queue, inspections: data.stats?.inspections,
    chain_rows: data.chain?.rows, chain_ok: data.chain?.ok,
  };
  for (const [key, value] of Object.entries(entries)) {
    const k = document.createElement("div"); k.textContent = key;
    const v = document.createElement("div"); v.textContent = value === true ? "yes" : value === false ? "no" : value ?? "--";
    box.appendChild(k); box.appendChild(v);
  }
  setText("runtime", "Vision: profile model on " + (data.device || "cpu") + ". Advisory: Built with Llama. Llama 3.2 1B Instruct runs via ExecuTorch in an isolated child process. The advisory cannot change the verdict; the template is the floor.");
}
async function fetchProfiles() {
  const response = await fetch("/profiles", {headers:headers()});
  const data = await response.json();
  const select = document.getElementById("profile"); select.textContent = "";
  for (const p of data.profiles || []) {
    const option = document.createElement("option"); option.value = p.id;
    option.textContent = p.domain + " / " + p.id; option.selected = p.id === data.active;
    select.appendChild(option);
  }
}
async function activate() {
  const id = document.getElementById("profile").value;
  const response = await fetch("/profiles/" + id + "/activate", {method:"POST", headers:headers()});
  if (!response.ok) { setText("advisory", "profile activation failed: " + await response.text()); return; }
  currentId = null; fetchMetrics(); fetchHistory();
}
async function review(label) {
  if (!currentId) return;
  await fetch("/review/" + currentId, {method:"POST", headers:{...headers(), "Content-Type":"application/json"}, body:JSON.stringify({label, note:"operator confirmation"})});
  fetchHistory();
}
document.getElementById("inspect").onclick = inspect;
document.getElementById("activate").onclick = activate;
document.getElementById("review-pass").onclick = () => review("PASS");
document.getElementById("review-fail").onclick = () => review("FAIL");
if (TOKEN) document.getElementById("export").href = "/export.csv?token=" + encodeURIComponent(TOKEN);
const socket = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws" + (TOKEN ? "?token=" + encodeURIComponent(TOKEN) : ""));
socket.onmessage = (event) => {
  const message = JSON.parse(event.data);
  if (message.type === "verdict") render(message.result);
  if (message.type === "advisory" && message.result.inspection_id === currentId) render(message.result);
};
fetchProfiles(); fetchMetrics(); fetchHistory();
</script>
</body>
</html>
"""
