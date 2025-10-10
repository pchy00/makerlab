# app.py — ShelfCam single-service app (FastAPI + OpenCV + SQLite)
# Run:
#   python -m pip install fastapi uvicorn opencv-python numpy python-multipart
#   python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000

import io, json, time, sqlite3
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, Form, HTTPException, Query
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.responses import HTMLResponse, RedirectResponse

app = FastAPI(title="ShelfCam")

# =========================
# Config / ROIs (slots)
# =========================
CANVAS_W, CANVAS_H = 640, 480
CONFIG_PATH = Path("slots.json")  

DEFAULT_SLOTS_CFG = {
    "canvas_size": [CANVAS_W, CANVAS_H],
    "edge_threshold": 0.35,
    # Start with 4 vertical blocks; tweak later or create slots.json
    "slots": [
        {"id": "S1", "x":  40, "y": 120, "w": 140, "h": 240},
        {"id": "S2", "x": 200, "y": 120, "w": 140, "h": 240},
        {"id": "S3", "x": 360, "y": 120, "w": 140, "h": 240},
        {"id": "S4", "x": 520, "y": 120, "w":  80, "h": 240},
    ]
}

def load_slots() -> Dict:
    if CONFIG_PATH.exists():
        cfg = json.loads(CONFIG_PATH.read_text())
    else:
        cfg = DEFAULT_SLOTS_CFG
    sz = cfg.get("canvas_size", [CANVAS_W, CANVAS_H])
    if sz != [CANVAS_W, CANVAS_H]:
        # Keep canvas consistent so ROIs line up after resize
        cfg["canvas_size"] = [CANVAS_W, CANVAS_H]
    return cfg

SLOTS_CFG = load_slots()

# ---- Stock presentation config ----
CAPACITY = {
    "A1-BIN-01": 10,
    "A1-BIN-02": 8,
    "A1-BIN-03": 12,
    "A1-BIN-04": 6,
}
DEFAULT_CAPACITY = 10
LOW_STOCK_AT = 0.25   # <=25% shows Low (amber)
OUT_AT = 0            # ==0 shows Out (red)


# =========================
# SQLite helpers
# =========================
DB_PATH = "shelfcam.sqlite"

def db_init():
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS shelf_counts(
        shelf_key TEXT PRIMARY KEY,
        count INTEGER NOT NULL DEFAULT 0,
        modified TEXT NOT NULL
    )""")
    con.commit(); con.close()

def db_set_count(shelf_key: str, count: int):
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    cur.execute("""
      INSERT INTO shelf_counts (shelf_key, count, modified)
      VALUES (?, ?, ?)
      ON CONFLICT(shelf_key) DO UPDATE SET
        count=excluded.count, modified=excluded.modified
    """, (shelf_key, int(count), ts))
    con.commit(); con.close()

def db_get_one(shelf_key: str):
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    row = cur.execute("SELECT shelf_key, count, modified FROM shelf_counts WHERE shelf_key=?",
                      (shelf_key,)).fetchone()
    con.close(); return row

def db_list_all():
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    rows = cur.execute("SELECT shelf_key, count, modified FROM shelf_counts ORDER BY shelf_key").fetchall()
    con.close(); return rows

@app.on_event("startup")
def _startup():
    db_init()

# =========================
# Counting logic (camera-only)
# =========================
def count_slot_items(roi: np.ndarray, thr: float) -> int:
    """
    Counts visible item faces using vertical-edge projection across the ROI.
    Best with steady lighting and products with vertical edges (boxes, labels).
    """
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3,3), 0)
    sobelx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    proj = np.mean(np.abs(sobelx), axis=0)  # vertical edge strength per column

    # normalize -> threshold -> close gaps
    p = (proj - proj.min()) / (proj.ptp() + 1e-6)
    binary = (p > thr).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones(7, np.uint8))

    runs, prev = 0, 0
    for v in binary:
        if v and not prev: runs += 1
        prev = v
    return int(runs)

def count_image_total(img_bgr: np.ndarray) -> int:
    canvas = cv2.resize(img_bgr, (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_AREA)
    thr = float(SLOTS_CFG.get("edge_threshold", 0.35))
    total = 0
    for s in SLOTS_CFG["slots"]:
        x, y, w, h = s["x"], s["y"], s["w"], s["h"]
        roi = canvas[y:y+h, x:x+w]
        total += count_slot_items(roi, thr)
    return total

# =========================
# API: analyze + JSON for UI
# =========================
@app.post("/analyze")
async def analyze(shelf_key: str = Form(...), frame: UploadFile = Form(...)):
    """
    ESP32-CAM posts here: form fields { shelf_key, frame (JPEG/PNG) }.
    We count in RAM and write the exact count to SQLite. No image stored.
    """
    if frame.content_type not in ("image/jpeg","image/jpg","image/png"):
        raise HTTPException(400, "frame must be JPG or PNG")

    data = await frame.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "invalid image")

    count = count_image_total(img)
    db_set_count(shelf_key, count)
    return {"ok": True, "shelf_key": shelf_key, "count": int(count)}

@app.get("/api/v1/shelves")
def api_list_shelves():
    rows = db_list_all()
    enriched = []
    for k, c, m in rows:
        cap = int(CAPACITY.get(k, DEFAULT_CAPACITY))
        pct = (int(c) / cap) if cap > 0 else 0
        if int(c) == OUT_AT:
            status = "out"
        elif pct <= LOW_STOCK_AT:
            status = "low"
        else:
            status = "ok"
        enriched.append({
            "key": k, "count": int(c), "modified": m,
            "capacity": cap, "percent": round(pct*100, 1), "status": status
        })
    # sort by status (out→low→ok), then key
    order = {"out":0,"low":1,"ok":2}
    enriched.sort(key=lambda r: (order.get(r["status"], 9), r["key"]))
    return enriched

@app.get("/api/v1/shelves/{shelf_key}")
def api_get_shelf(shelf_key: str):
    r = db_get_one(shelf_key)
    if not r:
        return JSONResponse(status_code=404, content={"error": "not found"})
    k, c, m = r[0], int(r[1]), r[2]
    cap = int(CAPACITY.get(k, DEFAULT_CAPACITY))
    pct = (c / cap) if cap > 0 else 0
    if c == OUT_AT:
        status = "out"
    elif pct <= LOW_STOCK_AT:
        status = "low"
    else:
        status = "ok"
    return {
        "key": k, "count": c, "modified": m,
        "capacity": cap, "percent": round(pct*100, 1), "status": status
    }


# =========================
# UI (Tailwind + Chart.js)
# =========================
HOME_HTML = """
<!doctype html><html lang="en"><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>ShelfCam</title>
<script src="https://cdn.tailwindcss.com"></script>
<body class="bg-slate-50 text-slate-900">
<header class="sticky top-0 bg-white border-b border-slate-200/80">
  <div class="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between gap-3">
    <div class="flex items-center gap-2">
      <div class="text-xl font-semibold"> ShelfCam</div>
      <a href="/dev" class="text-xs text-slate-500 hover:text-slate-700">dev</a>
    </div>
    <div class="text-xs text-slate-500">Status: <span id="legend" class="inline-flex items-center gap-2">
      <span class="inline-block w-2.5 h-2.5 rounded-full bg-emerald-500"></span>OK
      <span class="inline-block w-2.5 h-2.5 rounded-full bg-amber-500 ml-3"></span>Low
      <span class="inline-block w-2.5 h-2.5 rounded-full bg-rose-500 ml-3"></span>Out
    </span></div>
  </div>
</header>

<main class="max-w-7xl mx-auto px-4 py-6">
  <div class="flex items-center justify-between mb-4 gap-3">
    <h1 class="text-2xl font-semibold">All Shelves</h1>
    <div class="flex items-center gap-2">
      <button id="refresh" type="button" class="px-3 py-2 rounded-lg border border-slate-300 bg-white text-sm">Refresh now
      </button>
      <input id="search" placeholder="Search shelf…" class="px-3 py-2 rounded-lg border border-slate-300 bg-white text-sm w-56">
      <select id="sort" class="px-3 py-2 rounded-lg border border-slate-300 bg-white text-sm">
        <option value="status">Sort: Status</option>
        <option value="key">Sort: Name</option>
        <option value="count">Sort: Count</option>
        <option value="updated">Sort: Updated</option>
      </select>
    </div>
  </div>
  <div id="grid" class="grid gap-4 sm:grid-cols-2 xl:grid-cols-3"></div>
</main>

<script>
const API = location.origin + "/api/v1";
let rows = [], filtered = [];

function clsStatus(s){
  if(s==="out") return "bg-rose-100 text-rose-700 ring-1 ring-rose-200";
  if(s==="low") return "bg-amber-100 text-amber-700 ring-1 ring-amber-200";
  return "bg-emerald-100 text-emerald-700 ring-1 ring-emerald-200";
}
function dotStatus(s){
  if(s==="out") return "bg-rose-500";
  if(s==="low") return "bg-amber-500";
  return "bg-emerald-500";
}
function fmt(t){ try { return new Date(t).toLocaleString(); } catch(e){ return t; } }

function render(){
  const grid = document.getElementById("grid"); grid.innerHTML = "";
  filtered.forEach(r=>{
    const a = document.createElement("a");
    a.href = "/shelf/"+encodeURIComponent(r.key);
    a.className="block rounded-xl border border-slate-200 bg-white px-5 py-4 shadow-sm hover:shadow-md transition";
    a.innerHTML = `
      <div class="flex items-start justify-between gap-3">
        <div>
          <div class="text-xs text-slate-500">Shelf</div>
          <div class="text-lg font-semibold">${r.key}</div>
          <div class="mt-2 flex items-center gap-2">
            <span class="inline-flex items-center gap-2 text-xs px-2 py-1 rounded-full ${clsStatus(r.status)}">
              <span class="inline-block w-1.5 h-1.5 rounded-full ${dotStatus(r.status)}"></span>
              ${r.status === "ok" ? "OK" : r.status === "low" ? "Low Stock" : "Out of Stock"}
            </span>
            <span class="text-xs text-slate-500">Updated: ${fmt(r.modified)}</span>
          </div>
        </div>
        <div class="text-right">
          <div class="text-4xl font-extrabold tabular-nums">${r.count}</div>
          <div class="text-xs text-slate-500">of ${r.capacity}</div>
        </div>
      </div>
      <div class="mt-3 w-full h-2 bg-slate-100 rounded-full overflow-hidden">
        <div class="h-full ${r.status==='out'?'bg-rose-500': r.status==='low'?'bg-amber-500':'bg-emerald-500'}" style="width:${Math.min(100,Math.max(0,r.percent))}%"></div>
      </div>`;
    grid.appendChild(a);
  });
}

function applyFilters(){
  const q = document.getElementById("search").value.toLowerCase().trim();
  const sort = document.getElementById("sort").value;
  filtered = rows.filter(r => r.key.toLowerCase().includes(q));
  if(sort==="key") filtered.sort((a,b)=>a.key.localeCompare(b.key));
  else if(sort==="count") filtered.sort((a,b)=>b.count - a.count);
  else if(sort==="updated") filtered.sort((a,b)=>new Date(b.modified)-new Date(a.modified));
  else filtered.sort((a,b)=>{
    const order={out:0,low:1,ok:2};
    return (order[a.status]-order[b.status]) || a.key.localeCompare(b.key);
  });
  render();
}

async function load(force=false){
  const url = API + "/shelves" + (force ? ("?t=" + Date.now()) : "");
  const res = await fetch(url, { cache: "no-store" });
  if(!res.ok) return;
  rows = await res.json();
  applyFilters();
}
document.getElementById("search").addEventListener("input", applyFilters);
document.getElementById("sort").addEventListener("change", applyFilters);
document.getElementById("refresh").addEventListener("click", (e)=>{e.preventDefault();load(true);
});

load();
// poll every 5 minutes; change to 10000 (10s) while developing
setInterval(load, 300000);
</script>
</body></html>
"""


DETAIL_HTML = """
<!doctype html><html lang="en"><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>ShelfCam • Detail</title>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<body class="bg-slate-50 text-slate-900">
<header class="sticky top-0 bg-white border-b border-slate-200/80">
  <div class="max-w-5xl mx-auto px-4 py-3 flex items-center gap-3">
    <a href="/" class="text-slate-500 hover:text-slate-700">← Back</a>
    <div id="title" class="text-xl font-semibold">Shelf</div>
    <span id="chip" class="ml-auto text-xs px-2 py-1 rounded-full"></span>
  </div>
</header>

<main class="max-w-5xl mx-auto px-4 py-6">
  <div class="grid md:grid-cols-3 gap-4">
    <div class="md:col-span-1 rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <div class="text-sm text-slate-500">Current count</div>
      <div id="count" class="text-6xl font-extrabold tabular-nums mt-2">—</div>
      <div id="cap" class="text-sm text-slate-500">of —</div>
      <div class="mt-3 w-full h-2 bg-slate-100 rounded-full overflow-hidden">
        <div id="bar" class="h-full bg-emerald-500" style="width:0%"></div>
      </div>
      <div id="updated" class="text-xs text-slate-500 mt-2">Updated: —</div>
    </div>

    <div class="md:col-span-2 rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <div class="flex items-center justify-between mb-2">
        <div class="text-sm text-slate-500">Live trend (this session)</div>
      </div>
      <canvas id="chart" height="120"></canvas>
    </div>
  </div>
</main>
<script>
const API = location.origin + "/api/v1";
let rows = [], filtered = [];

function clsStatus(s){
  if(s==="out") return "bg-rose-100 text-rose-700 ring-1 ring-rose-200";
  if(s==="low") return "bg-amber-100 text-amber-700 ring-1 ring-amber-200";
  return "bg-emerald-100 text-emerald-700 ring-1 ring-emerald-200";
}
function dotStatus(s){ return s==="out" ? "bg-rose-500" : (s==="low" ? "bg-amber-500" : "bg-emerald-500"); }
function fmt(t){ try { return new Date(t).toLocaleString(); } catch { return t; } }

function render(){
  const grid = document.getElementById("grid"); grid.innerHTML = "";
  filtered.forEach(r=>{
    const a = document.createElement("a");
    a.href = "/shelf/"+encodeURIComponent(r.key);
    a.className="block rounded-xl border border-slate-200 bg-white px-5 py-4 shadow-sm hover:shadow-md transition";
    a.innerHTML = `
      <div class="flex items-start justify-between gap-3">
        <div>
          <div class="text-xs text-slate-500">Shelf</div>
          <div class="text-lg font-semibold">${r.key}</div>
          <div class="mt-2 flex items-center gap-2">
            <span class="inline-flex items-center gap-2 text-xs px-2 py-1 rounded-full ${clsStatus(r.status)}">
              <span class="inline-block w-1.5 h-1.5 rounded-full ${dotStatus(r.status)}"></span>
              ${r.status === "ok" ? "OK" : r.status === "low" ? "Low Stock" : "Out of Stock"}
            </span>
            <span class="text-xs text-slate-500">Updated: ${fmt(r.modified)}</span>
          </div>
        </div>
        <div class="text-right">
          <div class="text-4xl font-extrabold tabular-nums">${r.count}</div>
          <div class="text-xs text-slate-500">of ${r.capacity}</div>
        </div>
      </div>
      <div class="mt-3 w-full h-2 bg-slate-100 rounded-full overflow-hidden">
        <div class="h-full ${r.status==='out'?'bg-rose-500': r.status==='low'?'bg-amber-500':'bg-emerald-500'}" style="width:${Math.min(100,Math.max(0,r.percent))}%"></div>
      </div>`;
    grid.appendChild(a);
  });
}

function applyFilters(){
  const q = document.getElementById("search").value.toLowerCase().trim();
  const sort = document.getElementById("sort").value;
  filtered = rows.filter(r => r.key.toLowerCase().includes(q));
  if (sort==="key") filtered.sort((a,b)=>a.key.localeCompare(b.key));
  else if (sort==="count") filtered.sort((a,b)=>b.count - a.count);
  else if (sort==="updated") filtered.sort((a,b)=>new Date(b.modified)-new Date(a.modified));
  else {
    const order={out:0,low:1,ok:2};
    filtered.sort((a,b)=>(order[a.status]-order[b.status]) || a.key.localeCompare(b.key));
  }
  render();
}

async function load(force=false){
  const url = API + "/shelves" + (force ? ("?t=" + Date.now()) : "");
  const res = await fetch(url, { cache: "no-store" });
  if(!res.ok) return;
  rows = await res.json();
  applyFilters();
}

document.addEventListener("DOMContentLoaded", () => {
  // one-time bindings
  document.getElementById("search").addEventListener("input", applyFilters);
  document.getElementById("sort").addEventListener("change", applyFilters);
  const refreshBtn = document.getElementById("refresh");
  if (refreshBtn) {
    refreshBtn.setAttribute("type","button");
    refreshBtn.addEventListener("click", (e)=>{
      e.preventDefault();
      load(true);          // force fetch now
    });
  }

  load();                  // first load
  setInterval(()=>load(true), 300000); // 5-minute poll
});
</script>

</body></html>
"""


@app.get("/", response_class=HTMLResponse)
def ui_home():
    return HTMLResponse(HOME_HTML)

@app.get("/shelf/{shelf_key}", response_class=HTMLResponse)
def ui_shelf(shelf_key: str):
    return HTMLResponse(DETAIL_HTML)

# =========================
# Dev helpers (demo w/out camera)
# =========================

DEV_HTML = """
<!doctype html><html lang="en"><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>ShelfCam • Dev tools</title>
<script src="https://cdn.tailwindcss.com"></script>
<body class="bg-slate-50 text-slate-900">
<header class="sticky top-0 bg-white border-b border-slate-200/80">
  <div class="max-w-5xl mx-auto px-4 py-3 flex items-center justify-between">
    <div class="text-xl font-semibold"> Dev tools</div>
    <a href="/" class="text-sm text-slate-600 hover:text-slate-800">← Back to dashboard</a>
  </div>
</header>

<main class="max-w-5xl mx-auto px-4 py-6 space-y-6">
  <!-- Quick actions -->
  <section class="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
    <div class="flex flex-wrap items-center gap-2">
      <button id="seedBtn" class="px-3 py-2 rounded-lg border border-slate-300 hover:bg-slate-100">
        Seed demo shelves
      </button>
      <button id="refreshBtn" class="px-3 py-2 rounded-lg border border-slate-300 hover:bg-slate-100">
        Refresh list
      </button>
      <div id="toast" class="ml-auto text-sm text-slate-600"></div>
    </div>
  </section>

  <!-- Editor -->
  <section class="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
    <div class="flex flex-wrap items-end gap-3">
      <div>
        <label class="block text-xs text-slate-500">Shelf key</label>
        <input id="kInput" placeholder="A1-BIN-03"
               class="px-3 py-2 rounded-lg border border-slate-300 bg-white w-48">
      </div>
      <div>
        <label class="block text-xs text-slate-500">Delta</label>
        <input id="dInput" placeholder="+1" value="+1"
               class="px-3 py-2 rounded-lg border border-slate-300 bg-white w-24">
      </div>
      <button id="applyBtn" class="px-3 py-2 rounded-lg bg-sky-600 text-white hover:bg-sky-700">Apply</button>
      <div class="text-xs text-slate-500">Tip: use +5, -1, -5, etc.</div>
    </div>
  </section>

  <!-- List -->
  <section class="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
    <div class="flex items-center justify-between mb-3">
      <h2 class="text-lg font-semibold">Shelves</h2>
      <input id="search" placeholder="Search…"
             class="px-3 py-2 rounded-lg border border-slate-300 bg-white w-56">
    </div>
    <div class="overflow-x-auto">
      <table class="w-full text-sm">
        <thead class="text-left text-slate-500">
          <tr>
            <th class="py-2">Key</th>
            <th class="py-2">Count</th>
            <th class="py-2">Capacity</th>
            <th class="py-2">Updated</th>
            <th class="py-2 text-right">Actions</th>
          </tr>
        </thead>
        <tbody id="tbody"></tbody>
      </table>
    </div>
  </section>
</main>

<script>
const API = location.origin + "/api/v1";
let rows = [];

function fmt(t){ try{ return new Date(t).toLocaleString(); }catch(e){ return t; } }
function toast(msg){
  const el = document.getElementById('toast');
  el.textContent = msg;
  setTimeout(()=> el.textContent = "", 2000);
}

async function load(){
  const r = await fetch(API+"/shelves", {cache:"no-store"});
  if(!r.ok) return;
  rows = await r.json();
  render();
}

function render(){
  const q = document.getElementById('search').value.toLowerCase().trim();
  const tb = document.getElementById('tbody'); tb.innerHTML = "";
  rows.filter(r => r.key.toLowerCase().includes(q)).forEach(r=>{
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="py-2 pr-3 font-medium">${r.key}</td>
      <td class="py-2 pr-3 tabular-nums">${r.count}</td>
      <td class="py-2 pr-3 tabular-nums">${r.capacity}</td>
      <td class="py-2 pr-3 text-slate-500">${fmt(r.modified)}</td>
      <td class="py-2 pr-0">
        <div class="flex justify-end gap-2">
          <button class="px-2 py-1 rounded-md border border-slate-300 hover:bg-slate-100" onclick="bump('${r.key}', -5)">-5</button>
          <button class="px-2 py-1 rounded-md border border-slate-300 hover:bg-slate-100" onclick="bump('${r.key}', -1)">-1</button>
          <button class="px-2 py-1 rounded-md border border-slate-300 hover:bg-slate-100" onclick="bump('${r.key}', +1)">+1</button>
          <button class="px-2 py-1 rounded-md border border-slate-300 hover:bg-slate-100" onclick="bump('${r.key}', +5)">+5</button>
          <button class="px-2 py-1 rounded-md bg-white border border-amber-300 text-amber-700 hover:bg-amber-50" onclick="setAbs('${r.key}')">Set…</button>
        </div>
      </td>`;
    tb.appendChild(tr);
  });
}

async function bump(key, delta){
  const r = await fetch(`/dev/bump?key=${encodeURIComponent(key)}&delta=${delta}`);
  if(r.ok){ toast(`Bumped ${key} by ${delta}`); load(); }
}

async function setAbs(key){
  const v = prompt(`Set absolute count for ${key}:`, "");
  if(v===null) return;
  const n = Number(v);
  if(Number.isNaN(n)) { alert("Enter a number"); return; }
  const r = await fetch(`/dev/set?key=${encodeURIComponent(key)}&count=${n}`);
  if(r.ok){ toast(`Set ${key} to ${n}`); load(); }
}

document.getElementById('seedBtn').addEventListener('click', async ()=>{
  const r = await fetch('/dev/seed'); if(r.ok){ toast('Seeded'); load(); }
});
document.getElementById('refreshBtn').addEventListener('click', load);
document.getElementById('applyBtn').addEventListener('click', ()=>{
  const k = document.getElementById('kInput').value.trim();
  const d = parseInt(document.getElementById('dInput').value,10);
  if(!k || Number.isNaN(d)) return alert('Enter key and numeric delta');
  bump(k, d);
});
document.getElementById('search').addEventListener('input', render);

// initial
load();
</script>
</body></html>
"""

@app.get("/dev", response_class=HTMLResponse)
def dev_page():
    return HTMLResponse(DEV_HTML)

# Keep seed & bump, and add a "set absolute" for convenience
@app.get("/dev/seed")
def dev_seed():
    db_set_count("A1-BIN-01", 5)
    db_set_count("A1-BIN-02", 3)
    db_set_count("A1-BIN-03", 7)
    db_set_count("A1-BIN-04", 1)
    return {"ok": True, "seeded": 4}

@app.get("/dev/bump")
def dev_bump(key: str = Query(...), delta: int = Query(...)):
    row = db_get_one(key)
    if not row:
        db_set_count(key, max(0, delta))
        return {"ok": True, "key": key, "count": max(0, delta)}
    new = max(0, int(row[1]) + int(delta))
    db_set_count(key, new)
    return {"ok": True, "key": key, "count": new}

@app.get("/dev/set")
def dev_set(key: str = Query(...), count: int = Query(...)):
    db_set_count(key, max(0, int(count)))
    return {"ok": True, "key": key, "count": max(0, int(count))}
