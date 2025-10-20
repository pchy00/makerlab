# app.py — ShelfCam (adaptive poll + draw-only rects + red detector + Mentos debug)

import io, os, json, time, sqlite3, threading, requests, warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# -------- Optional: suppress urllib3 LibreSSL warning (dev only) --------
try:
    from urllib3.exceptions import NotOpenSSLWarning
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except Exception:
    pass

# -------- Config --------
ESP32_CAPTURE_URL = os.environ.get("ESP32_CAPTURE_URL", "http://192.168.18.84/capture")
TARGET_W   = int(os.environ.get("TARGET_W", "960"))
BASE_DIR   = Path(__file__).resolve().parent
DB_PATH    = BASE_DIR / "shelfcam.sqlite"
CONFIG_PATH= BASE_DIR / "slots.json"
STATIC_DIR = BASE_DIR / "static"
TEMPL_DIR  = BASE_DIR / "templates"
GALLERY_MAX= 36

STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPL_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="ShelfCam")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPL_DIR))

_recent_frames: List[Tuple[float, bytes]] = []

# ================= DB =================
def db_conn():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

def db_init():
    with db_conn() as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS shelves(
            key TEXT PRIMARY KEY, name TEXT,
            capacity INTEGER DEFAULT 10,
            count INTEGER DEFAULT 0,
            updated_at INTEGER DEFAULT 0
        )""")
        con.commit()
    seed = [
        ("A1-BIN-01","A1-BIN-01",10),
        ("A1-BIN-02","A1-BIN-02", 8),
        ("A1-BIN-03","A1-BIN-03",12),
        ("A1-BIN-04","A1-BIN-04", 6),
    ]
    with db_conn() as con:
        cur, now = con.cursor(), int(time.time())
        for k,n,cap in seed:
            cur.execute("INSERT OR IGNORE INTO shelves VALUES(?,?,?,?,?)",(k,n,cap,0,now))
        con.commit()

def db_set_count(shelf_key: str, count: int):
    with db_conn() as con:
        row = con.execute("SELECT capacity FROM shelves WHERE key=?", (shelf_key,)).fetchone()
        cap = int(row["capacity"]) if row else 999
        safe = max(0, min(int(count), cap))
        con.execute("UPDATE shelves SET count=?, updated_at=? WHERE key=?",
                    (safe, int(time.time()), shelf_key))
        con.commit()

def db_get_shelves():
    with db_conn() as con:
        rows = con.execute("SELECT key,name,capacity,count,updated_at FROM shelves").fetchall()
    out = []
    for r in rows:
        c = int(r["count"]); cap = int(r["capacity"])
        status = "out" if c<=0 else ("low" if c<=max(1,cap//3) else "ok")
        out.append({"key": r["key"], "name": r["name"], "capacity": cap,
                    "count": c, "status": status, "updated_at": int(r["updated_at"])})
    return out

# =============== Config & detectors ===============
DEFAULT_CFG: Dict[str, Any] = {
  "global": { "target_w": 960 },
  "slots": [
    {
      "id": "S-MENTOS", "key": "A1-BIN-01", "detector": "mentos",
      "x": 70, "y": 110, "w": 260, "h": 360, "pad_pct": 0.12,
      "draw_pad_pct": -0.02, "draw_nudge": [-42, -4],
      "params": { "circle_p2":255, "min_radius":12, "max_radius":36, "min_dist":38,
                  "x_max_frac":0.8, "h_min":28, "h_max":92, "min_ar":1.6, "merge_y":28 }
    },
    {
      "id": "S-REDBOX", "key": "A1-BIN-03", "detector": "red_blobs",
      "x": 320, "y": 70, "w": 300, "h": 220, "pad_pct": 0.08,
      "draw_pad_pct": -0.02, "draw_nudge": [48, -12],
      "params": { "h1_low":0, "h1_high":12, "h2_low":168, "h2_high":179,
                  "s_low":60, "v_low":55, "min_blob_area":1400, "merge_close":22, "x_gap":34 }
    },
    {
      "id": "S-TISSUE", "key": "A1-BIN-04", "detector": "presence_box",
      "x": 285, "y": 300, "w": 360, "h": 220, "pad_pct": 0.10,
      "draw_pad_pct": -0.02, "draw_nudge": [18, 34],
      "params": { "color_min":[130,150,140], "color_max":[235,255,245], "min_fill_ratio":0.16 }
    }
  ]
}

def _validate_slot(s: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for k in ["id","key","detector","x","y","w","h"]:
        if k not in s: return None
    for k in ["x","y","w","h"]:
        s[k] = int(s[k])
    if "params" not in s or not isinstance(s["params"], dict):
        s["params"] = {}
    return s

def load_cfg() -> Dict[str, Any]:
    if CONFIG_PATH.exists():
        try:
            raw = json.loads(CONFIG_PATH.read_text())
            g = raw.get("global", {})
            ok = []
            for s in raw.get("slots", []):
                vs = _validate_slot(dict(s))
                if vs: ok.append(vs)
            if ok:
                return {"global":{"target_w": int(g.get("target_w", TARGET_W))}, "slots": ok}
        except Exception:
            pass
    return DEFAULT_CFG

CFG = load_cfg()
TARGET_W = int(CFG["global"]["target_w"])
_CFG_MTIME: float = CONFIG_PATH.stat().st_mtime if CONFIG_PATH.exists() else 0.0

def maybe_reload_cfg():
    global CFG, TARGET_W, _CFG_MTIME
    if not CONFIG_PATH.exists(): return
    m = CONFIG_PATH.stat().st_mtime
    if m != _CFG_MTIME:
        CFG = load_cfg()
        TARGET_W = int(CFG["global"]["target_w"])
        _CFG_MTIME = m
        # _slot_ema.clear(); _slot_last.clear(); _slot_streak.clear()

# =============== Image helpers ===============
def decode_jpeg(buf: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(buf, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def encode_jpeg(img: np.ndarray, q: int = 88) -> bytes:
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return bytes(enc) if ok else b""

def _resize_adaptive(img_bgr: np.ndarray) -> np.ndarray:
    H, W = img_bgr.shape[:2]
    if W <= TARGET_W: return img_bgr.copy()
    scale = TARGET_W / float(W)
    return cv2.resize(img_bgr, (TARGET_W, int(round(H*scale))), interpolation=cv2.INTER_AREA)

def _beautify(img: np.ndarray) -> np.ndarray:
    out = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    L,a,b = cv2.split(lab)
    L = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(L)
    return cv2.cvtColor(cv2.merge([L,a,b]), cv2.COLOR_LAB2BGR)

# =============== Detectors ===============
def _wb_grayworld(bgr: np.ndarray) -> np.ndarray:
    eps = 1e-6
    m = bgr.reshape(-1,3).mean(axis=0) + eps
    scale = m.mean()/m
    return (bgr.astype(np.float32)*scale).clip(0,255).astype(np.uint8)

def _preprocess_roi(bgr: np.ndarray) -> np.ndarray:
    bgr = _wb_grayworld(bgr)
    bgr = cv2.bilateralFilter(bgr, 5, 40, 40)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    v = cv2.createCLAHE(2.0, (8,8)).apply(v)
    return cv2.cvtColor(cv2.merge([h,s,v]), cv2.COLOR_HSV2BGR)

def det_presence_box(roi_bgr: np.ndarray, params: Dict) -> int:
    roi_bgr = _preprocess_roi(roi_bgr)
    cmin = tuple(params.get("color_min", (140,170,160)))
    cmax = tuple(params.get("color_max", (220,255,240)))
    min_fill = float(params.get("min_fill_ratio", 0.18))
    mask = cv2.inRange(roi_bgr, np.array(cmin,np.uint8), np.array(cmax,np.uint8))
    fill = float((mask>0).sum())/float(mask.size)
    return 1 if fill >= min_fill else 0

def det_blue_blobs(roi_bgr: np.ndarray, params: Dict) -> int:
    roi_bgr = _preprocess_roi(roi_bgr)
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h1 = int(params.get("h_low",95)); h2 = int(params.get("h_high",135))
    s_low = int(params.get("s_low",40)); v_low = int(params.get("v_low",35))
    mask = cv2.inRange(hsv, (h1,s_low,v_low), (h2,255,255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3,3),np.uint8),1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8),1)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = int(params.get("min_blob_area",1200))
    spans=[]
    for c in cnts:
        if cv2.contourArea(c) < min_area: continue
        x,y,w,h = cv2.boundingRect(c)
        spans.append((x,x+w))
    if not spans: return 0
    spans.sort(key=lambda t:t[0])
    merge = int(params.get("merge_close",18))
    out=[]; cur=list(spans[0])
    for a,b in spans[1:]:
        if a<=cur[1]+merge: cur[1]=max(cur[1],b)
        else: out.append(tuple(cur)); cur=[a,b]
    out.append(tuple(cur))
    return len(out)

def det_red_blobs(roi_bgr: np.ndarray, params: Dict) -> int:
    """Count distinct red packs by clustering red regions along X."""
    roi_bgr = _preprocess_roi(roi_bgr)
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    # Red wraps around 0 deg ⇒ two bands (OpenCV H in [0,179])
    h1_low  = int(params.get("h1_low", 0))
    h1_high = max(0, min(179, int(params.get("h1_high", 12))))
    h2_low  = int(params.get("h2_low", 168))
    h2_high = max(0, min(179, int(params.get("h2_high", 179))))
    s_low   = int(params.get("s_low", 60))
    v_low   = int(params.get("v_low", 55))

    m1 = cv2.inRange(hsv, (h1_low, s_low, v_low), (h1_high, 255, 255))
    m2 = cv2.inRange(hsv, (h2_low, s_low, v_low), (h2_high, 255, 255))
    mask = cv2.bitwise_or(m1, m2)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3,3),np.uint8), 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8), 1)

    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = int(params.get("min_blob_area", 1400))
    xs = []
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        x,y,w,h = cv2.boundingRect(c)
        xs.append(x + w/2.0)

    if not xs:
        return 0

    xs.sort()
    x_gap = int(params.get("x_gap", 34))
    count = 1
    last = xs[0]
    for cx in xs[1:]:
        if cx - last >= x_gap:
            count += 1
            last = cx
    return count

def _count_horizontal_blobs(roi: np.ndarray, params: Dict) -> int:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),0)
    edges = cv2.Canny(gray, 50, 140)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8), 1)
    cnts,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_min = int(params.get("h_min",36)); h_max = int(params.get("h_max",92))
    min_ar = float(params.get("min_ar",1.6))
    centers=[]
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if h < h_min or h > h_max: continue
        ar = w/float(h+1e-6)
        if ar < min_ar: continue
        centers.append(y + h/2.0)
    if not centers: return 0
    centers = sorted(centers)
    merge_y = int(params.get("merge_y",28))
    count=1; last=centers[0]
    for cy in centers[1:]:
        if cy - last >= merge_y: count += 1; last = cy
    return count

def det_mentos(roi_bgr: np.ndarray, params: Dict) -> int:
    roi = _preprocess_roi(roi_bgr)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(7,7),1.5)
    min_r = int(params.get("min_radius",12))
    max_r = int(params.get("max_radius",36))
    min_dist = int(params.get("min_dist",38))
    p1 = int(params.get("circle_p1",140))
    p2 = int(params.get("circle_p2",255))
    x_max = float(params.get("x_max_frac",0.8)) * roi.shape[1]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2,
                               minDist=min_dist, param1=p1, param2=p2,
                               minRadius=min_r, maxRadius=max_r)
    circ_count = 0
    if circles is not None:
        circ = np.uint16(np.around(circles[0, :]))
        circ = [c for c in circ if c[0] < x_max]
        ys = sorted([c[1] for c in circ])
        if ys:
            merge_y = int(params.get("merge_y",28))
            circ_count = 1; last = ys[0]
            for yy in ys[1:]:
                if yy - last >= merge_y: circ_count += 1; last = yy
    blob_count = _count_horizontal_blobs(roi, params)
    return circ_count if circ_count>0 else blob_count

DETECTORS = {
    "presence_box": det_presence_box,
    "blue_blobs":   det_blue_blobs,
    "red_blobs":    det_red_blobs,
    "mentos":       det_mentos,
}

# =============== Stability & overlay ===============
_slot_ema: Dict[str, float] = {}
_slot_last: Dict[str, int] = {}
_slot_streak: Dict[str, int] = {}

def stable_ema(slot_key: str, raw: int, alpha: float=0.55, confirm: int=2) -> int:
    ema = _slot_ema.get(slot_key, float(raw))
    ema = alpha*raw + (1.0-alpha)*ema
    _slot_ema[slot_key] = ema
    cand = int(round(ema))
    last = _slot_last.get(slot_key, cand)
    if cand == last:
        _slot_streak[slot_key] = 0
        return last
    st = _slot_streak.get(slot_key, 0) + 1
    _slot_streak[slot_key] = st
    if st >= confirm:
        _slot_last[slot_key] = cand
        _slot_streak[slot_key] = 0
        return cand
    return last

def resolve_slot_roi(canvas: np.ndarray, slot: Dict[str, Any]) -> Tuple[int,int,int,int]:
    H, W = canvas.shape[:2]
    x = int(slot["x"]); y = int(slot["y"]); w = int(slot["w"]); h = int(slot["h"])
    pad_px  = int(slot.get("pad_px", 0))
    pad_pct = float(slot.get("pad_pct", 0.0))
    pad = pad_px if pad_px > 0 else int(round(min(w, h) * pad_pct))
    x -= pad; y -= pad; w += 2*pad; h += 2*pad
    x = max(0, min(x, W-1)); y = max(0, min(y, H-1))
    w = max(1, min(w, W - x)); h = max(1, min(h, H - y))
    return x, y, w, h

# ---- draw-only rectangle (negative shrinks, positive expands) ----
def resolve_draw_rect(canvas: np.ndarray, slot: Dict[str, Any]) -> Tuple[int,int,int,int]:
    x, y, w, h = resolve_slot_roi(canvas, slot)

    dpad_px_raw  = slot.get("draw_pad_px", 0)
    dpad_pct_raw = slot.get("draw_pad_pct", -0.02)  # default: very small shrink

    if dpad_px_raw:
        amt = int(abs(int(dpad_px_raw)))
        sign = 1 if int(dpad_px_raw) >= 0 else -1
    else:
        amt = int(round(min(w, h) * abs(float(dpad_pct_raw))))
        sign = 1 if float(dpad_pct_raw) >= 0 else -1

    if sign > 0:  # expand
        x -= amt; y -= amt; w += 2*amt; h += 2*amt
    else:         # shrink
        w = max(1, w - 2*amt)
        h = max(1, h - 2*amt)
        x += amt; y += amt

    dx, dy = slot.get("draw_nudge", [0, 0])
    x += int(dx); y += int(dy)

    H, W = canvas.shape[:2]
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return x, y, w, h

def count_image_total(img_bgr: np.ndarray):
    maybe_reload_cfg()
    canvas = _resize_adaptive(img_bgr)
    total = 0; per_slot = []
    for s in CFG["slots"]:
        x,y,w,h = resolve_slot_roi(canvas, s)
        roi = canvas[y:y+h, x:x+w]
        det_fn = DETECTORS.get(str(s.get("detector","mentos")).lower(), det_mentos)
        raw  = int(det_fn(roi, s.get("params", {})))
        stab = stable_ema(s.get("key", s["id"]), raw, alpha=0.55, confirm=2)
        total += stab
        per_slot.append((s, raw, stab))
    return total, per_slot, canvas

def _separate_draw_rects(canvas: np.ndarray,
                         rects: List[Tuple[int,int,int,int]],
                         margin: int = 12,
                         iters: int = 24) -> List[Tuple[int,int,int,int]]:
    """Gently repels overlapping rectangles so they don't collide."""
    H, W = canvas.shape[:2]
    rects = [list(r) for r in rects]
    for _ in range(iters):
        moved = False
        for i in range(len(rects)):
            x, y, w, h = rects[i]
            for j in range(i+1, len(rects)):
                x2, y2, w2, h2 = rects[j]
                l1, r1, t1, b1 = x, x+w, y, y+h
                l2, r2, t2, b2 = x2, x2+w2, y2, y2+h2
                overlap_x = min(r1, r2) - max(l1, l2) - margin
                overlap_y = min(b1, b2) - max(t1, t2) - margin
                if overlap_x > 0 and overlap_y > 0:
                    if overlap_x < overlap_y:
                        dx = (overlap_x / 2.0) + 1
                        if l1 < l2: x, x2 = x-dx, x2+dx
                        else:       x, x2 = x+dx, x2-dx
                    else:
                        dy = (overlap_y / 2.0) + 1
                        if t1 < t2: y, y2 = y-dy, y2+dy
                        else:       y, y2 = y+dy, y2-dy
                    x  = max(0, min(int(round(x)),  W - w))
                    y  = max(0, min(int(round(y)),  H - h))
                    x2 = max(0, min(int(round(x2)), W - w2))
                    y2 = max(0, min(int(round(y2)), H - h2))
                    rects[i] = [x, y, w, h]
                    rects[j] = [x2, y2, w2, h2]
                    moved = True
        if not moved:
            break
    return [tuple(r) for r in rects]

def draw_overlay(img_bgr: np.ndarray):
    maybe_reload_cfg()
    img_bgr = _resize_adaptive(img_bgr)
    canvas  = _beautify(img_bgr.copy())
    total, per_slot, _ = count_image_total(img_bgr)

    init_rects = [resolve_draw_rect(canvas, s) for (s, _, _) in per_slot]
    rects = _separate_draw_rects(canvas, init_rects, margin=14, iters=24)

    placed_labels: List[Tuple[int,int,int,int]] = []
    for (s, raw, stab), (dx, dy, dw, dh) in zip(per_slot, rects):
        cv2.rectangle(canvas, (dx, dy), (dx+dw, dy+dh), (0, 200, 255), 4)
        label = f'{s["id"]} [{s.get("key","")}] = {stab}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

        cand = (dx, max(0, dy - th - 10), tw + 12, th + 10)
        def coll(ax, ay, aw, ah):
            for (bx, by, bw, bh) in placed_labels:
                if not (ax+aw < bx or bx+bw < ax or ay+ah < by or by+bh < ay):
                    return True
            return False
        if coll(*cand):
            cand = (dx, min(canvas.shape[0]-th-10, dy + dh + 10), tw + 12, th + 10)
            if coll(*cand):
                cx, cy, cw, ch = cand
                for _ in range(8):
                    cy = min(canvas.shape[0]-ch-2, cy + 8)
                    cand = (cx, cy, cw, ch)
                    if not coll(*cand): break
        cx, cy, cw, ch = cand
        cv2.rectangle(canvas, (cx, cy), (cx+cw, cy+ch), (0, 200, 255), -1)
        cv2.putText(canvas, label, (cx + 6, cy + ch - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20,20,20), 2, cv2.LINE_AA)
        placed_labels.append(cand)

    cv2.putText(canvas, f'Total: {total}', (22,58),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0,0,0), 6, cv2.LINE_AA)
    cv2.putText(canvas, f'Total: {total}', (22,58),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (35,235,35), 4, cv2.LINE_AA)
    return canvas, per_slot, total

# =============== Adaptive poller ===============
class CapturePoller:
    def __init__(self, url: str):
        self.url = url
        self.sess = requests.Session()
        self.lock = threading.Lock()
        self.running = False
        self.last_overlay_jpg: bytes = b""
        self.last_frame: Optional[np.ndarray] = None
        self.bg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=16, detectShadows=False)
        self.state = "IDLE"
        self.t_calm_start = 0.0
        self.settle_ms = 600
        self.motion_low = 5000
        self.motion_high = 15000
        self.base_period = 1.0     # 1 FPS idle
        self.active_period = 0.14  # ~7 FPS moving

    def start(self):
        if self.running: return
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self.running = False

    def _run(self):
        last_count_time = 0.0
        while self.running:
            t0 = time.time()
            try:
                r = self.sess.get(self.url, timeout=4)
                if r.status_code != 200:
                    time.sleep(self.base_period); continue
                img = decode_jpeg(r.content)
                if img is None:
                    time.sleep(self.base_period); continue

                with self.lock:
                    self.last_frame = img.copy()

                small = _resize_adaptive(img)
                fg = self.bg.apply(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
                fg = cv2.medianBlur(fg, 5)
                motion = int((fg > 40).sum())
                now = time.time()

                if self.state == "IDLE":
                    if motion >= self.motion_high: self.state = "MOVING"
                elif self.state == "MOVING":
                    if motion < self.motion_low:
                        self.state = "SETTLING"; self.t_calm_start = now
                elif self.state == "SETTLING":
                    if motion >= self.motion_high:
                        self.state = "MOVING"
                    elif (now - self.t_calm_start)*1000 >= self.settle_ms and (now - last_count_time) > 0.8:
                        ovl, per_slot, total = draw_overlay(img)
                        jpg = encode_jpeg(ovl, 88)
                        with self.lock:
                            self.last_overlay_jpg = jpg
                        for s, raw, stab in per_slot:
                            db_set_count(s.get("key", s["id"]), int(stab))
                        last_count_time = now
                        self.state = "IDLE"

                # keep overlay fresh (~5 Hz)
                if int(now*5) % 1 == 0:
                    ovl, _, _ = draw_overlay(img)
                    with self.lock:
                        self.last_overlay_jpg = encode_jpeg(ovl, 80)

            except Exception:
                pass

            period = self.active_period if self.state in ("MOVING","SETTLING") else self.base_period
            dt = time.time() - t0
            if period - dt > 0: time.sleep(period - dt)

poller = CapturePoller(ESP32_CAPTURE_URL)

# ================= Routes =================
@app.on_event("startup")
def _boot():
    db_init()
    poller.start()

@app.on_event("shutdown")
def _shutdown():
    poller.stop()

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "home.html",
        {"request": request, "shelves": db_get_shelves()}
    )

@app.get("/partials/shelves", response_class=HTMLResponse)
def shelves_partial(request: Request):
    resp = templates.TemplateResponse(
        "shelves.html",
        {"request": request, "shelves": db_get_shelves()}
    )
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp

@app.get("/api/shelves")
def api_shelves():
    return {"shelves": db_get_shelves(), "ts": int(time.time())}

@app.get("/shelf/{key}", response_class=HTMLResponse)
def shelf_page(request: Request, key: str):
    return templates.TemplateResponse("live.html", {"request": request, "key": key})

@app.get("/live", response_class=HTMLResponse)
def live_page(request: Request):
    return shelf_page(request, "any")

@app.get("/live.jpg")
def live_jpg():
    with poller.lock:
        jpg = poller.last_overlay_jpg
    if not jpg:
        blank = np.zeros((480,640,3), np.uint8)
        cv2.putText(blank,"Connecting...",(30,250), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
        jpg = encode_jpeg(blank, 80)
    return StreamingResponse(io.BytesIO(jpg), media_type="image/jpeg")

@app.post("/analyze")
async def analyze(frame: UploadFile = File(...), shelf_key: Optional[str] = Form(None)):
    data = await frame.read()
    img = decode_jpeg(data)
    if img is None: return JSONResponse({"error":"invalid frame"}, status_code=400)
    _recent_frames.append((time.time(), data))
    if len(_recent_frames) > GALLERY_MAX: _recent_frames.pop(0)
    ovl, per_slot, total = draw_overlay(img)
    for s, raw, stab in per_slot:
        db_set_count(s.get("key", s["id"]), int(stab))
    return {"ok":True, "total": int(total),
            "slots":[{"id":s["id"],"key":s.get("key",""),"raw":int(raw),"stable":int(stab)} for s,raw,stab in per_slot]}

@app.get("/debug/slot")
def debug_slot(id: str = "S-MENTOS"):
    with poller.lock:
        frame = None if poller.last_frame is None else poller.last_frame.copy()
    if frame is None:
        return JSONResponse({"error":"no frame yet"}, status_code=400)

    slot = next((s for s in CFG["slots"] if s.get("id")==id), None)
    if not slot: return JSONResponse({"error":"unknown slot"}, status_code=404)

    x,y,w,h = resolve_slot_roi(frame, slot)
    roi = frame[y:y+h, x:x+w].copy()
    p = slot.get("params", {})

    g = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY),(7,7),1.5)
    min_r = int(p.get("min_radius",12)); max_r = int(p.get("max_radius",36))
    min_dist = int(p.get("min_dist",38))
    p1 = int(p.get("circle_p1",140)); p2 = int(p.get("circle_p2",255))
    x_max = int(float(p.get("x_max_frac",0.8)) * roi.shape[1])
    view = roi.copy()
    cir = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2,
                           minDist=min_dist, param1=p1, param2=p2,
                           minRadius=min_r, maxRadius=max_r)
    if cir is not None:
        for c in np.uint16(np.around(cir[0, :])):
            if c[0] < x_max:
                cv2.circle(view, (c[0], c[1]), c[2], (0,255,255), 2)
                cv2.circle(view, (c[0], c[1]), 2, (0,255,255), 2)
    cv2.line(view, (x_max,0), (x_max,view.shape[0]-1), (0,255,255), 1)

    g2 = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY),(5,5),0)
    edges = cv2.Canny(g2, 50, 140)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8), 1)
    cnts,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_min = int(p.get("h_min",36)); h_max = int(p.get("h_max",92)); min_ar = float(p.get("min_ar",1.6))
    for c in cnts:
        x2,y2,w2,h2 = cv2.boundingRect(c)
        if h2 < h_min or h2 > h_max: continue
        if (w2/(h2+1e-6)) < min_ar: continue
        cv2.rectangle(view, (x2,y2), (x2+w2, y2+h2), (255,200,0), 2)

    return StreamingResponse(io.BytesIO(encode_jpeg(view, 88)), media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    db_init()
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT","8000")), reload=True)
