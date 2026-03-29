#!/usr/bin/env python3
"""
HIKVISION Notch Inspector – Capture & Detect GUI
-------------------------------------------------
Double-click the built EXE or run:  python src/hikvision_capture_gui.py

Layout
------
  Top row   : Camera settings (IP / port / user / pass / channel) + Connect/Stop
  Middle    : LEFT = live preview stream  |  RIGHT = last capture with detection
  Centre    : Big green  [📷  CAPTURE & INSPECT]  button
  Bottom    : Scrollable history table of every capture
"""

from __future__ import annotations

import csv
import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageDraw, ImageFont, ImageTk
from ultralytics import YOLO

import multiprocessing
from verdict_rules import strict_verdict_from_counts

# ── Constants ──────────────────────────────────────────────────────────────────
APP_TITLE = "Harmony - JD Block Notch Detection"
APP_VERSION = "1.0"

CLASS_NAMES = {
    0: "circle",
    1: "notch_toward_hole",
    2: "notch_not_toward_hole",
}

COLORS_BGR = {           # OpenCV BGR
    0: (0, 220, 0),
    1: (0, 220, 255),
    2: (0, 0, 255),
}

VERDICT_HEX = {
    "OK":       "#10b981",   # green
    "NOT_OKAY": "#ef4444",   # red
    "UNKNOWN":  "#9ca3af",   # grey
}

DEFAULT_CFG: dict[str, Any] = {
    "camera_ip":   "192.168.1.64",
    "camera_port": 554,
    "camera_user": "admin",
    "camera_pass": "password",
    "camera_channel": 101,
    "rtsp_url":    "",          # if non-empty, overrides the builder above
    "weights":     "models/best.pt",
    "imgsz":       960,
    "conf":        0.25,
    "device":      "cpu",
    "save_dir":    "results/captures",
    "rtsp_tcp":    True,
    "auto_connect": False,
}

# ── Path helpers ───────────────────────────────────────────────────────────────

def runtime_base() -> Path:
    """Root directory – works both for .py and frozen .exe."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


BASE = runtime_base()


def resolve_weights(raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    for base in (BASE, Path.cwd()):
        c = (base / p).resolve()
        if c.exists():
            return c
    raise FileNotFoundError(f"Model weights not found: {raw}")


# ── RTSP helpers ───────────────────────────────────────────────────────────────

def build_rtsp_url(ip: str, port: int, user: str, pwd: str, channel: int) -> str:
    return f"rtsp://{user}:{pwd}@{ip}:{port}/Streaming/Channels/{channel}"


def open_cap(url: str, rtsp_tcp: bool) -> cv2.VideoCapture:
    source = int(url) if url.strip().isdigit() else url
    if isinstance(source, str) and rtsp_tcp and source.lower().startswith("rtsp://"):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    
    if isinstance(source, str):
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)
        
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


# ── Inference ──────────────────────────────────────────────────────────────────

def infer_frame(model: YOLO, frame, imgsz: int, conf: float, device: str):
    """Run YOLO on *frame* and return (annotated_bgr, verdict, counts)."""
    pred = model.predict(source=frame, imgsz=imgsz, conf=conf,
                         device=device, verbose=False)[0]
    annotated = frame.copy()
    counts = {0: 0, 1: 0, 2: 0}

    if pred.boxes is not None:
        for box, cls, score in zip(pred.boxes.xyxy, pred.boxes.cls, pred.boxes.conf):
            cls_id = int(cls.item())
            conf_v = float(score.item())
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            color = COLORS_BGR.get(cls_id, (255, 255, 255))
            label = f"{CLASS_NAMES.get(cls_id, str(cls_id))} {conf_v:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
            counts[cls_id] = counts.get(cls_id, 0) + 1

    verdict = strict_verdict_from_counts(counts).verdict

    return annotated, verdict, counts


# ── CSV event log ──────────────────────────────────────────────────────────────

def ensure_log(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["timestamp", "verdict", "circle", "toward", "away", "image"])


def append_log(path: Path, verdict: str, counts: dict, img_path: Path) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            datetime.now().isoformat(timespec="seconds"),
            verdict,
            counts.get(0, 0), counts.get(1, 0), counts.get(2, 0),
            str(img_path),
        ])


# ── Main GUI class ─────────────────────────────────────────────────────────────

class HikvisionInspector:
    # ── init ──────────────────────────────────────────────────────────────
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(f"{APP_TITLE}  v{APP_VERSION}")
        self.root.geometry("1380x860")
        self.root.minsize(1100, 700)
        self.root.configure(bg="#1e1e2e")

        self.model:  YOLO | None = None
        self.cap:    cv2.VideoCapture | None = None
        self.running = False          # live preview thread
        self.capture_in_progress = False

        self.last_live_frame = None   # raw BGR
        self.last_result_img = None   # annotated BGR
        self.last_verdict = "UNKNOWN"
        self.last_counts  = {0: 0, 1: 0, 2: 0}

        self.photo_live:   ImageTk.PhotoImage | None = None
        self.photo_result: ImageTk.PhotoImage | None = None

        self.history: list[dict] = []   # [{ts, verdict, circle, toward, away, img}]
        self.capture_count = 0

        self.cfg = self._load_cfg()
        self._build_ui()
        self._apply_cfg_to_ui()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        if self.cfg.get("auto_connect", False):
            self.root.after(600, self._connect)

    # ── Config ────────────────────────────────────────────────────────────
    def _cfg_path(self) -> Path:
        return (BASE / "client_config.json").resolve()

    def _load_cfg(self) -> dict:
        p = self._cfg_path()
        if p.exists():
            try:
                payload = json.loads(p.read_text("utf-8"))
                if isinstance(payload, dict):
                    merged = dict(DEFAULT_CFG)
                    merged.update(payload)
                    return merged
            except Exception:
                pass
        # try example
        ex = (BASE / "client_config.example.json").resolve()
        if ex.exists():
            try:
                payload = json.loads(ex.read_text("utf-8"))
                if isinstance(payload, dict):
                    merged = dict(DEFAULT_CFG)
                    merged.update(payload)
                    p.write_text(json.dumps(merged, indent=2), "utf-8")
                    return merged
            except Exception:
                pass
        p.write_text(json.dumps(DEFAULT_CFG, indent=2), "utf-8")
        return dict(DEFAULT_CFG)

    def _save_cfg(self) -> None:
        self._cfg_path().write_text(
            json.dumps(self.cfg, indent=2), "utf-8")

    # ── UI build ──────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        BG   = "#1e1e2e"
        CARD = "#2a2a3e"
        FG   = "#cdd6f4"
        ACC  = "#89b4fa"   # blue accent

        # ── Title bar ─────────────────────────────────────────────────────
        title_bar = tk.Frame(self.root, bg="#13131f", height=48)
        title_bar.pack(side=tk.TOP, fill=tk.X)
        title_bar.pack_propagate(False)
        tk.Label(title_bar,
                 text=f"🔍  {APP_TITLE}",
                 font=("Segoe UI", 16, "bold"),
                 bg="#13131f", fg=FG).pack(side=tk.LEFT, padx=14, pady=8)
        tk.Label(title_bar,
                 text=f"v{APP_VERSION}",
                 font=("Segoe UI", 10),
                 bg="#13131f", fg="#6c7086").pack(side=tk.LEFT, pady=8)

        # ── Camera settings row ───────────────────────────────────────────
        cam_frame = tk.Frame(self.root, bg=CARD, pady=8, padx=10)
        cam_frame.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(6, 0))

        def lbl(parent, text, col, row=0, sticky="e", padx=(0, 4)):
            tk.Label(parent, text=text, font=("Segoe UI", 9), bg=CARD, fg=FG
                     ).grid(row=row, column=col, sticky=sticky, padx=padx, pady=2)

        def entry(parent, var, width, col, row=0):
            e = tk.Entry(parent, textvariable=var, width=width,
                         bg="#313244", fg=FG, insertbackground=FG,
                         relief=tk.FLAT, font=("Consolas", 10),
                         highlightthickness=1, highlightbackground="#45475a",
                         highlightcolor=ACC)
            e.grid(row=row, column=col, sticky="we", padx=(0, 8), pady=2)
            return e

        # Row 0 : IP / Port / User / Pass / Channel
        lbl(cam_frame, "Camera IP:", 0)
        self.v_ip   = tk.StringVar()
        entry(cam_frame, self.v_ip, 18, 1)

        lbl(cam_frame, "Port:", 2)
        self.v_port = tk.StringVar()
        entry(cam_frame, self.v_port, 6, 3)

        lbl(cam_frame, "Channel:", 4)
        self.v_ch   = tk.StringVar()
        entry(cam_frame, self.v_ch, 6, 5)

        lbl(cam_frame, "Username:", 6)
        self.v_user = tk.StringVar()
        entry(cam_frame, self.v_user, 10, 7)

        lbl(cam_frame, "Password:", 8)
        self.v_pass = tk.StringVar()
        pe = entry(cam_frame, self.v_pass, 14, 9)
        pe.configure(show="*")

        # Row 1 : RTSP URL override / Conf / Imgsz / RTSP-TCP
        lbl(cam_frame, "Source (RTSP/USB):", 0, row=1)
        self.v_rtsp = tk.StringVar()
        self.rtsp_e = ttk.Combobox(cam_frame, textvariable=self.v_rtsp, width=48, font=("Consolas", 9))
        self.rtsp_e.grid(row=1, column=1, columnspan=4, sticky="we", padx=(0, 8), pady=2)
        
        self.btn_detect_usb = tk.Button(cam_frame, text="Detect USB", bg="#45475a", fg=FG, command=self._detect_usb, relief=tk.FLAT, font=("Segoe UI", 9), cursor="hand2")
        self.btn_detect_usb.grid(row=1, column=5, padx=(0, 8), pady=2)
        
        tk.Label(cam_frame, text="(leave blank to auto-build from fields above)",
                 font=("Segoe UI", 8), bg=CARD, fg="#6c7086"
                 ).grid(row=1, column=6, columnspan=2, sticky="w")

        lbl(cam_frame, "Conf:", 8, row=1)
        self.v_conf  = tk.StringVar()
        entry(cam_frame, self.v_conf, 7, 9, row=1)

        # Row 2 : Imgsz / TCP toggle / Model / Buttons
        lbl(cam_frame, "Imgsz:", 0, row=2)
        self.v_imgsz = tk.StringVar()
        entry(cam_frame, self.v_imgsz, 7, 1, row=2)

        self.v_tcp = tk.BooleanVar(value=True)
        tk.Checkbutton(cam_frame, text="RTSP TCP",
                       variable=self.v_tcp,
                       bg=CARD, fg=FG, selectcolor="#313244",
                       activebackground=CARD, activeforeground=FG,
                       font=("Segoe UI", 9)
                       ).grid(row=2, column=2, columnspan=2, sticky="w", pady=2)

        # Buttons
        btn_style = dict(relief=tk.FLAT, font=("Segoe UI", 10, "bold"),
                         cursor="hand2", padx=12, pady=4)
        self.btn_connect = tk.Button(cam_frame, text="▶  Connect",
                                     bg="#40a02b", fg="white",
                                     command=self._connect, **btn_style)
        self.btn_connect.grid(row=2, column=4, padx=(0, 6), pady=2)

        self.btn_stop = tk.Button(cam_frame, text="■  Stop",
                                  bg="#e64553", fg="white",
                                  command=self._stop, state=tk.DISABLED,
                                  **btn_style)
        self.btn_stop.grid(row=2, column=5, padx=(0, 6), pady=2)

        tk.Button(cam_frame, text="💾  Save Config",
                  bg="#45475a", fg=FG, command=self._save_cfg_from_ui,
                  **btn_style).grid(row=2, column=6, padx=(0, 6), pady=2)

        tk.Button(cam_frame, text="📂  Open Results",
                  bg="#45475a", fg=FG, command=self._open_results,
                  **btn_style).grid(row=2, column=7, padx=(0, 6), pady=2)

        cam_frame.columnconfigure(1, weight=1)

        # ── Status bar ────────────────────────────────────────────────────
        self.v_status = tk.StringVar(value="Ready – configure camera and click Connect")
        status_bar = tk.Frame(self.root, bg="#13131f", height=24)
        status_bar.pack(side=tk.TOP, fill=tk.X)
        status_bar.pack_propagate(False)
        tk.Label(status_bar, textvariable=self.v_status,
                 font=("Consolas", 9), bg="#13131f", fg="#6c7086",
                 anchor="w").pack(fill=tk.X, padx=8)

        # ── Main video area ────────────────────────────────────────────────
        main_area = tk.Frame(self.root, bg=BG)
        main_area.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=4)

        # LEFT: live preview
        left_card = tk.Frame(main_area, bg=CARD, bd=0)
        left_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))

        tk.Label(left_card, text="📡  LIVE PREVIEW",
                 font=("Segoe UI", 11, "bold"),
                 bg=CARD, fg=ACC).pack(side=tk.TOP, anchor="w", padx=8, pady=(6, 2))

        self.live_lbl = tk.Label(left_card, bg="#13131f", anchor="center",
                                  text="No stream\nconnected",
                                  fg="#45475a", font=("Segoe UI", 14))
        self.live_lbl.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        # RIGHT: last result
        right_card = tk.Frame(main_area, bg=CARD, bd=0)
        right_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))

        tk.Label(right_card, text="🔍  LAST CAPTURE RESULT",
                 font=("Segoe UI", 11, "bold"),
                 bg=CARD, fg=ACC).pack(side=tk.TOP, anchor="w", padx=8, pady=(6, 2))

        # Verdict banner
        self.v_verdict = tk.StringVar(value="VERDICT: —")
        self.verdict_lbl = tk.Label(right_card,
                                     textvariable=self.v_verdict,
                                     font=("Segoe UI", 20, "bold"),
                                     bg=CARD,
                                     fg=VERDICT_HEX["UNKNOWN"])
        self.verdict_lbl.pack(side=tk.TOP, anchor="w", padx=8, pady=(2, 4))

        # Detection stats
        self.v_stats = tk.StringVar(value="")
        tk.Label(right_card, textvariable=self.v_stats,
                 font=("Consolas", 10), bg=CARD, fg="#cba6f7"
                 ).pack(side=tk.TOP, anchor="w", padx=10, pady=(0, 2))

        self.result_lbl = tk.Label(right_card, bg="#13131f", anchor="center",
                                    text="No capture yet",
                                    fg="#45475a", font=("Segoe UI", 14))
        self.result_lbl.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        # ── CAPTURE button ────────────────────────────────────────────────
        capture_bar = tk.Frame(self.root, bg=BG, pady=6)
        capture_bar.pack(side=tk.TOP, fill=tk.X)

        self.btn_capture = tk.Button(
            capture_bar,
            text="📷   CAPTURE & INSPECT",
            font=("Segoe UI", 16, "bold"),
            bg="#40a02b", fg="white",
            activebackground="#2e8b1e", activeforeground="white",
            relief=tk.FLAT,
            padx=40, pady=14,
            cursor="hand2",
            command=self._capture,
            state=tk.DISABLED,
        )
        self.btn_capture.pack(expand=True)

        # ── History table ──────────────────────────────────────────────────
        hist_outer = tk.Frame(self.root, bg=CARD)
        hist_outer.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(2, 6))

        tk.Label(hist_outer, text="📋  Capture History",
                 font=("Segoe UI", 10, "bold"),
                 bg=CARD, fg=ACC).pack(side=tk.TOP, anchor="w", padx=8, pady=(4, 0))

        cols = ("timestamp", "verdict", "circle", "toward_hole", "not_toward", "saved")
        self.tree = ttk.Treeview(hist_outer, columns=cols, show="headings",
                                  height=5, selectmode="browse")

        col_cfg = {
            "timestamp":   ("Timestamp",        160, "w"),
            "verdict":     ("Verdict",            90, "center"),
            "circle":      ("Circle",             60, "center"),
            "toward_hole": ("Toward Hole",        90, "center"),
            "not_toward":  ("Not Toward",         90, "center"),
            "saved":       ("Saved Image",        320, "w"),
        }
        for cid, (hdr, width, anchor) in col_cfg.items():
            self.tree.heading(cid, text=hdr)
            self.tree.column(cid, width=width, anchor=anchor, stretch=(cid == "saved"))

        # Tag colours for OK / NOT_OKAY / UNKNOWN rows
        style = ttk.Style()
        style.configure("Treeview",
                         background="#1e1e2e",
                         foreground="#cdd6f4",
                         fieldbackground="#1e1e2e",
                         rowheight=22,
                         font=("Consolas", 9))
        style.configure("Treeview.Heading",
                         background="#313244",
                         foreground="#cdd6f4",
                         font=("Segoe UI", 9, "bold"))
        self.tree.tag_configure("OK",       foreground="#a6e3a1")
        self.tree.tag_configure("NOT_OKAY", foreground="#f38ba8")
        self.tree.tag_configure("UNKNOWN",  foreground="#9ca3af")

        sb = ttk.Scrollbar(hist_outer, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 4), pady=4)
        self.tree.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0), pady=4)

    # ── UI helpers ────────────────────────────────────────────────────────
    def _apply_cfg_to_ui(self) -> None:
        c = self.cfg
        self.v_ip.set(str(c.get("camera_ip",   "192.168.1.64")))
        self.v_port.set(str(c.get("camera_port", 554)))
        self.v_ch.set(str(c.get("camera_channel", 101)))
        self.v_user.set(str(c.get("camera_user", "admin")))
        self.v_pass.set(str(c.get("camera_pass", "")))
        self.v_rtsp.set(str(c.get("rtsp_url", "")))
        self.v_conf.set(str(c.get("conf", 0.25)))
        self.v_imgsz.set(str(c.get("imgsz", 960)))
        self.v_tcp.set(bool(c.get("rtsp_tcp", True)))

    def _read_ui_to_cfg(self) -> None:
        self.cfg["camera_ip"]      = self.v_ip.get().strip()
        self.cfg["camera_port"]    = int(self.v_port.get().strip() or 554)
        self.cfg["camera_channel"] = int(self.v_ch.get().strip() or 101)
        self.cfg["camera_user"]    = self.v_user.get().strip()
        self.cfg["camera_pass"]    = self.v_pass.get()
        self.cfg["rtsp_url"]       = self.v_rtsp.get().strip()
        self.cfg["conf"]           = float(self.v_conf.get().strip() or 0.25)
        self.cfg["imgsz"]          = int(self.v_imgsz.get().strip() or 960)
        self.cfg["rtsp_tcp"]       = bool(self.v_tcp.get())

    def _get_rtsp_url(self) -> str:
        override = self.v_rtsp.get().strip()
        if override:
            return override
        return build_rtsp_url(
            ip=self.v_ip.get().strip(),
            port=int(self.v_port.get().strip() or 554),
            user=self.v_user.get().strip(),
            pwd=self.v_pass.get(),
            channel=int(self.v_ch.get().strip() or 101),
        )

    def _set_status(self, msg: str) -> None:
        self.v_status.set(msg)

    def _set_verdict_ui(self, verdict: str, counts: dict) -> None:
        emoji = {"OK": "✅", "NOT_OKAY": "❌", "UNKNOWN": "❓"}.get(verdict, "❓")
        self.v_verdict.set(f"VERDICT:  {emoji}  {verdict}")
        self.verdict_lbl.configure(
            fg=VERDICT_HEX.get(verdict, VERDICT_HEX["UNKNOWN"]))
        self.v_stats.set(
            f"circle: {counts.get(0,0)}   "
            f"toward_hole: {counts.get(1,0)}   "
            f"not_toward_hole: {counts.get(2,0)}"
        )

    def _save_cfg_from_ui(self) -> None:
        self._read_ui_to_cfg()
        self._save_cfg()
        self._set_status("Config saved.")

    def _open_results(self) -> None:
        d = Path(str(self.cfg.get("save_dir", "results/captures")))
        if not d.is_absolute():
            d = BASE / d
        d.mkdir(parents=True, exist_ok=True)
        import platform
        import subprocess
        if platform.system() == "Windows":
            os.startfile(str(d))    # type: ignore[attr-defined]
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", str(d)])
        else:
            subprocess.Popen(["xdg-open", str(d)])

    def _detect_usb(self) -> None:
        self.btn_detect_usb.config(state=tk.DISABLED, text="Detecting...")
        self.root.update()
        
        available = []
        for i in range(5):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_V4L2)
            if not cap.isOpened():
                cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available.append(str(i))
                cap.release()
                
        if available:
            self.rtsp_e['values'] = available
            if self.v_rtsp.get() not in available:
                self.v_rtsp.set(available[0])
            messagebox.showinfo("Cameras Detected", f"Found USB cameras at indices: {', '.join(available)}")
        else:
            self.rtsp_e['values'] = []
            messagebox.showwarning("No Cameras", "No USB cameras detected.")
            
        self.btn_detect_usb.config(state=tk.NORMAL, text="Detect USB")

    # ── Model loader (runs in thread to avoid freezing UI) ────────────────
    def _ensure_model_loaded(self) -> None:
        """Load model if not already loaded.  Called from background thread."""
        if self.model is not None:
            return
        raw = str(self.cfg.get("weights", "models/best.pt"))
        try:
            weights_path = resolve_weights(raw)
        except FileNotFoundError as exc:
            raise RuntimeError(str(exc)) from exc
        self.root.after(0, self._set_status, f"Loading model: {weights_path.name} …")
        self.model = YOLO(str(weights_path))

    # ── Connect / disconnect ──────────────────────────────────────────────
    def _connect(self) -> None:
        if self.running:
            return
        self._read_ui_to_cfg()
        url = self._get_rtsp_url()
        self._set_status(f"Connecting to: {url} …")
        self.btn_connect.configure(state=tk.DISABLED)
        threading.Thread(target=self._connect_thread, args=(url,),
                         daemon=True).start()

    def _connect_thread(self, url: str) -> None:
        try:
            self._ensure_model_loaded()
            cap = open_cap(url, bool(self.cfg.get("rtsp_tcp", True)))
            if not cap.isOpened():
                raise RuntimeError(
                    f"Cannot open stream.\n\n"
                    f"URL tried: {url}\n\n"
                    "Check:\n"
                    "  • Camera IP / port / credentials\n"
                    "  • Camera is powered on and on the same network\n"
                    "  • RTSP is enabled in HIKVISION camera settings"
                )
            self.cap = cap
            self.running = True
            self.root.after(0, self._on_connected, url)
            self._live_loop()
        except Exception as exc:
            self.root.after(0, self._on_connect_failed, str(exc))

    def _on_connected(self, url: str) -> None:
        self._set_status(f"Connected ✓  {url}")
        self.btn_stop.configure(state=tk.NORMAL)
        self.btn_connect.configure(state=tk.DISABLED)
        self.btn_capture.configure(state=tk.NORMAL)
        self.live_lbl.configure(text="")

    def _on_connect_failed(self, msg: str) -> None:
        self.running = False
        self.btn_connect.configure(state=tk.NORMAL)
        self.btn_capture.configure(state=tk.DISABLED)
        self._set_status("Connection failed.")
        messagebox.showerror("Connection Failed", msg)

    def _stop(self) -> None:
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_stop.configure(state=tk.DISABLED)
        self.btn_connect.configure(state=tk.NORMAL)
        self.btn_capture.configure(state=tk.DISABLED)
        self.live_lbl.configure(image="", text="No stream\nconnected")
        self.live_lbl.image = None
        self._set_status("Disconnected.")

    # ── Live preview loop (background thread, posts to Tk via after) ──────
    def _live_loop(self) -> None:
        failed = 0
        while self.running and self.cap is not None:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                failed += 1
                if failed > 60:
                    self.root.after(0, self._set_status,
                                    "Stream lost – reconnecting…")
                    try:
                        url = self._get_rtsp_url()
                        self.cap.release()
                        self.cap = open_cap(url, bool(self.cfg.get("rtsp_tcp", True)))
                        failed = 0
                    except Exception:
                        pass
                time.sleep(0.05)
                continue
            failed = 0
            self.last_live_frame = frame.copy()
            self.root.after(0, self._update_live_panel, frame)
            time.sleep(0.033)   # ~30 fps display cap

    def _update_live_panel(self, frame) -> None:
        photo = self._frame_to_photo(frame, self.live_lbl)
        if photo:
            self.photo_live = photo
            self.live_lbl.configure(image=photo)

    # ── Capture & inspect ────────────────────────────────────────────────
    def _capture(self) -> None:
        if self.capture_in_progress:
            return
        frame = self.last_live_frame
        if frame is None:
            messagebox.showinfo("Capture", "No live frame available yet.")
            return
        self.capture_in_progress = True
        self.btn_capture.configure(state=tk.DISABLED,
                                    text="⏳  Inspecting…")
        threading.Thread(target=self._capture_thread, args=(frame.copy(),),
                         daemon=True).start()

    def _capture_thread(self, frame) -> None:
        try:
            annotated, verdict, counts = infer_frame(
                model=self.model,
                frame=frame,
                imgsz=int(self.cfg.get("imgsz", 960)),
                conf=float(self.cfg.get("conf", 0.25)),
                device=str(self.cfg.get("device", "cpu")),
            )
            img_path = self._save_capture(annotated, verdict)
            self.root.after(0, self._on_capture_done,
                            annotated, verdict, counts, img_path)
        except Exception as exc:
            self.root.after(0, self._on_capture_error, str(exc))

    def _on_capture_done(self, annotated, verdict: str, counts: dict,
                          img_path: Path) -> None:
        self.last_result_img = annotated
        self.last_verdict    = verdict
        self.last_counts     = counts
        self.capture_count  += 1

        self._set_verdict_ui(verdict, counts)

        photo = self._frame_to_photo(annotated, self.result_lbl)
        if photo:
            self.photo_result = photo
            self.result_lbl.configure(image=photo)

        # Log entry
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = dict(ts=ts, verdict=verdict,
                   circle=counts.get(0, 0),
                   toward=counts.get(1, 0),
                   away=counts.get(2, 0),
                   img=str(img_path))
        self.history.insert(0, row)
        tag = verdict if verdict in ("OK", "NOT_OKAY") else "UNKNOWN"
        self.tree.insert("", 0,
                         values=(ts, verdict, counts.get(0,0),
                                 counts.get(1,0), counts.get(2,0),
                                 str(img_path)),
                         tags=(tag,))

        self._set_status(
            f"Capture #{self.capture_count}  →  {verdict}  "
            f"[circle={counts.get(0,0)} toward={counts.get(1,0)} away={counts.get(2,0)}]  "
            f"saved: {img_path.name}"
        )

        # Flash verdict background
        flash_colour = {"OK": "#1a4731", "NOT_OKAY": "#4b1a1a"}.get(verdict, "#2a2a3e")
        self.result_lbl.master.configure(bg=flash_colour)
        self.root.after(800, lambda: self.result_lbl.master.configure(bg="#2a2a3e"))

        self.capture_in_progress = False
        if self.running:
            self.btn_capture.configure(state=tk.NORMAL,
                                        text="📷   CAPTURE & INSPECT")

    def _on_capture_error(self, msg: str) -> None:
        self.capture_in_progress = False
        self.btn_capture.configure(state=tk.NORMAL,
                                    text="📷   CAPTURE & INSPECT")
        self._set_status(f"Capture error: {msg}")
        messagebox.showerror("Capture Error", msg)

    # ── Save captured frame ───────────────────────────────────────────────
    def _save_capture(self, frame, verdict: str) -> Path:
        save_dir = Path(str(self.cfg.get("save_dir", "results/captures")))
        if not save_dir.is_absolute():
            save_dir = BASE / save_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        out = save_dir / f"{stamp}_{verdict}.jpg"
        cv2.imwrite(str(out), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        log_path = save_dir / "capture_log.csv"
        ensure_log(log_path)
        append_log(log_path, verdict, self.last_counts, out)
        return out

    # ── Tkinter photo helper ──────────────────────────────────────────────
    def _frame_to_photo(self, frame, widget: tk.Label) -> ImageTk.PhotoImage | None:
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tw = max(widget.winfo_width(),  320)
            th = max(widget.winfo_height(), 240)
            h, w = rgb.shape[:2]
            scale = min(tw / w, th / h)
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
            return ImageTk.PhotoImage(Image.fromarray(resized))
        except Exception:
            return None

    # ── Close ─────────────────────────────────────────────────────────────
    def _on_close(self) -> None:
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    multiprocessing.freeze_support()
    root = tk.Tk()
    app = HikvisionInspector(root)   # noqa: F841
    root.mainloop()


if __name__ == "__main__":
    main()
