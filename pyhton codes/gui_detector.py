#!/usr/bin/env python3
"""Desktop GUI app for live notch orientation detection."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import messagebox, ttk
from ultralytics import YOLO

from verdict_rules import strict_verdict_from_counts


CLASS_NAMES = {
    0: "circle",
    1: "notch_toward_hole",
    2: "notch_not_toward_hole",
}

COLORS = {
    0: (0, 255, 0),
    1: (0, 220, 255),
    2: (0, 0, 255),
}

VERDICT_COLORS = {
    "OK": "#10b981",
    "NOT_OKAY": "#ef4444",
    "UNKNOWN": "#9ca3af",
}

DEFAULT_CONFIG: dict[str, Any] = {
    "source": "0",
    "weights": "models/best.pt",
    "imgsz": 960,
    "conf": 0.25,
    "device": "cpu",
    "save_dir": "results/live",
    "save_mode": "not_okay",
    "cooldown_sec": 2.0,
    "rtsp_tcp": True,
    "reconnect_after": 30,
    "auto_start": True,
}


def runtime_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GUI notch detector")
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def resolve_source(source: str) -> int | str:
    source = source.strip()
    if source.lstrip("-").isdigit():
        return int(source)
    return source


def resolve_existing_path(path: Path, bases: list[Path]) -> Path | None:
    if path.is_absolute():
        return path if path.exists() else None
    for base in bases:
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return None


def open_capture(source: int | str, rtsp_tcp: bool) -> cv2.VideoCapture:
    if isinstance(source, str) and source.lower().startswith("rtsp://") and rtsp_tcp:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

    if isinstance(source, str):
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def ensure_event_log(path: Path) -> None:
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "timestamp",
                "frame_id",
                "verdict",
                "circle",
                "notch_toward_hole",
                "notch_not_toward_hole",
                "reason",
                "image_path",
            ]
        )


def append_event(
    path: Path,
    frame_id: int,
    verdict: str,
    counts: dict[int, int],
    reason: str,
    image_path: Path,
) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                datetime.now().isoformat(timespec="seconds"),
                frame_id,
                verdict,
                counts.get(0, 0),
                counts.get(1, 0),
                counts.get(2, 0),
                reason,
                str(image_path),
            ]
        )


def save_frame(save_dir: Path, frame, frame_id: int, verdict: str) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    out_path = save_dir / f"{stamp}_f{frame_id:06d}_{verdict}.jpg"
    cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return out_path


def draw_detection(image, box, cls_id: int, conf: float) -> None:
    x1, y1, x2, y2 = [int(v) for v in box]
    color = COLORS.get(cls_id, (255, 255, 255))
    label = f"{CLASS_NAMES.get(cls_id, str(cls_id))} {conf:.2f}"
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
    cv2.putText(image, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def infer_frame(model: YOLO, frame, imgsz: int, conf: float, device: str):
    pred = model.predict(
        source=frame,
        imgsz=imgsz,
        conf=conf,
        device=device,
        verbose=False,
    )[0]

    annotated = frame.copy()
    counts = {0: 0, 1: 0, 2: 0}

    if pred.boxes is not None:
        for box, cls, score in zip(pred.boxes.xyxy, pred.boxes.cls, pred.boxes.conf):
            cls_id = int(cls.item())
            conf_v = float(score.item())
            xyxy = [float(v) for v in box.tolist()]
            counts[cls_id] = counts.get(cls_id, 0) + 1
            draw_detection(annotated, xyxy, cls_id, conf_v)

    verdict = strict_verdict_from_counts(counts).verdict

    return annotated, verdict, counts


class NotchDetectorGUI:
    def __init__(self, root: tk.Tk, config_path: Path | None) -> None:
        self.root = root
        self.root.title("Harmony - JD Block Notch Detection")
        self.root.geometry("1280x820")
        self.root.minsize(980, 680)

        self.base_dir = runtime_base_dir()
        self.config_path = self._resolve_config_path(config_path)
        self.config = self._load_or_create_config()

        self.model: YOLO | None = None
        self.model_path: Path | None = None
        self.cap: cv2.VideoCapture | None = None
        self.running = False
        self.failed_reads = 0
        self.frame_id = 0
        self.last_saved_at = 0.0
        self.start_time = time.time()
        self.last_frame = None
        self.last_annotated = None
        self.last_verdict = "UNKNOWN"
        self.last_counts = {0: 0, 1: 0, 2: 0}
        self.photo_ref: ImageTk.PhotoImage | None = None

        self._setup_styles()
        self._build_ui()
        self._load_config_to_ui()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        if bool(self.config.get("auto_start", True)):
            self.root.after(500, self.start)

    def _setup_styles(self) -> None:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Top.TFrame", padding=8)
        style.configure("Card.TFrame", padding=8)

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, style="Top.TFrame")
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Source").grid(row=0, column=0, sticky="w", padx=(0, 4))
        self.source_var = tk.StringVar()
        self.source_entry = ttk.Combobox(top, textvariable=self.source_var, width=40)
        self.source_entry.grid(row=0, column=1, sticky="we", padx=(0, 4))
        
        self.detect_btn = ttk.Button(top, text="Detect USB", command=self.detect_usb_cameras)
        self.detect_btn.grid(row=0, column=2, padx=(0, 8))

        ttk.Label(top, text="Conf").grid(row=0, column=3, sticky="w", padx=(0, 4))
        self.conf_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.conf_var, width=8).grid(row=0, column=4, sticky="w", padx=(0, 8))

        ttk.Label(top, text="Imgsz").grid(row=0, column=5, sticky="w", padx=(0, 4))
        self.imgsz_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.imgsz_var, width=8).grid(row=0, column=6, sticky="w", padx=(0, 8))

        self.rtsp_tcp_var = tk.BooleanVar()
        ttk.Checkbutton(top, text="RTSP TCP", variable=self.rtsp_tcp_var).grid(row=0, column=7, padx=(0, 8))

        self.start_btn = ttk.Button(top, text="Start", command=self.start)
        self.start_btn.grid(row=0, column=8, padx=(0, 6))

        self.stop_btn = ttk.Button(top, text="Stop", command=self.stop, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=9, padx=(0, 6))

        ttk.Button(top, text="Snapshot", command=self.snapshot).grid(row=0, column=10, padx=(0, 6))
        ttk.Button(top, text="Save Config", command=self.save_config).grid(row=0, column=11, padx=(0, 6))
        ttk.Button(top, text="Open Results", command=self.open_results_dir).grid(row=0, column=12)
        top.columnconfigure(1, weight=1)

        status_card = ttk.Frame(self.root, style="Card.TFrame")
        status_card.pack(side=tk.TOP, fill=tk.X)

        self.verdict_var = tk.StringVar(value="VERDICT: UNKNOWN")
        self.verdict_label = ttk.Label(
            status_card,
            textvariable=self.verdict_var,
            font=("Segoe UI", 18, "bold"),
            foreground=VERDICT_COLORS["UNKNOWN"],
        )
        self.verdict_label.pack(side=tk.LEFT, padx=(4, 18))

        self.counts_var = tk.StringVar(value="circle=0 toward=0 away=0 fps=0.0")
        ttk.Label(
            status_card,
            textvariable=self.counts_var,
            font=("Consolas", 13, "normal"),
        ).pack(side=tk.LEFT, padx=(4, 18))

        self.status_var = tk.StringVar(value=f"Config: {self.config_path}")
        ttk.Label(
            status_card,
            textvariable=self.status_var,
            font=("Segoe UI", 10, "normal"),
        ).pack(side=tk.LEFT, padx=(4, 0))

        video_card = ttk.Frame(self.root, style="Card.TFrame")
        video_card.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.video_label = ttk.Label(video_card, anchor="center")
        self.video_label.pack(fill=tk.BOTH, expand=True)

    def detect_usb_cameras(self) -> None:
        self.detect_btn.config(state=tk.DISABLED, text="Detecting...")
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
            self.source_entry['values'] = available
            if self.source_var.get() not in available:
                self.source_var.set(available[0])
            messagebox.showinfo("Cameras Detected", f"Found USB cameras at indices: {', '.join(available)}")
        else:
            self.source_entry['values'] = []
            messagebox.showwarning("No Cameras", "No USB cameras detected.")
            
        self.detect_btn.config(state=tk.NORMAL, text="Detect USB")

    def _resolve_config_path(self, provided: Path | None) -> Path:
        if provided is not None:
            return provided.resolve()
        return (self.base_dir / "client_config.json").resolve()

    def _load_or_create_config(self) -> dict[str, Any]:
        if self.config_path.exists():
            payload = json.loads(self.config_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                cfg = dict(DEFAULT_CONFIG)
                cfg.update(payload)
                return cfg

        example_path = (self.base_dir / "client_config.example.json").resolve()
        if example_path.exists():
            payload = json.loads(example_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                cfg = dict(DEFAULT_CONFIG)
                cfg.update(payload)
                self.config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
                return cfg

        self.config_path.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")
        return dict(DEFAULT_CONFIG)

    def _load_config_to_ui(self) -> None:
        self.source_var.set(str(self.config.get("source", "0")))
        self.conf_var.set(str(self.config.get("conf", 0.25)))
        self.imgsz_var.set(str(self.config.get("imgsz", 960)))
        self.rtsp_tcp_var.set(bool(self.config.get("rtsp_tcp", True)))

    def _parse_ui_values(self) -> tuple[str, float, int, bool]:
        source = self.source_var.get().strip()
        if not source:
            raise ValueError("Source cannot be empty.")

        conf = float(self.conf_var.get().strip())
        if conf <= 0 or conf > 1:
            raise ValueError("Conf must be between 0 and 1.")

        imgsz = int(self.imgsz_var.get().strip())
        if imgsz < 64:
            raise ValueError("Imgsz must be >= 64.")

        rtsp_tcp = bool(self.rtsp_tcp_var.get())
        return source, conf, imgsz, rtsp_tcp

    def _resolved_save_dir(self) -> Path:
        save_dir = Path(str(self.config.get("save_dir", "results/live")))
        if save_dir.is_absolute():
            return save_dir
        return (self.base_dir / save_dir).resolve()

    def _resolved_weights(self) -> Path:
        value = Path(str(self.config.get("weights", "models/best.pt")))
        bases = [self.config_path.parent, self.base_dir, Path.cwd()]
        resolved = resolve_existing_path(value, bases)
        if resolved is None:
            raise FileNotFoundError(f"Weights not found: {value}")
        return resolved

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _set_verdict_ui(self, verdict: str, counts: dict[int, int], fps: float) -> None:
        self.verdict_var.set(f"VERDICT: {verdict}")
        self.verdict_label.configure(foreground=VERDICT_COLORS.get(verdict, VERDICT_COLORS["UNKNOWN"]))
        self.counts_var.set(
            "circle={0} toward={1} away={2} fps={3:.1f}".format(
                counts.get(0, 0),
                counts.get(1, 0),
                counts.get(2, 0),
                fps,
            )
        )

    def _draw_video(self, frame) -> None:
        if frame is None:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        target_w = max(self.video_label.winfo_width(), 320)
        target_h = max(self.video_label.winfo_height(), 240)

        h, w = rgb.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        image = Image.fromarray(resized)
        photo = ImageTk.PhotoImage(image=image)
        self.photo_ref = photo
        self.video_label.configure(image=photo)

    def _maybe_auto_save(self, frame, verdict: str, counts: dict[int, int]) -> None:
        save_mode = str(self.config.get("save_mode", "not_okay")).lower()
        cooldown = float(self.config.get("cooldown_sec", 2.0))
        now = time.monotonic()
        should_save = False
        reason = ""

        if save_mode == "all":
            if now - self.last_saved_at >= max(cooldown, 0.0):
                should_save = True
                reason = "periodic"
        elif save_mode == "not_okay" and verdict == "NOT_OKAY":
            if now - self.last_saved_at >= max(cooldown, 0.0):
                should_save = True
                reason = "not_okay"

        if not should_save:
            return

        save_dir = self._resolved_save_dir()
        out_path = save_frame(save_dir, frame, self.frame_id, verdict)
        events_csv = save_dir / "events.csv"
        ensure_event_log(events_csv)
        append_event(events_csv, self.frame_id, verdict, counts, reason, out_path)
        self.last_saved_at = now

    def save_config(self) -> None:
        try:
            source, conf, imgsz, rtsp_tcp = self._parse_ui_values()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Invalid Settings", str(exc))
            return

        self.config["source"] = source
        self.config["conf"] = conf
        self.config["imgsz"] = imgsz
        self.config["rtsp_tcp"] = rtsp_tcp
        self.config_path.write_text(json.dumps(self.config, indent=2), encoding="utf-8")
        self._set_status(f"Saved config: {self.config_path}")

    def start(self) -> None:
        if self.running:
            return

        try:
            source, conf, imgsz, rtsp_tcp = self._parse_ui_values()
            self.config["source"] = source
            self.config["conf"] = conf
            self.config["imgsz"] = imgsz
            self.config["rtsp_tcp"] = rtsp_tcp

            weights = self._resolved_weights()
            if self.model is None or self.model_path != weights:
                self._set_status(f"Loading model: {weights.name}")
                self.root.update_idletasks()
                self.model = YOLO(str(weights))
                self.model_path = weights

            source_obj = resolve_source(source)
            self.cap = open_capture(source_obj, rtsp_tcp)
            if not self.cap.isOpened():
                raise RuntimeError(
                    f"Could not open source: {source}. Check camera IP/RTSP credentials."
                )

            self.running = True
            self.failed_reads = 0
            self.frame_id = 0
            self.start_time = time.time()
            self.start_btn.configure(state=tk.DISABLED)
            self.stop_btn.configure(state=tk.NORMAL)
            self._set_status(f"Running source: {source}")
            self._loop()
        except Exception as exc:  # noqa: BLE001
            self._set_status(str(exc))
            messagebox.showerror("Start Failed", str(exc))
            self.stop()

    def stop(self) -> None:
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self._set_status("Stopped")

    def snapshot(self) -> None:
        if self.last_annotated is None:
            messagebox.showinfo("Snapshot", "No frame to save yet.")
            return

        save_dir = self._resolved_save_dir()
        out_path = save_frame(save_dir, self.last_annotated, self.frame_id, self.last_verdict)
        events_csv = save_dir / "events.csv"
        ensure_event_log(events_csv)
        append_event(events_csv, self.frame_id, self.last_verdict, self.last_counts, "manual", out_path)
        self._set_status(f"Saved snapshot: {out_path.name}")

    def open_results_dir(self) -> None:
        save_dir = self._resolved_save_dir()
        save_dir.mkdir(parents=True, exist_ok=True)
        import platform
        import subprocess
        if platform.system() == "Windows":
            os.startfile(save_dir)  # type: ignore[attr-defined]
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", str(save_dir)])
        else:
            subprocess.Popen(["xdg-open", str(save_dir)])

    def _loop(self) -> None:
        if not self.running or self.cap is None or self.model is None:
            return

        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.failed_reads += 1
            reconnect_after = int(self.config.get("reconnect_after", 30))
            if self.failed_reads >= max(1, reconnect_after):
                self.cap.release()
                time.sleep(0.2)
                self.cap = open_capture(resolve_source(str(self.config.get("source", "0"))), bool(self.config.get("rtsp_tcp", True)))
                self.failed_reads = 0
            self._set_status("Waiting for frames...")
            self.root.after(40, self._loop)
            return

        self.failed_reads = 0
        self.frame_id += 1

        annotated, verdict, counts = infer_frame(
            model=self.model,
            frame=frame,
            imgsz=int(self.config.get("imgsz", 960)),
            conf=float(self.config.get("conf", 0.25)),
            device=str(self.config.get("device", "cpu")),
        )

        elapsed = max(time.time() - self.start_time, 1e-6)
        fps = self.frame_id / elapsed
        self.last_frame = frame
        self.last_annotated = annotated
        self.last_verdict = verdict
        self.last_counts = counts

        self._set_verdict_ui(verdict, counts, fps)
        self._maybe_auto_save(annotated, verdict, counts)
        self._draw_video(annotated)
        self.root.after(1, self._loop)

    def on_close(self) -> None:
        self.stop()
        self.root.destroy()


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    NotchDetectorGUI(root, args.config)
    root.mainloop()


if __name__ == "__main__":
    main()
