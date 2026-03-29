#!/usr/bin/env python3
"""Live camera detection for notch orientation."""

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
    "OK": (0, 220, 0),
    "NOT_OKAY": (0, 0, 255),
    "UNKNOWN": (200, 200, 200),
}

SAVE_MODES = {"none", "not_okay", "all"}


def runtime_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    return payload


def read_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
    return default


def parse_args() -> argparse.Namespace:
    preload = argparse.ArgumentParser(add_help=False)
    preload.add_argument("--config", type=Path, default=None)
    preload_args, _ = preload.parse_known_args()
    cfg = load_config(preload_args.config)

    source_default = str(cfg.get("source", "0"))
    weights_default = cfg.get("weights")
    imgsz_default = int(cfg.get("imgsz", 960))
    conf_default = float(cfg.get("conf", 0.25))
    device_default = str(cfg.get("device", "cpu"))
    save_dir_default = Path(str(cfg.get("save_dir", "results/live")))
    save_mode_default = str(cfg.get("save_mode", "not_okay")).lower()
    if save_mode_default not in SAVE_MODES:
        save_mode_default = "not_okay"
    cooldown_default = float(cfg.get("cooldown_sec", 2.0))
    max_frames_default = int(cfg.get("max_frames", 0))
    show_default = read_bool(cfg.get("show", True), True)
    single_shot_default = read_bool(cfg.get("single_shot", False), False)
    rtsp_tcp_default = read_bool(cfg.get("rtsp_tcp", True), True)
    reconnect_after_default = int(cfg.get("reconnect_after", 30))

    parser = argparse.ArgumentParser(
        description="Live notch orientation detection from camera stream"
    )
    parser.add_argument("--config", type=Path, default=preload_args.config)
    parser.add_argument(
        "--source",
        type=str,
        default=source_default,
        help="Camera index (0,1,...) or stream URL (rtsp/http/file path)",
    )
    parser.add_argument("--weights", type=Path, default=Path(weights_default) if weights_default else None)
    parser.add_argument("--imgsz", type=int, default=imgsz_default)
    parser.add_argument("--conf", type=float, default=conf_default)
    parser.add_argument("--device", type=str, default=device_default)
    parser.add_argument("--save-dir", type=Path, default=save_dir_default)
    parser.add_argument(
        "--save-mode",
        type=str,
        choices=["none", "not_okay", "all"],
        default=save_mode_default,
    )
    parser.add_argument(
        "--cooldown-sec",
        type=float,
        default=cooldown_default,
        help="Minimum seconds between saved images",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=max_frames_default,
        help="0 means run until manual stop",
    )
    parser.add_argument(
        "--single-shot",
        action="store_true",
        default=single_shot_default,
        help="Capture and process only one frame",
    )
    parser.add_argument(
        "--reconnect-after",
        type=int,
        default=reconnect_after_default,
        help="Reconnect camera after this many consecutive read failures",
    )

    display_group = parser.add_mutually_exclusive_group()
    display_group.add_argument("--show", dest="show", action="store_true", help="Display live window")
    display_group.add_argument("--no-show", dest="show", action="store_false", help="Disable live window")
    parser.set_defaults(show=show_default)

    transport_group = parser.add_mutually_exclusive_group()
    transport_group.add_argument("--rtsp-tcp", dest="rtsp_tcp", action="store_true")
    transport_group.add_argument("--rtsp-udp", dest="rtsp_tcp", action="store_false")
    parser.set_defaults(rtsp_tcp=rtsp_tcp_default)

    return parser.parse_args()


def resolve_source(source: str) -> int | str:
    source = source.strip()
    if source.lstrip("-").isdigit():
        return int(source)
    return source


def resolve_existing_path(path: Path, extra_bases: list[Path] | None = None) -> Path | None:
    candidates: list[Path] = []
    if extra_bases:
        candidates.extend(extra_bases)
    candidates.extend([runtime_base_dir(), Path.cwd()])

    if path.is_absolute():
        return path if path.exists() else None

    for base in candidates:
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return None


def find_latest_weights() -> Path | None:
    candidates: list[Path] = []
    checked_roots: set[Path] = set()
    for root in [runtime_base_dir(), Path.cwd()]:
        root = root.resolve()
        if root in checked_roots:
            continue
        checked_roots.add(root)
        candidates.extend(root.glob("runs/detect/runs/*/weights/best.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_weights(args: argparse.Namespace) -> Path:
    config_dir = args.config.resolve().parent if args.config else None
    extra_bases = [config_dir] if config_dir else []

    if args.weights is not None:
        weights = resolve_existing_path(args.weights, extra_bases=extra_bases)
        if weights is None:
            raise FileNotFoundError(f"Weights not found: {args.weights}")
        return weights

    default_model = resolve_existing_path(Path("models/best.pt"), extra_bases=extra_bases)
    if default_model is not None:
        return default_model

    latest = find_latest_weights()
    if latest is None:
        raise FileNotFoundError(
            "No weights found. Pass --weights path/to/best.pt or place model at models/best.pt"
        )
    return latest


def resolve_output_dir(path: Path, config_path: Path | None) -> Path:
    if path.is_absolute():
        return path
    if config_path is not None:
        return (config_path.resolve().parent / path).resolve()
    return (runtime_base_dir() / path).resolve()


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


def draw_status(frame, verdict: str, counts: dict[int, int], fps: float) -> None:
    verdict_color = VERDICT_COLORS.get(verdict, (255, 255, 255))
    cv2.putText(
        frame,
        f"VERDICT: {verdict}",
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        verdict_color,
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        (
            "circle={0} toward={1} away={2} fps={3:.1f}".format(
                counts.get(0, 0),
                counts.get(1, 0),
                counts.get(2, 0),
                fps,
            )
        ),
        (24, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


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
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    out_path = save_dir / f"{stamp}_f{frame_id:06d}_{verdict}.jpg"
    cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return out_path


def main() -> None:
    args = parse_args()
    source = resolve_source(args.source)
    weights = resolve_weights(args)
    save_dir = resolve_output_dir(args.save_dir, args.config)
    save_dir.mkdir(parents=True, exist_ok=True)
    events_csv = save_dir / "events.csv"
    ensure_event_log(events_csv)

    model = YOLO(str(weights))
    cap = open_capture(source, args.rtsp_tcp)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video source: {args.source}. Check camera/IP/credentials."
        )

    print(f"source: {args.source}")
    print(f"weights: {weights}")
    print(f"save_dir: {save_dir}")
    print(f"save_mode: {args.save_mode}")
    if args.show:
        print("controls: press 'q' to quit, press 's' to save current frame")

    frame_id = 0
    failed_reads = 0
    start_at = time.time()
    last_saved_at = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                failed_reads += 1
                if failed_reads >= max(1, args.reconnect_after):
                    cap.release()
                    time.sleep(0.4)
                    cap = open_capture(source, args.rtsp_tcp)
                    failed_reads = 0
                time.sleep(0.03)
                continue

            failed_reads = 0
            frame_id += 1

            annotated, verdict, counts = infer_frame(
                model=model,
                frame=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
            )

            elapsed = max(time.time() - start_at, 1e-6)
            fps = frame_id / elapsed
            draw_status(annotated, verdict, counts, fps)

            save_reason = ""
            now = time.monotonic()
            if args.single_shot:
                save_reason = "single_shot"
            elif args.save_mode == "all":
                if now - last_saved_at >= max(args.cooldown_sec, 0.0):
                    save_reason = "periodic"
            elif args.save_mode == "not_okay" and verdict == "NOT_OKAY":
                if now - last_saved_at >= max(args.cooldown_sec, 0.0):
                    save_reason = "not_okay"

            if save_reason:
                out_path = save_frame(save_dir, annotated, frame_id, verdict)
                append_event(
                    path=events_csv,
                    frame_id=frame_id,
                    verdict=verdict,
                    counts=counts,
                    reason=save_reason,
                    image_path=out_path,
                )
                last_saved_at = now
                print(
                    (
                        f"[saved] frame={frame_id} verdict={verdict} "
                        f"circle={counts.get(0, 0)} toward={counts.get(1, 0)} "
                        f"away={counts.get(2, 0)} -> {out_path.name}"
                    )
                )

            if frame_id % 30 == 0 and not save_reason:
                print(
                    (
                        f"frame={frame_id} verdict={verdict} "
                        f"circle={counts.get(0, 0)} toward={counts.get(1, 0)} "
                        f"away={counts.get(2, 0)}"
                    )
                )

            if args.show:
                cv2.imshow("Notch Detector", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    out_path = save_frame(save_dir, annotated, frame_id, verdict)
                    append_event(
                        path=events_csv,
                        frame_id=frame_id,
                        verdict=verdict,
                        counts=counts,
                        reason="manual",
                        image_path=out_path,
                    )
                    print(f"[saved] manual snapshot -> {out_path.name}")

            if args.single_shot:
                break
            if args.max_frames > 0 and frame_id >= args.max_frames:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"done. processed_frames={frame_id}")
    print(f"event_log={events_csv}")


if __name__ == "__main__":
    main()
