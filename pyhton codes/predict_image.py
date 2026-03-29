#!/usr/bin/env python3
"""Predict notch orientation for a single image path."""

from __future__ import annotations

import argparse
from pathlib import Path

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


def find_latest_weights() -> Path | None:
    """Find newest best.pt under runs/detect/runs/*/weights/."""
    candidates = sorted(
        Path("runs").glob("detect/runs/*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single image prediction")
    parser.add_argument("image", type=Path, help="Input image path")
    parser.add_argument("--weights", type=Path, default=None, help="Model weights path")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--save", action="store_true", help="Save annotated output image")
    parser.add_argument("--out-dir", type=Path, default=Path("results/single"))
    return parser.parse_args()


def draw_box(image, box, cls_id: int, conf: float) -> None:
    x1, y1, x2, y2 = [int(v) for v in box]
    color = COLORS.get(cls_id, (255, 255, 255))
    label = f"{CLASS_NAMES.get(cls_id, str(cls_id))} {conf:.2f}"

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
    cv2.putText(image, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def main() -> None:
    args = parse_args()

    if not args.image.exists() or not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")

    weights = args.weights if args.weights is not None else find_latest_weights()
    if weights is None:
        raise FileNotFoundError(
            "No weights found. Train first or pass --weights path/to/best.pt"
        )
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    model = YOLO(str(weights))
    pred = model.predict(
        source=str(args.image),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        verbose=False,
    )[0]

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {args.image}")

    counts = {0: 0, 1: 0, 2: 0}
    detections: list[dict] = []

    if pred.boxes is not None:
        for box, cls, conf in zip(pred.boxes.xyxy, pred.boxes.cls, pred.boxes.conf):
            cls_id = int(cls.item())
            conf_v = float(conf.item())
            xyxy = [float(v) for v in box.tolist()]
            counts[cls_id] = counts.get(cls_id, 0) + 1
            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": CLASS_NAMES.get(cls_id, str(cls_id)),
                    "confidence": conf_v,
                    "xyxy": xyxy,
                }
            )
            draw_box(image, xyxy, cls_id, conf_v)

    verdict = strict_verdict_from_counts(counts).verdict

    print(f"image: {args.image}")
    print(f"weights: {weights}")
    print(
        f"counts -> circle={counts.get(0, 0)}, toward={counts.get(1, 0)}, "
        f"away={counts.get(2, 0)}"
    )
    print(f"verdict: {verdict}")

    if args.save:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.out_dir / f"{args.image.stem}_pred.jpg"
        cv2.putText(
            image,
            f"VERDICT: {verdict}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(out_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        print(f"saved: {out_path}")

    if detections:
        print("detections:")
        for det in detections:
            print(
                f"  - {det['class_name']} conf={det['confidence']:.3f} "
                f"box={det['xyxy']}"
            )


if __name__ == "__main__":
    main()
