#!/usr/bin/env python3
"""Run inference and produce per-image OK/NOT_OKAY verdict."""

from __future__ import annotations

import argparse
import json
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for notch orientation")
    parser.add_argument("--source", type=Path, required=True, help="Image path or directory")
    parser.add_argument("--weights", type=Path, default=Path("runs/detect/runs/notch_face/weights/best.pt"))
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def list_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []

    exts = ("*.bmp", "*.png", "*.jpg", "*.jpeg")
    files: list[Path] = []
    for ext in exts:
        files.extend(path.glob(ext))
    return sorted(files)


def draw_detection(image, box, cls_id: int, conf: float) -> None:
    x1, y1, x2, y2 = [int(v) for v in box]
    color = COLORS.get(cls_id, (255, 255, 255))
    label = f"{CLASS_NAMES.get(cls_id, str(cls_id))} {conf:.2f}"

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
    cv2.putText(image, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def main() -> None:
    args = parse_args()
    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    image_paths = list_images(args.source)
    if not image_paths:
        raise RuntimeError(f"No images found in source: {args.source}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))
    summary: dict[str, dict] = {}

    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        pred = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False,
        )[0]

        detections = []
        counts = {0: 0, 1: 0, 2: 0}

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
                draw_detection(image, xyxy, cls_id, conf_v)

        verdict = strict_verdict_from_counts(counts).verdict

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

        out_image = args.out_dir / f"{image_path.stem}_pred.jpg"
        cv2.imwrite(str(out_image), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        summary[image_path.name] = {
            "verdict": verdict,
            "counts": {
                "circle": counts.get(0, 0),
                "notch_toward_hole": counts.get(1, 0),
                "notch_not_toward_hole": counts.get(2, 0),
            },
            "detections": detections,
            "output_image": str(out_image),
        }

        print(
            f"{image_path.name}: circles={counts.get(0, 0)} "
            f"toward={counts.get(1, 0)} away={counts.get(2, 0)} => {verdict}"
        )

    out_json = args.out_dir / "predictions.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved predictions: {out_json}")


if __name__ == "__main__":
    main()
