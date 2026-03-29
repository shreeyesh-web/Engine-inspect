#!/usr/bin/env python3
"""Train YOLO model for notch-facing-hole detection."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train notch orientation detector")
    parser.add_argument("--data", type=Path, default=Path("dataset/data.yaml"), help="Path to YOLO data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model checkpoint")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--project", type=Path, default=Path("runs"))
    parser.add_argument("--name", type=str, default="notch_face")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="0" if torch.cuda.is_available() else "cpu",
        help="Device id or 'cpu'",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {args.data}")

    model = YOLO(args.model)

    print("=" * 70)
    print("Training Run")
    print("=" * 70)
    print(f"data     : {args.data}")
    print(f"model    : {args.model}")
    print(f"epochs   : {args.epochs}")
    print(f"imgsz    : {args.imgsz}")
    print(f"batch    : {args.batch}")
    print(f"device   : {args.device}")

    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        device=args.device,
        seed=args.seed,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        cos_lr=True,
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.35,
        degrees=8.0,
        translate=0.08,
        scale=0.25,
        shear=2.0,
        fliplr=0.5,
        flipud=0.5,
        mosaic=0.8,
        mixup=0.1,
    )

    run_dir = Path(results.save_dir)
    best_path = run_dir / "weights" / "best.pt"
    print("=" * 70)
    print("Training Complete")
    print("=" * 70)
    print(f"run dir  : {run_dir}")
    print(f"best pt  : {best_path}")


if __name__ == "__main__":
    main()
