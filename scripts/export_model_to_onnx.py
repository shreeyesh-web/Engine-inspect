from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PT_MODEL = ROOT / "models" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a YOLO PyTorch model to ONNX.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_PT_MODEL,
        help="PyTorch model to export",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Export image size",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=12,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Export with dynamic input shapes",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device used during export",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = args.weights.resolve()
    if not weights.exists():
        raise SystemExit(f"PyTorch model not found: {weights}")

    model = YOLO(str(weights))
    exported = Path(
        model.export(
            format="onnx",
            imgsz=args.imgsz,
            opset=args.opset,
            simplify=True,
            dynamic=args.dynamic,
            device=args.device,
        )
    ).resolve()
    if not exported.exists():
        raise SystemExit(f"Expected ONNX model not found after export: {exported}")

    print(f"weights: {weights}")
    print(f"onnx   : {exported}")


if __name__ == "__main__":
    main()
