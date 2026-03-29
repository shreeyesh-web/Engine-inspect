#!/usr/bin/env python3
"""Benchmark a notch detector with the strict image-level verdict rule."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ultralytics import YOLO

from build_dataset import collect_samples
from verdict_rules import CLASS_NAMES, named_counts, strict_verdict_from_counts


@dataclass(frozen=True)
class EvalItem:
    image_path: Path
    gt_label: str
    eval_set: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a notch detector")
    parser.add_argument("--weights", type=Path, required=True, help="Path to model weights")
    parser.add_argument("--real-data-dir", type=Path, default=Path("data"))
    parser.add_argument("--skip-real-data", action="store_true", help="Skip the labeled data/ benchmark")
    parser.add_argument(
        "--eval-dir",
        action="append",
        default=[],
        metavar="LABEL=DIR",
        help="Benchmark all images in DIR with assumed image-level ground truth LABEL (OK or NOT_OKAY)",
    )
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="Optional image limit per evaluation set")
    parser.add_argument("--name", type=str, default="benchmark")
    parser.add_argument("--out-dir", type=Path, default=Path("results/benchmarks"))
    return parser.parse_args()


def list_images(path: Path) -> list[Path]:
    exts = ("*.bmp", "*.png", "*.jpg", "*.jpeg")
    files: list[Path] = []
    for ext in exts:
        files.extend(path.glob(ext))
    return sorted(files)


def parse_eval_dir(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Expected LABEL=DIR, got: {spec}")
    raw_label, raw_dir = spec.split("=", 1)
    label = raw_label.strip().upper()
    if label not in {"OK", "NOT_OKAY"}:
        raise ValueError(f"Unsupported label '{raw_label}'. Use OK or NOT_OKAY.")
    path = Path(raw_dir.strip())
    if not path.exists():
        raise FileNotFoundError(f"Evaluation directory not found: {path}")
    return label, path


def collect_real_data_items(data_dir: Path) -> list[EvalItem]:
    ok_samples = collect_samples(
        image_dir=data_dir / "OK",
        polygon_json=data_dir / "OK" / "labels_okay-labales_2026-02-19-07-29-49.json",
        circle_dir=data_dir / "OK" / "circle",
        source="ok",
        fallback_notch_cls=1,
    )
    not_ok_samples = collect_samples(
        image_dir=data_dir / "NOT_OKAY",
        polygon_json=data_dir / "NOT_OKAY" / "not_okay.json",
        circle_dir=data_dir / "NOT_OKAY" / "not_okay_circle",
        source="not_ok",
        fallback_notch_cls=2,
    )

    items = [
        EvalItem(image_path=sample.image_path, gt_label="OK", eval_set="real_labeled")
        for sample in ok_samples
    ]
    items.extend(
        EvalItem(image_path=sample.image_path, gt_label="NOT_OKAY", eval_set="real_labeled")
        for sample in not_ok_samples
    )
    return items


def build_eval_sets(args: argparse.Namespace) -> dict[str, list[EvalItem]]:
    eval_sets: dict[str, list[EvalItem]] = {}

    if not args.skip_real_data:
        eval_sets["real_labeled"] = collect_real_data_items(args.real_data_dir)

    for spec in args.eval_dir:
        label, directory = parse_eval_dir(spec)
        set_name = f"{label.lower()}::{directory.name}"
        files = list_images(directory)
        eval_sets[set_name] = [
            EvalItem(image_path=path, gt_label=label, eval_set=set_name)
            for path in files
        ]

    if args.limit > 0:
        eval_sets = {
            set_name: items[: args.limit]
            for set_name, items in eval_sets.items()
        }

    if not eval_sets:
        raise RuntimeError("No evaluation sets found. Add --eval-dir or use the default real data benchmark.")

    return eval_sets


def run_predictions(
    model: YOLO,
    items: Iterable[EvalItem],
    imgsz: int,
    conf: float,
    device: str,
    batch: int,
) -> list[dict]:
    rows: list[dict] = []
    items = list(items)
    if not items:
        return rows

    for start in range(0, len(items), batch):
        chunk = items[start : start + batch]
        predictions = model.predict(
            source=[str(item.image_path) for item in chunk],
            imgsz=imgsz,
            conf=conf,
            device=device,
            verbose=False,
        )

        for item, pred in zip(chunk, predictions):
            counts = {cls_id: 0 for cls_id in CLASS_NAMES}
            if pred.boxes is not None:
                for cls in pred.boxes.cls:
                    counts[int(cls.item())] += 1

            verdict = strict_verdict_from_counts(counts)
            rows.append(
                {
                    "image_name": item.image_path.name,
                    "image_path": str(item.image_path),
                    "eval_set": item.eval_set,
                    "ground_truth": item.gt_label,
                    "prediction": verdict.verdict,
                    "reasons": list(verdict.reasons),
                    "counts": named_counts(counts),
                }
            )

    return rows


def summarize_rows(rows: list[dict]) -> dict:
    confusion = Counter((row["ground_truth"], row["prediction"]) for row in rows)
    total = len(rows)
    correct = sum(1 for row in rows if row["ground_truth"] == row["prediction"])

    tp_ok = confusion[("OK", "OK")]
    fn_ok = confusion[("OK", "NOT_OKAY")]
    fp_ok = confusion[("NOT_OKAY", "OK")]
    tn_ok = confusion[("NOT_OKAY", "NOT_OKAY")]

    precision_ok = tp_ok / (tp_ok + fp_ok) if (tp_ok + fp_ok) else 0.0
    recall_ok = tp_ok / (tp_ok + fn_ok) if (tp_ok + fn_ok) else 0.0
    specificity_ok = tn_ok / (tn_ok + fp_ok) if (tn_ok + fp_ok) else 0.0

    by_reason = Counter()
    for row in rows:
        if row["prediction"] == "NOT_OKAY":
            by_reason.update(row["reasons"])

    return {
        "images": total,
        "accuracy": correct / total if total else 0.0,
        "ok_precision": precision_ok,
        "ok_recall": recall_ok,
        "ok_specificity": specificity_ok,
        "confusion": {
            "OK->OK": confusion[("OK", "OK")],
            "OK->NOT_OKAY": confusion[("OK", "NOT_OKAY")],
            "NOT_OKAY->OK": confusion[("NOT_OKAY", "OK")],
            "NOT_OKAY->NOT_OKAY": confusion[("NOT_OKAY", "NOT_OKAY")],
        },
        "failure_reasons": dict(by_reason),
    }


def print_summary(set_name: str, summary: dict) -> None:
    print("=" * 72)
    print(set_name)
    print("=" * 72)
    print(f"images          : {summary['images']}")
    print(f"accuracy        : {summary['accuracy']:.4f}")
    print(f"ok precision    : {summary['ok_precision']:.4f}")
    print(f"ok recall       : {summary['ok_recall']:.4f}")
    print(f"ok specificity  : {summary['ok_specificity']:.4f}")
    print(f"confusion       : {summary['confusion']}")
    print(f"failure reasons : {summary['failure_reasons']}")


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "eval_set",
        "image_name",
        "image_path",
        "ground_truth",
        "prediction",
        "circle",
        "notch_toward_hole",
        "notch_not_toward_hole",
        "reasons",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "eval_set": row["eval_set"],
                    "image_name": row["image_name"],
                    "image_path": row["image_path"],
                    "ground_truth": row["ground_truth"],
                    "prediction": row["prediction"],
                    "circle": row["counts"]["circle"],
                    "notch_toward_hole": row["counts"]["notch_toward_hole"],
                    "notch_not_toward_hole": row["counts"]["notch_not_toward_hole"],
                    "reasons": " | ".join(row["reasons"]),
                }
            )


def main() -> None:
    args = parse_args()
    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    eval_sets = build_eval_sets(args)
    model = YOLO(str(args.weights))

    all_rows: list[dict] = []
    summaries: dict[str, dict] = {}

    for set_name, items in eval_sets.items():
        print(f"running         : {set_name} ({len(items)} images)")
        rows = run_predictions(
            model=model,
            items=items,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            batch=args.batch,
        )
        summary = summarize_rows(rows)
        summaries[set_name] = summary
        print_summary(set_name, summary)
        all_rows.extend(rows)

    overall = summarize_rows(all_rows)
    summaries["overall"] = overall
    print_summary("overall", overall)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"{args.name}.json"
    csv_path = args.out_dir / f"{args.name}.csv"

    payload = {
        "weights": str(args.weights.resolve()),
        "config": {
            "imgsz": args.imgsz,
            "conf": args.conf,
            "device": args.device,
            "batch": args.batch,
            "limit": args.limit,
        },
        "summaries": summaries,
        "rows": all_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(csv_path, all_rows)

    print(f"saved json      : {json_path}")
    print(f"saved csv       : {csv_path}")


if __name__ == "__main__":
    main()
