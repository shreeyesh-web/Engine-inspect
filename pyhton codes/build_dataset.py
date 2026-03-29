#!/usr/bin/env python3
"""Build a clean YOLO dataset for notch orientation detection.

Uses only the current data layout:
- data/OK/*.bmp + data/OK/labels_okay-*.json + data/OK/circle/*.txt
- data/NOT_OKAY/*.bmp + data/NOT_OKAY/not_okay.json + data/NOT_OKAY/not_okay_circle/*.txt

Output classes:
0: circle
1: notch_toward_hole
2: notch_not_toward_hole
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


CLASS_NAMES = {
    0: "circle",
    1: "notch_toward_hole",
    2: "notch_not_toward_hole",
}


@dataclass
class Sample:
    source: str  # "ok" | "ok_extra" | "not_ok"
    image_path: Path
    labels: list[tuple[int, float, float, float, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build YOLO dataset from polygon + circle annotations.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Input data directory")
    parser.add_argument("--out-dir", type=Path, default=Path("dataset"), help="Output YOLO dataset directory")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio per class group")
    parser.add_argument(
        "--extra-ok-dir",
        action="append",
        default=[],
        type=Path,
        help="Optional directory of extra OK images that will use template labels derived from the real OK set",
    )
    parser.add_argument(
        "--ok-augment-factor",
        type=int,
        default=3,
        help="Synthetic samples generated per real OK train image",
    )
    parser.add_argument(
        "--not-ok-augment-factor",
        type=int,
        default=24,
        help="Synthetic samples generated per real NOT_OKAY train image",
    )
    parser.add_argument(
        "--neg-augment-factor",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_json(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def polygon_to_bbox_norm(points: list[tuple[float, float]], width: int, height: int) -> tuple[float, float, float, float] | None:
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1 = max(0.0, min(xs))
    y1 = max(0.0, min(ys))
    x2 = min(float(width), max(xs))
    y2 = min(float(height), max(ys))

    bw = x2 - x1
    bh = y2 - y1
    if bw <= 1.0 or bh <= 1.0:
        return None

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return cx / width, cy / height, bw / width, bh / height


def parse_region_label(raw_label: str, fallback_cls: int) -> int:
    value = (raw_label or "").strip().lower()
    if "not_toward" in value or "not-toward" in value or "notch_not" in value:
        return 2
    if "toward" in value:
        return 1
    return fallback_cls


def parse_polygons(json_path: Path, fallback_cls: int) -> dict[str, list[tuple[int, list[tuple[float, float]]]]]:
    data = load_json(json_path)
    parsed: dict[str, list[tuple[int, list[tuple[float, float]]]]] = {}

    for entry in data.values():
        filename = entry.get("filename")
        if not filename:
            continue

        regions = entry.get("regions", {})
        if isinstance(regions, dict):
            iter_regions = regions.values()
        else:
            iter_regions = regions

        items: list[tuple[int, list[tuple[float, float]]]] = []
        for region in iter_regions:
            shape = region.get("shape_attributes", {})
            if shape.get("name") != "polygon":
                continue

            xs = shape.get("all_points_x", [])
            ys = shape.get("all_points_y", [])
            if not xs or not ys or len(xs) != len(ys):
                continue

            points = [(float(x), float(y)) for x, y in zip(xs, ys)]
            label = region.get("region_attributes", {}).get("label", "")
            cls_id = parse_region_label(label, fallback_cls)
            items.append((cls_id, points))

        parsed[filename] = items

    return parsed


def parse_circle_labels(circle_dir: Path) -> dict[str, list[tuple[float, float, float, float]]]:
    parsed: dict[str, list[tuple[float, float, float, float]]] = {}

    if not circle_dir.exists():
        return parsed

    for txt_path in sorted(circle_dir.glob("*.txt")):
        circles: list[tuple[float, float, float, float]] = []
        with txt_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                # input class id can be 0 or 1 depending on source folder; normalize to class 0
                cx, cy, bw, bh = map(float, parts[1:5])
                circles.append((cx, cy, bw, bh))
        parsed[f"{txt_path.stem}.bmp"] = circles

    return parsed


def collect_samples(
    image_dir: Path,
    polygon_json: Path,
    circle_dir: Path,
    source: str,
    fallback_notch_cls: int,
) -> list[Sample]:
    polygons_by_image = parse_polygons(polygon_json, fallback_notch_cls)
    circles_by_image = parse_circle_labels(circle_dir)

    samples: list[Sample] = []
    image_paths = sorted(image_dir.glob("*.bmp")) + sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))

    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        height, width = image.shape[:2]
        labels: list[tuple[int, float, float, float, float]] = []

        for (cx, cy, bw, bh) in circles_by_image.get(image_path.name, []):
            labels.append((0, cx, cy, bw, bh))

        for cls_id, points in polygons_by_image.get(image_path.name, []):
            bbox = polygon_to_bbox_norm(points, width, height)
            if bbox is None:
                continue
            labels.append((cls_id, *bbox))

        if labels:
            samples.append(Sample(source=source, image_path=image_path, labels=labels))

    return samples


def build_ok_template_labels(ok_samples: list[Sample]) -> list[tuple[int, float, float, float, float]]:
    expected_counts = {0: 3, 1: 4}
    template_labels: list[tuple[int, float, float, float, float]] = []

    for cls_id, expected_count in expected_counts.items():
        slots: list[list[tuple[float, float, float, float]]] = [[] for _ in range(expected_count)]

        for sample in ok_samples:
            boxes = sorted([label[1:] for label in sample.labels if label[0] == cls_id], key=lambda b: b[0])
            if len(boxes) != expected_count:
                continue
            for idx, box in enumerate(boxes):
                slots[idx].append(box)

        for slot in slots:
            if not slot:
                continue
            cols = list(zip(*slot))
            template_box = tuple(float(np.median(values)) for values in cols)
            template_labels.append((cls_id, *template_box))

    return template_labels


def collect_template_ok_samples(
    extra_ok_dirs: Iterable[Path],
    template_labels: list[tuple[int, float, float, float, float]],
    existing_names: set[str],
) -> list[Sample]:
    samples: list[Sample] = []
    seen_names = set(existing_names)

    for directory in extra_ok_dirs:
        if not directory.exists():
            continue
        image_paths = sorted(directory.glob("*.bmp")) + sorted(directory.glob("*.png")) + sorted(directory.glob("*.jpg"))
        for image_path in image_paths:
            if image_path.name in seen_names:
                continue
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            seen_names.add(image_path.name)
            samples.append(Sample(source="ok_extra", image_path=image_path, labels=list(template_labels)))

    return samples


def split_samples(samples: list[Sample], val_ratio: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    rng = random.Random(seed)

    ok_samples = [s for s in samples if s.source == "ok"]
    extra_ok_samples = [s for s in samples if s.source == "ok_extra"]
    not_ok_samples = [s for s in samples if s.source == "not_ok"]

    rng.shuffle(ok_samples)
    rng.shuffle(extra_ok_samples)
    rng.shuffle(not_ok_samples)

    n_ok_val = max(1, int(round(len(ok_samples) * val_ratio))) if ok_samples else 0
    n_not_ok_val = max(1, int(round(len(not_ok_samples) * val_ratio))) if not_ok_samples else 0

    val_samples = ok_samples[:n_ok_val] + not_ok_samples[:n_not_ok_val]
    train_samples = ok_samples[n_ok_val:] + extra_ok_samples + not_ok_samples[n_not_ok_val:]

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def yolo_box_from_corners(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float] | None:
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = x2 - x1
    bh = y2 - y1
    if bw < 1e-4 or bh < 1e-4:
        return None
    return cx, cy, bw, bh


def transform_boxes(
    labels: Iterable[tuple[int, float, float, float, float]],
    transform_name: str,
) -> list[tuple[int, float, float, float, float]]:
    transformed: list[tuple[int, float, float, float, float]] = []

    for cls_id, cx, cy, bw, bh in labels:
        x1 = cx - bw / 2.0
        y1 = cy - bh / 2.0
        x2 = cx + bw / 2.0
        y2 = cy + bh / 2.0

        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

        if transform_name == "identity":
            new_corners = corners
        elif transform_name == "hflip":
            new_corners = [(1.0 - x, y) for (x, y) in corners]
        elif transform_name == "vflip":
            new_corners = [(x, 1.0 - y) for (x, y) in corners]
        elif transform_name == "rot90":
            new_corners = [(1.0 - y, x) for (x, y) in corners]
        elif transform_name == "rot180":
            new_corners = [(1.0 - x, 1.0 - y) for (x, y) in corners]
        elif transform_name == "rot270":
            new_corners = [(y, 1.0 - x) for (x, y) in corners]
        else:
            raise ValueError(f"Unsupported transform: {transform_name}")

        xs = [p[0] for p in new_corners]
        ys = [p[1] for p in new_corners]
        clipped = yolo_box_from_corners(min(xs), min(ys), max(xs), max(ys))
        if clipped is not None:
            transformed.append((cls_id, *clipped))

    return transformed


def transform_image(image: np.ndarray, transform_name: str) -> np.ndarray:
    if transform_name == "identity":
        return image.copy()
    if transform_name == "hflip":
        return cv2.flip(image, 1)
    if transform_name == "vflip":
        return cv2.flip(image, 0)
    if transform_name == "rot90":
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if transform_name == "rot180":
        return cv2.rotate(image, cv2.ROTATE_180)
    if transform_name == "rot270":
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unsupported transform: {transform_name}")


def apply_photometric_aug(image: np.ndarray, rng: random.Random) -> np.ndarray:
    out = image.astype(np.float32)

    # contrast + brightness
    alpha = rng.uniform(0.72, 1.35)
    beta = rng.uniform(-28.0, 28.0)
    out = out * alpha + beta

    # gamma exposure swing helps the brighter March captures
    gamma = rng.uniform(0.7, 1.45)
    out = 255.0 * np.power(np.clip(out / 255.0, 0.0, 1.0), gamma)

    # local contrast boost
    if rng.random() < 0.4:
        lab = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clip_limit = rng.uniform(1.5, 3.0)
        tile_size = rng.choice([4, 6, 8])
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        l = clahe.apply(l)
        out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR).astype(np.float32)

    # occasional blur
    if rng.random() < 0.35:
        ksize = rng.choice([3, 5])
        out = cv2.GaussianBlur(out, (ksize, ksize), 0)

    # occasional sharpen after blur/brightness shifts
    if rng.random() < 0.3:
        blurred = cv2.GaussianBlur(out, (0, 0), rng.uniform(1.0, 2.0))
        out = cv2.addWeighted(out, 1.35, blurred, -0.35, 0)

    # mild gaussian noise
    if rng.random() < 0.65:
        sigma = rng.uniform(2.0, 15.0)
        noise = np.random.normal(0.0, sigma, out.shape).astype(np.float32)
        out = out + noise

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def write_labels(label_path: Path, labels: list[tuple[int, float, float, float, float]]) -> None:
    with label_path.open("w", encoding="utf-8") as f:
        for cls_id, cx, cy, bw, bh in labels:
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_dataset(
    out_dir: Path,
    train_samples: list[Sample],
    val_samples: list[Sample],
    ok_augment_factor: int,
    not_ok_augment_factor: int,
    seed: int,
) -> dict[str, dict[int, int]]:
    rng = random.Random(seed)

    images_train = out_dir / "images" / "train"
    images_val = out_dir / "images" / "val"
    labels_train = out_dir / "labels" / "train"
    labels_val = out_dir / "labels" / "val"

    ensure_clean_dir(images_train)
    ensure_clean_dir(images_val)
    ensure_clean_dir(labels_train)
    ensure_clean_dir(labels_val)

    stats = {
        "train": {0: 0, 1: 0, 2: 0},
        "val": {0: 0, 1: 0, 2: 0},
    }

    def write_split(samples: list[Sample], split: str) -> None:
        img_dir = images_train if split == "train" else images_val
        lbl_dir = labels_train if split == "train" else labels_val

        for sample in samples:
            base_name = f"{sample.source}_{sample.image_path.stem}"
            out_image = img_dir / f"{base_name}.jpg"
            out_label = lbl_dir / f"{base_name}.txt"

            image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue

            cv2.imwrite(str(out_image), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            write_labels(out_label, sample.labels)

            for cls_id, _, _, _, _ in sample.labels:
                stats[split][cls_id] += 1

    write_split(train_samples, "train")
    write_split(val_samples, "val")

    # Synthetic train augmentation for both OK and NOT_OKAY samples.
    augment_factors = {
        "ok": max(0, ok_augment_factor),
        "ok_extra": max(0, ok_augment_factor),
        "not_ok": max(0, not_ok_augment_factor),
    }
    geom_ops = ["hflip", "rot180", "vflip", "rot90", "rot270", "identity"]

    for source_name, augment_factor in augment_factors.items():
        if augment_factor <= 0:
            continue

        aug_samples = [s for s in train_samples if s.source == source_name]
        for sample in aug_samples:
            image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue

            for idx in range(augment_factor):
                op = geom_ops[idx % len(geom_ops)]
                aug_img = transform_image(image, op)
                aug_img = apply_photometric_aug(aug_img, rng)
                aug_labels = transform_boxes(sample.labels, op)
                if not aug_labels:
                    continue

                out_stem = f"syn_{source_name}_{sample.image_path.stem}_{idx:03d}"
                out_image = images_train / f"{out_stem}.jpg"
                out_label = labels_train / f"{out_stem}.txt"

                cv2.imwrite(str(out_image), aug_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                write_labels(out_label, aug_labels)

                for cls_id, _, _, _, _ in aug_labels:
                    stats["train"][cls_id] += 1

    yaml_text = (
        f"path: {out_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "nc: 3\n"
        "names:\n"
        "  0: circle\n"
        "  1: notch_toward_hole\n"
        "  2: notch_not_toward_hole\n"
    )
    (out_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")

    return stats


def main() -> None:
    args = parse_args()
    if args.neg_augment_factor is not None:
        args.not_ok_augment_factor = args.neg_augment_factor

    data_dir: Path = args.data_dir
    out_dir: Path = args.out_dir

    ok_samples = collect_samples(
        image_dir=data_dir / "OK",
        polygon_json=data_dir / "OK" / "labels_okay-labales_2026-02-19-07-29-49.json",
        circle_dir=data_dir / "OK" / "circle",
        source="ok",
        fallback_notch_cls=1,
    )
    ok_template_labels = build_ok_template_labels(ok_samples)
    extra_ok_samples = collect_template_ok_samples(
        extra_ok_dirs=args.extra_ok_dir,
        template_labels=ok_template_labels,
        existing_names={sample.image_path.name for sample in ok_samples},
    )
    not_ok_samples = collect_samples(
        image_dir=data_dir / "NOT_OKAY",
        polygon_json=data_dir / "NOT_OKAY" / "not_okay.json",
        circle_dir=data_dir / "NOT_OKAY" / "not_okay_circle",
        source="not_ok",
        fallback_notch_cls=2,
    )

    all_samples = ok_samples + extra_ok_samples + not_ok_samples

    if not all_samples:
        raise RuntimeError("No labeled samples found. Check data directory and annotation files.")

    train_samples, val_samples = split_samples(all_samples, args.val_ratio, args.seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    stats = write_dataset(
        out_dir=out_dir,
        train_samples=train_samples,
        val_samples=val_samples,
        ok_augment_factor=args.ok_augment_factor,
        not_ok_augment_factor=args.not_ok_augment_factor,
        seed=args.seed,
    )

    print("=" * 70)
    print("Dataset Build Complete")
    print("=" * 70)
    print(f"OK samples       : {len(ok_samples)}")
    print(f"Extra OK samples : {len(extra_ok_samples)}")
    print(f"NOT_OKAY samples : {len(not_ok_samples)}")
    print(f"OK aug factor    : {args.ok_augment_factor}")
    print(f"NOT_OKAY aug     : {args.not_ok_augment_factor}")
    print(f"Train images     : {len(list((out_dir / 'images' / 'train').glob('*')))}")
    print(f"Val images       : {len(list((out_dir / 'images' / 'val').glob('*')))}")
    print("Train class counts:", stats["train"])
    print("Val class counts  :", stats["val"])
    print(f"data.yaml        : {out_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
