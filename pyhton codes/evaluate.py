#!/usr/bin/env python3
"""
Evaluate YOLOv8 model and generate:
- Confusion matrix
- Precision, Recall, F1 per class
- Sample prediction images
"""

import os
import random
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

from ultralytics import YOLO

# ─────────────────────────────────────────
# YOUR EXACT PATHS
# ─────────────────────────────────────────
MODEL_PATH  = Path(r"D:\src\src\models\best.pt")
YAML_PATH   = Path(r"D:\src\src\dataset\data.yaml")
DATA_ROOT   = Path(r"D:\src\src\dataset")
RESULTS_DIR = Path(r"D:\src\src\results")

CLASS_NAMES = {
    0: "circle",
    1: "notch_toward_hole",
    2: "notch_not_toward_hole",
}

COLORS = {
    0: (84,  193, 255),   # blue  — circle
    1: (26,  200,  60),   # green — toward hole
    2: (192,  57,  43),   # red   — not toward hole
}

# ─────────────────────────────────────────
# STEP 1 — Find images for evaluation
# ─────────────────────────────────────────

def prepare_test_images() -> tuple[list[Path], list[Path]]:
    for split in ["val", "train"]:
        img_dir   = DATA_ROOT / "images" / split
        label_dir = DATA_ROOT / "labels" / split

        if not img_dir.exists():
            continue

        all_images = sorted(img_dir.glob("*.jpg"))
        if not all_images:
            all_images = sorted(img_dir.glob("*.png"))
        if not all_images:
            continue

        print(f"Using images from: {img_dir}")
        print(f"Total images found: {len(all_images)}")

        if split == "val":
            test_images = all_images
        else:
            random.seed(42)
            random.shuffle(all_images)
            n = max(5, int(len(all_images) * 0.2))
            test_images = all_images[:n]

        valid_images = []
        valid_labels = []
        for img_path in test_images:
            label_path = label_dir / img_path.with_suffix(".txt").name
            if label_path.exists():
                valid_images.append(img_path)
                valid_labels.append(label_path)

        print(f"Images with labels: {len(valid_images)}")
        return valid_images, valid_labels

    raise FileNotFoundError(
        f"No images found in {DATA_ROOT}/images/val or /train\n"
        f"Check your dataset folder structure."
    )

# ─────────────────────────────────────────
# STEP 2 — Parse YOLO label file
# ─────────────────────────────────────────

def parse_label_file(
    label_path: Path, img_w: int, img_h: int
) -> list[dict]:
    detections = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            xc = float(parts[1]) * img_w
            yc = float(parts[2]) * img_h
            w  = float(parts[3]) * img_w
            h  = float(parts[4]) * img_h
            x1 = int(xc - w/2)
            y1 = int(yc - h/2)
            x2 = int(xc + w/2)
            y2 = int(yc + h/2)
            detections.append({
                'cls': cls_id,
                'box': [x1, y1, x2, y2]
            })
    return detections

# ─────────────────────────────────────────
# STEP 3 — IoU calculation
# ─────────────────────────────────────────

def calculate_iou(box1: list, box2: list) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2-x1) * max(0, y2-y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

# ─────────────────────────────────────────
# STEP 4 — Match predictions to ground truth
# ─────────────────────────────────────────

def match_predictions(
    pred_boxes: list[dict],
    true_boxes: list[dict],
    iou_threshold: float = 0.5
) -> tuple[list, list]:
    true_classes = []
    pred_classes = []
    matched_true = set()

    for pred in pred_boxes:
        best_iou  = 0
        best_true = None
        best_idx  = -1

        for idx, true in enumerate(true_boxes):
            if idx in matched_true:
                continue
            iou = calculate_iou(pred['box'], true['box'])
            if iou > best_iou:
                best_iou  = iou
                best_true = true
                best_idx  = idx

        if best_iou >= iou_threshold and best_true is not None:
            true_classes.append(best_true['cls'])
            pred_classes.append(pred['cls'])
            matched_true.add(best_idx)
        else:
            true_classes.append(-1)
            pred_classes.append(pred['cls'])

    for idx, true in enumerate(true_boxes):
        if idx not in matched_true:
            true_classes.append(true['cls'])
            pred_classes.append(-1)

    return true_classes, pred_classes

# ─────────────────────────────────────────
# STEP 5 — Calculate metrics
# ─────────────────────────────────────────

def calculate_metrics(
    all_true: list, all_pred: list, num_classes: int
) -> dict:
    metrics = {}
    for cls_id in range(num_classes):
        tp = sum(1 for t, p in zip(all_true, all_pred)
                 if t == cls_id and p == cls_id)
        fp = sum(1 for t, p in zip(all_true, all_pred)
                 if t != cls_id and p == cls_id)
        fn = sum(1 for t, p in zip(all_true, all_pred)
                 if t == cls_id and p != cls_id)

        precision = tp / (tp+fp) if (tp+fp) > 0 else 0.0
        recall    = tp / (tp+fn) if (tp+fn) > 0 else 0.0
        f1 = (2*precision*recall / (precision+recall)
              if (precision+recall) > 0 else 0.0)

        metrics[cls_id] = {
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': precision,
            'recall':    recall,
            'f1':        f1
        }
    return metrics

# ─────────────────────────────────────────
# STEP 6 — Build confusion matrix
# ─────────────────────────────────────────

def build_confusion_matrix(
    all_true: list, all_pred: list, num_classes: int
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(all_true, all_pred):
        if true >= 0 and pred >= 0:
            matrix[true][pred] += 1
    return matrix

# ─────────────────────────────────────────
# STEP 7 — Plot confusion matrix
# ─────────────────────────────────────────

def plot_confusion_matrix(
    matrix: np.ndarray,
    save_path: Path
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(CLASS_NAMES)
    display_labels = ["Circle", "Toward\nHole", "Not Toward\nHole"]

    fig, ax = plt.subplots(figsize=(8, 6))

    row_sums    = matrix.sum(axis=1, keepdims=True)
    norm_matrix = np.where(row_sums > 0, matrix/row_sums, 0)

    im = ax.imshow(norm_matrix, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Proportion')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(display_labels, fontsize=11)
    ax.set_yticklabels(display_labels, fontsize=11)
    ax.set_xlabel("Predicted",  fontsize=13, fontweight='bold')
    ax.set_ylabel("Actual",     fontsize=13, fontweight='bold')
    ax.set_title(
        "Confusion Matrix — EngineInspect Defect Detection",
        fontsize=13, fontweight='bold', pad=15
    )

    for i in range(n):
        for j in range(n):
            count = matrix[i][j]
            pct   = norm_matrix[i][j] * 100
            color = "white" if norm_matrix[i][j] > 0.5 else "black"
            ax.text(j, i, f"{count}\n({pct:.0f}%)",
                    ha='center', va='center',
                    color=color, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

# ─────────────────────────────────────────
# STEP 8 — Plot metrics bar chart
# ─────────────────────────────────────────

def plot_metrics(metrics: dict, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    short      = ["Circle", "Toward Hole", "Not Toward\nHole"]
    precisions = [metrics[i]['precision'] for i in range(len(CLASS_NAMES))]
    recalls    = [metrics[i]['recall']    for i in range(len(CLASS_NAMES))]
    f1s        = [metrics[i]['f1']        for i in range(len(CLASS_NAMES))]

    x     = np.arange(len(CLASS_NAMES))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x-width, precisions, width,
                   label='Precision', color='#2196F3', alpha=0.85)
    bars2 = ax.bar(x,       recalls,   width,
                   label='Recall',    color='#4CAF50', alpha=0.85)
    bars3 = ax.bar(x+width, f1s,       width,
                   label='F1 Score',  color='#FF9800', alpha=0.85)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2,
                h + 0.01, f'{h:.2f}',
                ha='center', va='bottom',
                fontsize=9, fontweight='bold'
            )

    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=13, fontweight='bold')
    ax.set_title(
        "Precision · Recall · F1 per Class — EngineInspect",
        fontsize=14, fontweight='bold', pad=15
    )
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

# ─────────────────────────────────────────
# STEP 9 — Save sample prediction images
# ─────────────────────────────────────────

def save_sample_predictions(
    model: YOLO,
    image_paths: list[Path],
    label_paths: list[Path],
    save_dir: Path,
    n_samples: int = 6
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    samples = list(zip(image_paths, label_paths))[:n_samples]

    for img_path, label_path in samples:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        # Ground truth in white
        true_boxes = parse_label_file(label_path, w, h)
        for tb in true_boxes:
            x1,y1,x2,y2 = tb['box']
            cls_name = CLASS_NAMES.get(tb['cls'], '?')
            cv2.rectangle(img, (x1,y1), (x2,y2),
                          (255,255,255), 1)
            cv2.putText(img, f"GT:{cls_name[:8]}",
                        (x1, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (255,255,255), 1)

        # Predictions in class colors
        pred = model(img, verbose=False)[0]
        if pred.boxes is not None:
            for box in pred.boxes:
                cls_id       = int(box.cls[0])
                conf         = float(box.conf[0])
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                color        = COLORS.get(cls_id, (128,128,128))
                label        = (f"{CLASS_NAMES.get(cls_id,'?')}"
                                f" {conf:.2f}")
                cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
                cv2.putText(img, label,
                            (x1, y2+15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, color, 1)

        out = save_dir / f"pred_{img_path.name}"
        cv2.imwrite(str(out), img)

    print(f"Saved {len(samples)} sample images in: {save_dir}")

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main():
    print("=" * 60)
    print("EngineInspect — Model Evaluation")
    print("=" * 60)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not YAML_PATH.exists():
        raise FileNotFoundError(f"YAML not found: {YAML_PATH}")

    print(f"Model : {MODEL_PATH}")
    print(f"YAML  : {YAML_PATH}")

    model = YOLO(str(MODEL_PATH))

    image_paths, label_paths = prepare_test_images()

    if not image_paths:
        print("ERROR: No labelled images found")
        return

    # Run evaluation
    all_true = []
    all_pred = []

    print("\nRunning predictions...")
    for i, (img_path, label_path) in enumerate(
        zip(image_paths, label_paths)
    ):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        pred = model(img, verbose=False)[0]

        true_boxes = parse_label_file(label_path, w, h)
        pred_boxes = []

        if pred.boxes is not None:
            for box in pred.boxes:
                cls_id       = int(box.cls[0])
                conf         = float(box.conf[0])
                if conf < 0.25:
                    continue
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                pred_boxes.append({
                    'cls':  cls_id,
                    'conf': conf,
                    'box':  [x1,y1,x2,y2]
                })

        tc, pc = match_predictions(pred_boxes, true_boxes)
        all_true.extend(tc)
        all_pred.extend(pc)

        if (i+1) % 5 == 0 or (i+1) == len(image_paths):
            print(f"  {i+1}/{len(image_paths)} images done")

    # Metrics
    num_classes = len(CLASS_NAMES)
    metrics     = calculate_metrics(all_true, all_pred, num_classes)
    matrix      = build_confusion_matrix(
                    all_true, all_pred, num_classes)

    # Print results
    print("\n" + "=" * 60)
    print("PER CLASS RESULTS")
    print("=" * 60)

    total_tp = total_fp = total_fn = 0

    for cls_id in range(num_classes):
        m = metrics[cls_id]
        total_tp += m['tp']
        total_fp += m['fp']
        total_fn += m['fn']
        print(f"\n{CLASS_NAMES[cls_id]}:")
        print(f"  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}")
        print(f"  Precision : {m['precision']:.3f}")
        print(f"  Recall    : {m['recall']:.3f}")
        print(f"  F1        : {m['f1']:.3f}")

    overall_p  = (total_tp/(total_tp+total_fp)
                  if (total_tp+total_fp) > 0 else 0)
    overall_r  = (total_tp/(total_tp+total_fn)
                  if (total_tp+total_fn) > 0 else 0)
    overall_f1 = (2*overall_p*overall_r/(overall_p+overall_r)
                  if (overall_p+overall_r) > 0 else 0)

    print("\n" + "=" * 60)
    print("OVERALL")
    print("=" * 60)
    print(f"Precision : {overall_p:.3f}")
    print(f"Recall    : {overall_r:.3f}")
    print(f"F1 Score  : {overall_f1:.3f}")

    # YOLO built-in mAP
    print("\nRunning YOLO validation for mAP...")
    try:
        val_results = model.val(
            data=str(YAML_PATH), verbose=False
        )
        print(f"mAP@0.5      : {val_results.box.map50:.3f}")
        print(f"mAP@0.5:0.95 : {val_results.box.map:.3f}")
    except Exception as e:
        print(f"mAP calculation skipped: {e}")

    # Save charts
    print("\nGenerating charts...")
    plot_confusion_matrix(
        matrix,
        RESULTS_DIR / "confusion_matrix.png"
    )
    plot_metrics(
        metrics,
        RESULTS_DIR / "metrics.png"
    )

    # Sample predictions
    print("\nSaving sample predictions...")
    save_sample_predictions(
        model, image_paths, label_paths,
        RESULTS_DIR / "samples"
    )

    print("\n" + "=" * 60)
    print("EngineInspect — Evaluation Complete")
    print("Results saved in:")
    print(f"  {RESULTS_DIR / 'confusion_matrix.png'}")
    print(f"  {RESULTS_DIR / 'metrics.png'}")
    print(f"  {RESULTS_DIR / 'samples'}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
