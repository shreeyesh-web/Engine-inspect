#!/usr/bin/env python3
"""
Model Evaluation Script for YOLOv8
Generates confusion matrix and p-r-f1 charts.
"""

import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO

# --- Config & Paths ---
# Using joined paths for a more manual feel
ROOT = r"D:\src\src"
MODEL_FILE = os.path.join(ROOT, "models", "best.pt")
DATA_YAML  = os.path.join(ROOT, "dataset", "data.yaml")
DATA_DIR   = Path(ROOT) / "dataset"
OUT_DIR    = Path(ROOT) / "results"

# Labels for the engine defect project
LABELS = {0: "circle", 1: "notch_toward_hole", 2: "notch_not_toward_hole"}

# Class colors for plotting (B-G-R)
MAP_COLORS = {
    0: (255, 193, 84),   # blue
    1: (60, 200, 26),    # green
    2: (43, 57, 192),    # red
}

# TODO: Support different IoU thresholds for precision-recall curves
IOU_LIMIT = 0.5

def get_data():
    """Loads images from val or train split."""
    for split in ["val", "train"]:
        img_p = DATA_DIR / "images" / split
        lbl_p = DATA_DIR / "labels" / split

        if not img_p.exists():
            continue

        imgs = sorted(img_p.glob("*.jpg"))
        if not imgs: imgs = sorted(img_p.glob("*.png"))
        if not imgs: continue

        print(f"--> Found {len(imgs)} images in {split} folder")

        # Just take a subset if we are using the train folder for testing
        if split == "train":
            random.seed(42)
            random.shuffle(imgs)
            imgs = imgs[:max(5, int(len(imgs) * 0.2))]

        valid_i, valid_l = [], []
        for p in imgs:
            lp = lbl_p / p.with_suffix(".txt").name
            if lp.exists():
                valid_i.append(p)
                valid_l.append(lp)
        
        return valid_i, valid_l

    return [], []

def get_iou(b1, b2):
    """Standard IoU calculation for boxes."""
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])

    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    u = a1 + a2 - inter
    return inter / u if u > 0 else 0.0

def match_preds(preds, actuals):
    """Matches pred boxes to ground truth for scoring."""
    y_true, y_pred = [], []
    matched = set()

    for p in preds:
        best_iou, best_idx = 0, -1
        for i, a in enumerate(actuals):
            if i in matched: continue
            iou = get_iou(p['box'], a['box'])
            if iou > best_iou:
                best_iou, best_idx = iou, i
        
        if best_iou >= IOU_LIMIT and best_idx != -1:
            y_true.append(actuals[best_idx]['cls'])
            y_pred.append(p['cls'])
            matched.add(best_idx)
        else:
            y_true.append(-1) # Background/False Positive
            y_pred.append(p['cls'])

    for i, a in enumerate(actuals):
        if i not in matched:
            y_true.append(a['cls'])
            y_pred.append(-1) # Miss/False Negative
            
    return y_true, y_pred

def main():
    print("Starting evaluation...")
    if not os.path.exists(MODEL_FILE):
        print(f"Error: Could not find model at {MODEL_FILE}")
        return

    net = YOLO(MODEL_FILE)
    img_list, lbl_list = get_data()

    if not img_list:
        print("Check dataset paths - no data found.")
        return

    all_t, all_p = [], []
    for idx, (ip, lp) in enumerate(zip(img_list, lbl_list)):
        raw = cv2.imread(str(ip))
        h, w = raw.shape[:2]
        res = net(raw, verbose=False)[0]

        # Parse labels
        actuals = []
        with open(lp) as f:
            for line in f:
                d = line.split()
                xc, yc, bw, bh = map(float, d[1:])
                actuals.append({
                    'cls': int(d[0]),
                    'box': [int((xc-bw/2)*w), int((yc-bh/2)*h), 
                            int((xc+bw/2)*w), int((yc+bh/2)*h)]
                })

        # Process detections
        preds = []
        if res.boxes:
            for b in res.boxes:
                if b.conf[0] < 0.25: continue
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                preds.append({'cls': int(b.cls[0]), 'box': [x1, y1, x2, y2]})

        t, p = match_preds(preds, actuals)
        all_t.extend(t); all_p.extend(p)

        if (idx+1) % 10 == 0:
            print(f"Processed {idx+1}/{len(img_list)} images")

    # Metrics calculation logic
    num_c = len(LABELS)
    matrix = np.zeros((num_c, num_c), dtype=int)
    stats = {}

    for c in range(num_c):
        tp = sum(1 for t, p in zip(all_t, all_p) if t == c and p == c)
        fp = sum(1 for t, p in zip(all_t, all_p) if t != c and p == c)
        fn = sum(1 for t, p in zip(all_t, all_p) if t == c and p != c)
        
        prec = tp/(tp+fp) if (tp+fp) > 0 else 0
        rec  = tp/(tp+fn) if (tp+fn) > 0 else 0
        f1   = (2*prec*rec)/(prec+rec) if (prec+rec) > 0 else 0
        stats[c] = {'p': prec, 'r': rec, 'f1': f1}
        print(f"Class {LABELS[c]} -> F1: {f1:.3f}")

    # Generate Confusion Matrix data
    for t, p in zip(all_t, all_p):
        if t >= 0 and p >= 0: matrix[t][p] += 1

    # --- Plotting Section ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Simple heatmap
    plt.figure(figsize=(7,5))
    norm_m = matrix / matrix.sum(axis=1, keepdims=True)
    plt.imshow(norm_m, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig(OUT_DIR / "confusion_matrix.png")
    
    # Bar chart for metrics
    plt.figure(figsize=(8,5))
    x_axis = np.arange(num_c)
    plt.bar(x_axis - 0.2, [stats[i]['p'] for i in range(num_c)], 0.2, label='Prec')
    plt.bar(x_axis, [stats[i]['r'] for i in range(num_c)], 0.2, label='Rec')
    plt.bar(x_axis + 0.2, [stats[i]['f1'] for i in range(num_c)], 0.2, label='F1')
    plt.xticks(x_axis, [LABELS[i] for i in range(num_c)])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(OUT_DIR / "metrics.png")

    print(f"Done. Check {OUT_DIR} for results.")

if __name__ == "__main__":
    main()