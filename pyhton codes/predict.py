import os
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
from verdict_rules import strict_verdict_from_names

# Basic Setup 
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "best.onnx"
SAVE_DIR = ROOT / "predictions"

# Defective Classes
CLASSES = {
    0: "Circle",
    1: "Notch Toward Hole",
    2: "Notch Not Toward Hole"
}

COLORS = {
    0: (255, 193, 84),   # Blue-ish
    1: (60, 122, 26),    # Green
    2: (43, 57, 192)     # Red
}

def get_verdict(detected_ids):

    names = {CLASSES[i] for i in detected_ids if i in CLASSES}
    res = strict_verdict_from_names(names)
    status = "VALID" if res.verdict == "OK" else "INVALID"
    return status, list(res.reasons)

def draw_ui_panel(img, status, reasons, counts):
    h, w = img.shape[:2]
    overlay = img.copy()
    panel_x = w - 450
    cv2.rectangle(overlay, (panel_x, 20), (w - 20, 350), (30, 30, 30), -1)

    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    header_color = (0, 200, 0) if status == "VALID" else (0, 0, 200)
    cv2.rectangle(img, (panel_x, 20), (w - 20, 80), header_color, -1)
    cv2.putText(img, f"RESULT: {status}", (panel_x + 15, 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    # Reasons
    y = 120
    for r in reasons:
        cv2.putText(img, f"- {r}", (panel_x + 15, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 40

    # Class Status
    y += 20
    cv2.putText(img, "Detection Summary:", (panel_x + 15, y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)
    
    y += 40
    for cid, name in CLASSES.items():
        found = "YES" if counts[cid] > 0 else "NO"
        color = (100, 255, 100) if found == "YES" else (150, 150, 150)
        cv2.putText(img, f"{name}: {found}", (panel_x + 15, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        y += 30

    return img

def annotate_image(image, detections, status, reasons):

    output = image.copy()
    counts = {0: 0, 1: 0, 2: 0}

    # Drawing the Boxes
    for det in detections:
        cls_id = int(det.cls[0])
        conf = float(det.conf[0])
        x1, y1, x2, y2 = map(int, det.xyxy[0])
        
        counts[cls_id] += 1
        color = COLORS.get(cls_id, (255, 255, 255))
        
        # Simple box and labels
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASSES[cls_id]} {conf:.2f}"
        cv2.putText(output, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Drawing the info panel
    output = draw_ui_panel(output, status, reasons, counts)
    
    return output

