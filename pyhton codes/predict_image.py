import argparse
import os
import cv2
from pathlib import Path
from ultralytics import YOLO
from verdict_rules import strict_verdict_from_counts

# Settings
CLASSES = {0: "circle", 1: "notch_toward_hole", 2: "notch_not_toward_hole"}
COLORS = {0: (0, 255, 0), 1: (0, 220, 255), 2: (0, 0, 255)}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--weights", default="models/best.pt", help="Path to model")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--save", action="store_true")
    return parser.parse_args()

def main():
    args = get_args()
    
    # Check if image exists
    img_path = Path(args.image)
    if not img_path.exists():
        print(f"File not found: {args.image}")
        return

    # Load Model
    print(f"Loading model: {args.weights}")
    model = YOLO(args.weights)
    results = model.predict(source=str(img_path), conf=args.conf, verbose=False)[0]
    img = cv2.imread(str(img_path))
    counts = {0: 0, 1: 0, 2: 0}

    if results.boxes:
        for b in results.boxes:
            c_id = int(b.cls[0])
            conf = float(b.conf[0])
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            
            counts[c_id] += 1
            
            # Draw simple box and label
            color = COLORS.get(c_id, (255, 255, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{CLASSES[c_id]} {conf:.2f}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Calculating the Verdict
    res = strict_verdict_from_counts(counts)
    verdict = res.verdict

    
    print(f"Results for: {img_path.name}")
    print(f"Counts: Circle={counts[0]}, Toward={counts[1]}, Away={counts[2]}")
    print(f"FINAL VERDICT: {verdict}")
    

    if args.save:
        out_dir = Path("results/single")
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.putText(img, f"RESULT: {verdict}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        save_path = out_dir / f"{img_path.stem}_result.jpg"
        cv2.imwrite(str(save_path), img)
        print(f"Saved result image to: {save_path}")

if __name__ == "__main__":
    main()