from ultralytics import YOLO
import sys

if len(sys.argv) < 3:
    print("Usage: python debug_predict.py /path/to/best.pt /path/to/image_or_folder")
    sys.exit(1)

model_path = sys.argv[1]
source = sys.argv[2]

model = YOLO(model_path)

results = model(source, imgsz=640, conf=0.1)  # low conf to see more

for i, res in enumerate(results):
    print(f"\n=== Result {i} ===")
    if res.path:
        print("Image:", res.path)

    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        print("No boxes detected.")
        continue

    for j, box in enumerate(boxes):
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        xyxy = box.xyxy[0].tolist()
        print(f"  Box {j}: class={cls_id} conf={conf:.3f} xyxy={xyxy}")
