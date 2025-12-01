
# 🌱 Plant Image ML — Student README  
**Raspberry Pi → Linux Server → YOLO Object Detection Pipeline**  


This guide walks you **step‑by‑step** through the entire workflow to train your own plant-detection ML model on our Linux GPU server.  

---

# 📌 1. Create Your Working Folder

All work happens inside your own directory on the server.

```bash
cd ~/Desktop
mkdir <yourname>_plants
cd <yourname>_plants
```

Example:

```bash
mkdir alex_plants
cd alex_plants
```

---

# 📌 2. Clone the Teacher Repo Into Your Folder

Do **NOT** clone into `/Desktop/plants`.  
Clone inside *your personal folder*.

```bash
git clone https://github.com/sallywoolweaver/plantsImageML .
```

You should now see:

```
convert_masks_to_yolo.py
define_masks.py
debug_model.py
plant_data.yaml
yolov8n.pt
...
```

---

# 📌 3. Activate Python Virtual Environment

Every student uses the shared `.venv` located on the server.

```bash
cd ~/Desktop/plants/.venv
source bin/activate
```

You should now see:

```
(.venv) compsci@...
```

Then return to your folder:

```bash
cd ~/Desktop/<yourname>_plants
```

---

# 📌 4. Upload Your Images From the Raspberry Pi

Put your `.jpg` images into a folder:

```
/home/compsci/Desktop/<yourname>_plants/raspberry_images/
```

Make sure all filenames are valid.  
Avoid spaces—use `image001.jpg`, `image002.jpg`, etc.

---

# 📌 5. Draw Masks (Polygon annotations) for Each Plant

You must manually outline each plant **one time** in a few reference images.

Run:

```bash
python define_masks.py raspberry_images masks.json --plant_ids 1 2 3
```

Controls:

| Action | Key |
|-------|------|
| Add a point | left click |
| Undo point | backspace |
| Reset polygon | ESC |
| Finish plant | ENTER |
| Quit without saving | Q |

This saves a `masks.json` file with your polygons.

---

# 📌 6. Convert Masks → Bounding Boxes (YOLO format)

YOLO requires bounding boxes, not polygons.

Run:

```bash
python convert_masks_to_yolo.py masks.json raspberry_images
```

This generates:

```
plants_yolo_dataset/
    images/
    labels_yolo/
```

---

# 📌 7. Understand `plant_data.yaml` (important!)

YOLO training uses a config file called **YAML**.

### 📘 What is YAML?
A `.yaml` file is a simple text format used to store **settings**, like:

- where your images are  
- where your labels are  
- what class numbers mean  

Example:

```yaml
path: plants_yolo_dataset

train: images/train
val: images/val

names:
  0: Plant 1
  1: Plant 2
  2: Plant 3
```

YOLO reads this file automatically.

---

# 📌 8. Split the Dataset Into Training & Validation

Run the script:

```bash
python file_structure.py
```

This creates:

```
plants_yolo_dataset/images/train
plants_yolo_dataset/images/val
plants_yolo_dataset/labels/train
plants_yolo_dataset/labels/val
```

---

# 📌 9. Train YOLO on the Server GPU

Run:

```bash
yolo detect train model=yolov8n.pt data=plant_data.yaml epochs=30 imgsz=640 device=0
```

This will produce:

```
runs/detect/train/
    weights/best.pt
    weights/last.pt
```

`best.pt` = your trained model.

---

# 📌 10. Test Your Model on Your Own Images

```bash
python debug_model.py runs/detect/train/weights/best.pt raspberry_images
```

Output example:

```
class=0 conf=0.51
class=1 conf=0.10
```

If Plant 2 has low confidence, lower the threshold:

```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=raspberry_images conf=0.10
```

---

# 📌 11. Troubleshooting Guide

### ❌ YOLO shows only Plant 1  
Probably confidence is too high. Lower it:

```bash
conf=0.05
```

### ❌ “No labels found”  
Run:

```bash
python debug_labels.py
```

Check if both classes appear.

### ❌ Predict window not showing  
Make sure backend is:

```bash
python3 -c "import matplotlib; print(matplotlib.get_backend())"
```

Should output: `TkAgg`.

---

# 📌 12. Final Deliverable for Your Grade

Each student must turn in:

### ✅ 1. Your YOLO model folder  
`runs/detect/train/weights/best.pt`

### ✅ 2. A short reflection describing:
- How well your model works  
- Precision/Recall for Plant 1 and Plant 2  
- What confused the model  
- What you would improve with more time/data  

### ✅ 3. A screenshot of detections on new images

### ✅ 4. Explanation of:
- What YAML means  
- Why we annotated polygons  
- Why YOLO requires bounding boxes  
- Why confidence threshold matters  

---

# 🎉 You Now Have Your Own Object Detection Model!

Your model is *exactly* the same technology companies use for:

- Self-driving cars  
- Farm crop monitoring  
- Medical image detection  
- Robotics  



---

If anything breaks, ask questions — the errors are part of the learning process.