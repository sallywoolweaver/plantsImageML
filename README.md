
# 🌱 Plant Image ML — Student Guide  


This project teaches you how to train your **own object detection machine learning model** to recognize your **three plants** in your Raspberry Pi watering system photos.

You will:

- Build a **supervised learning** dataset from your own images  
- Convert polygon masks → YOLO **bounding boxes**  
- Train a **YOLO** object detection model on the Linux ML server (RTX 4080)  
- Evaluate your model using **validation images & metrics**  
- Connect all of this to the **IB Computer Science 2027 syllabus (A4.x Machine Learning)**

---

## 🌐 Big Picture: What Kind of ML Is This?

This project is:

- **Supervised learning**  
  - You provide labelled examples (plants with bounding boxes)  
  - The model learns a mapping: `image → (plant class + box location)`
- A **classification + localization** task  
  - Classifies: Plant 1 vs Plant 2 vs Plant 3  
  - Localizes: where each plant is in the image
- Implemented with **YOLO** (You Only Look Once)  
  - A deep learning–based **object detection** model

IB links (HL Machine Learning):

- A4.2: Data preprocessing  
- A4.3: Supervised learning, classification  
- A4.5: Data sets and features  
- A4.6: Perception and pattern recognition  
- A4.7: Evaluation and performance measures  

---

# 1️⃣ Setup: Your Personal Folder on the Server

Each student works in **their own folder**.

On the Linux server:

```bash
cd ~/Desktop
mkdir <yourname>_plants
cd <yourname>_plants
```

Example:

```bash
cd ~/Desktop
mkdir maya_plants
cd maya_plants
```

---

# 2️⃣ Get the Project Files Into Your Folder

Clone the teacher repo **inside your own folder**:

```bash
git clone https://github.com/sallywoolweaver/plantsImageML .
```

You should now see files like:

```text
convert_masks_to_yolo.py
define_masks.py
debug_model.py
file_structure.py
yolov8n.pt
...
```

Now copy in **your own data**:

- `masks.json` (exported from the Pi; contains your polygons)  
- `raspberry_images/` (folder with your plant photos)

Your folder should look like:

```text
~/Desktop/<yourname>_plants/
    masks.json
    raspberry_images/
    convert_masks_to_yolo.py
    file_structure.py
    debug_model.py
    define_masks.py
    yolov8n.pt
    ...
```

> 🔎 You are reusing your existing masks — you do **not** redraw them.

---

# 3️⃣ Activate the Shared Python Environment

We use a **shared virtual environment** already set up on the server.

```bash
source ~/Desktop/plants/.venv/bin/activate
```

Your prompt should now start with:

```text
(.venv) compsci@...
```

Then:

```bash
cd ~/Desktop/<yourname>_plants
```

Every command from now on runs **inside** this environment.

---

# 4️⃣ Convert Polygons → YOLO Bounding Boxes

Your `masks.json` contains **polygons** (multiple points clicked around each plant).  
YOLO **cannot** use polygons directly. It needs **bounding boxes** in a specific numeric format.

You convert them with:

```bash
python convert_masks_to_yolo.py masks.json raspberry_images
```

This script:

1. Loads your `masks.json` (plant polygons)  
2. For each polygon, finds:
   - `min_x`, `max_x`, `min_y`, `max_y`  
   - Computes the smallest **upright rectangle** that contains the polygon
3. Computes:
   - `width  = max_x - min_x`  
   - `height = max_y - min_y`  
   - `center_x = (min_x + max_x) / 2`  
   - `center_y = (min_y + max_y) / 2`  
4. **Normalizes** these values by the image size so everything is between 0 and 1:
   - `x_center_norm = center_x / image_width`  
   - `y_center_norm = center_y / image_height`  
   - `width_norm    = width / image_width`  
   - `height_norm   = height / image_height`
5. Saves YOLO label files like:

   ```text
   plants_yolo_dataset/labels/image(1022).txt
   ```

   Each line is:

   ```text
   class_id x_center y_center width height
   ```

   For example, for 3 plants:

   ```text
   0 0.35 0.60 0.20 0.30   # Plant 1
   1 0.65 0.58 0.22 0.29   # Plant 2
   2 0.50 0.25 0.18 0.20   # Plant 3
   ```

> 🧠 **Why center coordinates?**  
> YOLO divides the image into a grid and predicts objects by their **center point** and size.  
> Using center x/y, width, and height makes prediction faster and simpler for the network.

After this step you’ll have:

```text
plants_yolo_dataset/
    images/        # copies of your images
    labels_yolo/   # YOLO-format bounding boxes
```

---

# 5️⃣ Split Into Training + Validation Sets (Critical ML Concept)

Run:

```bash
python file_structure.py
```

This script is doing **real ML data preparation** — not just file shuffling.

It creates:

```text
plants_yolo_dataset/images/train/
plants_yolo_dataset/images/val/
plants_yolo_dataset/labels/train/
plants_yolo_dataset/labels/val/
```

### Why do we split the data?

In supervised learning we must **train** and **evaluate** on different images.

- **Training set**:  
  YOLO learns patterns from these images:
  - shapes, textures, brightness, positions of the plants  
  - which region belongs to Plant 1 vs Plant 2 vs Plant 3  

- **Validation set**:  
  Used **only for testing** during training:
  - the model does **not** learn from these  
  - helps measure performance on **unseen** images  
  - prevents **overfitting** (memorizing instead of learning)

The script typically splits your data like:

- ~80% of images → `train`  
- ~20% of images → `val`

YOLO uses the validation set to compute:

- **Precision** (of the boxes I predicted, how many were correct?)  
- **Recall** (of all real plant boxes, how many did I find?)  
- **mAP@50** (overall detection quality)  
- and generates `val_batch0_pred.jpg` — a picture of its predictions on your val images.

> 📌 IB Connection:  
> - A4.2 Data preprocessing  
> - A4.3 Supervised learning & evaluation  
> - A4.5 Overfitting & generalization  

---

# 6️⃣ Create Your Own `plant_data.yaml`

We **do not commit** `plant_data.yaml` to GitHub, because each student’s folder paths are different.

> In the repo’s `.gitignore` we add:
> ```text
> plant_data.yaml
> ```

### You must create your own config file:

From inside your folder:

```bash
nano plant_data.yaml
```

Paste this (update `<yourname>_plants`):

```yaml
# plant_data.yaml - YOUR YOLO dataset configuration

train: /home/compsci/Desktop/<yourname>_plants/plants_yolo_dataset/images/train
val: /home/compsci/Desktop/<yourname>_plants/plants_yolo_dataset/images/val

nc: 3
names: ["Plant 1", "Plant 2", "Plant 3"]
```

Save & exit:

- `CTRL + O` → Enter  
- `CTRL + X` → Exit

### What is YAML?

YAML = **Y**et **A**nother **M**arkup **L**anguage.

It’s a simple text format used for **configuration**. YOLO reads:

- where your images are (`train`, `val`)  
- how many classes (`nc`)  
- what each class is called (`names`)

> IB link: this is part of understanding **how ML systems are configured** and how data and labels are structured.

---

# 7️⃣ Train Your YOLO Model (Supervised Learning in Action)

Now run training:

```bash
yolo detect train model=yolov10m.pt data=plant_data.yaml epochs=100 imgsz=640 batch=16 device=0
```

### What each part means:

| Piece | Meaning |
|-------|---------|
| `yolo detect train` | Use YOLO to train an **object detection** model |
| `model=yolov8n.pt`  | Start from pre-trained YOLOv8 nano (small, fast model) |
| `data=plant_data.yaml` | Use your dataset & class definitions |
| `epochs=30` | Train for 30 passes over the **training set** |
| `imgsz=640` | Resize all images to 640×640 for training |
| `device=0` | Use GPU #0 (the RTX 4080) |

### What is an epoch?

One **epoch** = YOLO has seen **all training images once**.

- Too few epochs → **underfitting**  
  - hasn’t learned enough  
- Too many epochs → **overfitting**  
  - memorizes training images instead of learning patterns  

30 epochs is reasonable for this project.

During training you will see loss and metric values change — this is the model improving.

> IB links:  
> - A4.3 Supervised learning loop  
> - A4.5 Training vs overfitting/underfitting  
> - A4.7 Hyperparameters & tuning  

---

# 8️⃣ What YOLO Produces

After training, look in:

```text
runs/detect/train/
```

Key files:

- `weights/best.pt` → 🔑 **Your trained model**  
- `weights/last.pt` → Model at final epoch  
- `val_batch0_pred.jpg` → 🔍 YOLO’s predictions on validation images  
- `results.csv`, `results.png` → metrics over epochs  

You **must** keep `best.pt` and `val_batch0_pred.jpg` — they’re part of your grade.

---

# 9️⃣ Test Your Model on Your Own Images

To carefully inspect predictions:

```bash
python debug_model.py runs/detect/train/weights/best.pt raspberry_images
```

This prints out all detections with:

- class id (0, 1, 2 = Plant 1, 2, 3)  
- confidence  
- bounding box coordinates  

You can also have YOLO annotate images:

```bash
yolo detect predict \
  model=runs/detect/train/weights/best.pt \
  source=raspberry_images \
  conf=0.10
```

- `conf=0.10` lowers the confidence threshold, so even weak detections appear  
- If Plant 2 or Plant 3 is missing at `0.25`, try `0.10`

> This connects to IB’s ideas about **precision, recall, and thresholds**.

---

# 🔟 Required Deliverables (What You Submit)

Each student submits:

### 1. Model + Prediction Artifacts

- `runs/detect/train/weights/best.pt`  
  → your trained YOLO model  
- `runs/detect/train/val_batch0_pred.jpg`  
  → validation prediction grid  
- At least **one screenshot** of YOLO predictions on your own `raspberry_images/` (can be from `predict/` output)

### 2. Short Written Reflection (about ½ page)

You should be able to explain:

- Why this is **supervised learning**  
- Why your project uses **classification + localization**  
- How polygons became bounding boxes  
- Why we normalize `[x_center, y_center, width, height]`  
- Why we split into **training** & **validation** sets  
- What **epochs** are and how they affected your results  
- What **val_batch0_pred.jpg** shows about your model  
- Where your model struggled (e.g., occlusion, similar-looking plants)  
- What you would do to improve it (more data, better lighting, more labeling, hyperparameters)

IB-style discussion: talk about **generalization, bias, overfitting, data quality**.

---

# 1️⃣1️⃣ ML Concepts You Must Be Able to Explain (IB Style)

These are things IB can ask you about, and you should be able to answer using this project as your example.

### 🧠 Supervised Learning

- Input: plant images  
- Output: plant class + bounding box  
- Labels: you created them via masks → bounding boxes  
- The algorithm adjusts its internal weights based on error  

### 🧠 Data Preprocessing

- Splitting into train/val  
- Converting polygons to YOLO bounding boxes  
- Normalizing coordinates  
- Ensuring labels and images match 1:1  

### 🧠 Overfitting vs Generalization

- Overfitting: model “memorizes” training images  
- Generalization: model works on **new** images  
- Validation set is used to estimate this  

### 🧠 Evaluation

Through YOLO’s training logs & `val_batch0_pred.jpg`, you see:

- Box precision/recall  
- Per-class performance (Plant 1, Plant 2, Plant 3)  
- Confidence scores  

### 🧠 Hyperparameters

You controlled:

- `epochs`  
- `imgsz`  
- (optionally) confidence threshold at prediction time  

You should be able to describe:

- How these affected the model  
- What you would change if your model was underperforming  

---

# 1️⃣2️⃣ Troubleshooting Tips

### ❌ YOLO says: “no labels found”
Check:

- Did `convert_masks_to_yolo.py` run without errors?  
- Are there `.txt` files in `plants_yolo_dataset/labels_yolo/`?  
- Are the filenames exactly matching the images?

### ❌ Training runs but nothing seems to learn

- Check that you actually have:
  - at least a few dozen labeled images  
  - all three plants labeled consistently  
- Check that `nc: 3` and `names` has 3 classes  

### ❌ Only Plant 1 is ever detected

- Lower `conf` during prediction: `conf=0.10`  
- Make sure Plant 2 and Plant 3 are actually labeled in multiple images  
- Look at `val_batch0_pred.jpg` to see what YOLO is “thinking”

---

You are now working with the same kinds of ML tools used in:

- Autonomous vehicles  
- Agricultural monitoring systems  
- Industrial robotics  
- Medical imaging  

…and doing it on **your own real-world plant data**. That’s exactly the kind of authentic, higher-level computing IB is aiming for.
