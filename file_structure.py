#!/usr/bin/env python3
import json
import os
from pathlib import Path
from PIL import Image
import shutil

def polygon_to_bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)

def convert_masks_to_yolo(masks_json, images_root, out_root="plants_yolo_dataset", train_ratio=0.8):
    masks_json = Path(masks_json)
    images_root = Path(images_root)
    out_root = Path(out_root)

    with open(masks_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    images_data = data.get("images", [])
    if not images_data:
        print("No 'images' entries found in masks.json")
        return

    # Make YOLO folders
    img_train = out_root / "images" / "train"
    img_val   = out_root / "images" / "val"
    lbl_train = out_root / "labels" / "train"
    lbl_val   = out_root / "labels" / "val"
    for p in [img_train, img_val, lbl_train, lbl_val]:
        p.mkdir(parents=True, exist_ok=True)

    total = len(images_data)
    train_cut = int(total * train_ratio)
    print(f"Total images: {total}. Train: {train_cut}, Val: {total - train_cut}")

    for idx, entry in enumerate(images_data):
        ref_path_str = entry.get("reference_image")
        polygons = entry.get("polygons", [])
        if not ref_path_str or not polygons:
            continue

        # Get just the filename (ignore Windows path)
        filename = os.path.basename(ref_path_str)
        img_path = images_root / filename

        if not img_path.exists():
            print(f"[WARN] Missing image: {img_path}")
            continue

        # Decide split
        if idx < train_cut:
            img_out_dir = img_train
            lbl_out_dir = lbl_train
            split_name = "train"
        else:
            img_out_dir = img_val
            lbl_out_dir = lbl_val
            split_name = "val"

        # Open image to get size
        with Image.open(img_path) as im:
            W, H = im.size

        stem = img_path.stem
        label_path = lbl_out_dir / f"{stem}.txt"

        lines = []
        for poly in polygons:
            plant_id = poly.get("plant_id")
            if plant_id is None:
                continue

            # Class IDs: Plant 1→0, Plant 2→1, Plant 3→2
            cls = int(plant_id) - 1

            pts = poly["points"]
            x_min, y_min, x_max, y_max = polygon_to_bbox(pts)

            xc = ((x_min + x_max) / 2.0) / W
            yc = ((y_min + y_max) / 2.0) / H
            w  = (x_max - x_min) / W
            h  = (y_max - y_min) / H

            # Clamp
            xc = max(0.0, min(1.0, xc))
            yc = max(0.0, min(1.0, yc))
            w  = max(0.0, min(1.0, w))
            h  = max(0.0, min(1.0, h))

            lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        if not lines:
            continue

        # Copy the image into YOLO structure
        shutil.copy2(img_path, img_out_dir / filename)

        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        print(f"[OK] {img_path} -> {label_path} ({split_name})")

    print("\n✔ Finished building plants_yolo_dataset")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("masks_json", help="masks.json from define_masks_v4")
    ap.add_argument("images_root", help="folder where the images actually live")
    ap.add_argument("--out_root", default="plants_yolo_dataset")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    args = ap.parse_args()

    convert_masks_to_yolo(args.masks_json, args.images_root, args.out_root, args.train_ratio)
