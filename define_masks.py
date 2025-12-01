#!/usr/bin/env python3
# define_masks_v3.py
#
# Usage:
#   Single image:
#     python3 define_masks_v3.py /path/to/reference.jpg masks.json --plant_ids 1 2 3
#
#   Folder of images:
#     python3 define_masks_v3.py /path/to/folder masks.json --plant_ids 1 2 3
#
# Controls (same as before):
#   - Left click:   add vertex
#   - Backspace:    undo last vertex
#   - Esc:          reset current polygon
#   - Enter:        finish polygon for this plant & move to the next
#   - Q:            quit without saving current image (or whole session if at first image)

import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.image as mpimg
from PIL import Image, ExifTags
import matplotlib
matplotlib.use("TkAgg")

# ----------------- helpers for date/time -----------------

def get_image_datetime(path):
    """
    Try to get DateTimeOriginal from EXIF.
    If missing, fall back to file modified time.
    Return ISO-8601 string (YYYY-MM-DDTHH:MM:SS).
    """
    try:
        img = Image.open(path)
        exif = img.getexif()
        if exif:
            # Find EXIF tag for DateTimeOriginal
            tag_map = {ExifTags.TAGS.get(k, k): k for k in exif.keys()}
            if "DateTimeOriginal" in tag_map:
                dt_raw = exif.get(tag_map["DateTimeOriginal"])
                # EXIF format: "YYYY:MM:DD HH:MM:SS"
                if isinstance(dt_raw, str):
                    dt_raw = dt_raw.strip()
                    dt = datetime.strptime(dt_raw, "%Y:%m:%d %H:%M:%S")
                    return dt.isoformat()
    except Exception:
        pass  # fall back

    # Fall back to file modification time
    try:
        ts = os.path.getmtime(path)
        dt = datetime.fromtimestamp(ts)
        return dt.isoformat()
    except Exception:
        return None

# ----------------- helpers for JSON storage -----------------

def load_existing_masks(masks_json_path):
    """
    Load existing masks.json if it exists.
    Supports:
      - new format: {"images": [ ... ]}
      - old format: {"reference_image": ..., "polygons": [...]}
    Returns dictionary in unified form: {"images": [ ... ]}.
    """
    if not os.path.exists(masks_json_path):
        return {"images": []}

    with open(masks_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # New format already
    if isinstance(data, dict) and "images" in data and isinstance(data["images"], list):
        return data

    # Old single-image format → convert
    if isinstance(data, dict) and "reference_image" in data and "polygons" in data:
        img_entry = {
            "reference_image": data["reference_image"],
            # date was not stored before; leave as None or omit
            "date": data.get("date", None),
            "polygons": data["polygons"],
        }
        return {"images": [img_entry]}

    # Unknown structure → wrap it so we don't crash
    return {"images": []}

# ----------------- interactive polygon drawing -----------------

def draw_polygon_for_plant(ax, img, plant_id, fig):
    pts = []
    line, = ax.plot([], [], marker='o', linewidth=1, color='yellow')
    poly_patch = None
    ax.set_title(f"Draw polygon for plant_id={plant_id}. "
                 f"Enter=finish, Backspace=undo, Esc=reset, Q=quit")

    def redraw():
        nonlocal poly_patch
        xs, ys = zip(*pts) if pts else ([], [])
        line.set_data(xs, ys)
        if poly_patch:
            poly_patch.remove()
            poly_patch = None
        if len(pts) >= 3:
            poly_patch = Polygon(pts, closed=True, fill=False, edgecolor='y', linewidth=2)
            ax.add_patch(poly_patch)
        fig.canvas.draw_idle()

    def onclick(event):
        if event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        pts.append((event.xdata, event.ydata))
        redraw()

    finished = {"done": False, "cancel": False}

    def onkey(event):
        if event.key == 'enter':
            if len(pts) >= 3:
                finished["done"] = True
        elif event.key == 'backspace':
            if pts:
                pts.pop()
                redraw()
        elif event.key == 'escape':
            pts.clear()
            redraw()
        elif event.key in ('q', 'Q'):
            finished["cancel"] = True

    cid1 = fig.canvas.mpl_connect('button_press_event', onclick)
    cid2 = fig.canvas.mpl_connect('key_press_event', onkey)

    while not (finished["done"] or finished["cancel"]):
        plt.pause(0.05)

    fig.canvas.mpl_disconnect(cid1)
    fig.canvas.mpl_disconnect(cid2)

    if finished["cancel"]:
        return None
    return pts if len(pts) >= 3 else None

# ----------------- main logic -----------------

def list_images_in_folder(folder):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    files = []
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(exts):
            files.append(os.path.join(folder, name))
    return files


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("reference_path", help="Path to a single image OR a folder of images")
    ap.add_argument("masks_json", help="Output masks JSON (will append if exists)")
    ap.add_argument("--plant_ids", nargs='+', required=True, help="List of plant IDs, e.g. 1 2 3")
    args = ap.parse_args()

    # Determine if we got a file or a folder
    if os.path.isdir(args.reference_path):
        image_paths = list_images_in_folder(args.reference_path)
        if not image_paths:
            print("No images found in folder:", args.reference_path)
            raise SystemExit(1)
    else:
        image_paths = [args.reference_path]

    # Load existing masks.json (append mode)
    masks_data = load_existing_masks(args.masks_json)

    fig, ax = plt.subplots()
    plt.tight_layout()

    for img_idx, img_path in enumerate(image_paths, start=1):
        print(f"\n=== Image {img_idx}/{len(image_paths)}: {img_path} ===")

        img = mpimg.imread(img_path)
        ax.clear()
        ax.imshow(img)
        ax.set_axis_off()
        fig.canvas.draw_idle()

        polygons = []
        quit_entirely = False

        for pid in args.plant_ids:
            poly = draw_polygon_for_plant(ax, img, pid, fig)
            if poly is None:
                print(f"Skipped plant_id={pid} on this image (no polygon or user quit).")
                # If user hit Q on first plant, treat as "quit session"
                if not polygons:
                    quit_entirely = True
                    break
            else:
                polygons.append({"plant_id": str(pid), "points": poly})

        if quit_entirely:
            print("User requested quit. Stopping annotation.")
            break

        if polygons:
            date_str = get_image_datetime(img_path)
            entry = {
                "reference_image": img_path,
                "date": date_str,
                "polygons": polygons
            }
            masks_data["images"].append(entry)
            print(f"Added {len(polygons)} polygons for image: {img_path}")
        else:
            print(f"No polygons saved for image: {img_path}")

    # Save (append mode, preserving old + new)
    with open(args.masks_json, "w", encoding="utf-8") as f:
        json.dump(masks_data, f, indent=2)

    print(f"\nSaved updated masks to {args.masks_json}")
