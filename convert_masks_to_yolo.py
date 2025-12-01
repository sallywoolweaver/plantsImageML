#!/usr/bin/env python3
import json, os
from PIL import Image

#!/usr/bin/env python3
import json, os
from PIL import Image

def convert(masks_json, out_labels="labels_yolo"):

    with open(masks_json,"r") as f:
        data=json.load(f)

    os.makedirs(out_labels,exist_ok=True)

    # If old format is detected (single reference_image)
    if "reference_image" in data and "polygons" in data:
        data = {"images":[data]} # convert to new expected structure

    for entry in data["images"]:

        img_path = entry["reference_image"]
        polygons = entry["polygons"]

        img = Image.open(img_path)
        W,H = img.size

        base = os.path.splitext(os.path.basename(img_path))[0]
        label_file = os.path.join(out_labels,base+".txt")

        print(f"[Writing] {label_file}")

        with open(label_file,"w") as f:
            for poly in polygons:
                cls = int(poly["plant_id"])
                xs=[p[0] for p in poly["points"]]
                ys=[p[1] for p in poly["points"]]

                x_min,x_max=min(xs),max(xs)
                y_min,y_max=min(ys),max(ys)

                xc=((x_min+x_max)/2)/W
                yc=((y_min+y_max)/2)/H
                w=(x_max-x_min)/W
                h=(y_max-y_min)/H

                f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    print("\n✔ All YOLO label files generated.")



if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("masks_json", help="masks.json from annotation tool")
    ap.add_argument("--out", default="yolo_labels", help="directory for .txt output")
    args = ap.parse_args()

    convert(args.masks_json, args.out)
