import json
import os
import cv2

def convert_via_to_yolo(via_json_path, images_dir, output_labels_dir):
    os.makedirs(output_labels_dir, exist_ok=True)

    with open(via_json_path, 'r') as f:
        via_data = json.load(f)

    for key, item in via_data.items():
        filename = item['filename']
        regions = item.get('regions', [])

        if not regions:
            continue  # Skip images with no annotations

        image_path = os.path.join(images_dir, filename)
        if not os.path.exists(image_path):
            print(f"Image '{filename}' not found in '{images_dir}'. Skipping.")
            continue

        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to read '{image_path}'. Skipping.")
            continue
        h, w = img.shape[:2]

        lines = []
        for region in regions:
            shape = region.get("shape_attributes", {})
            region_attrs = region.get("region_attributes", {})

            if shape.get("name") != "rect":
                continue  # Skip non-rect shapes

            x = shape["x"]
            y = shape["y"]
            width = shape["width"]
            height = shape["height"]

            # YOLO format: class_id cx cy w h (all normalized)
            cx = (x + width / 2) / w
            cy = (y + height / 2) / h
            norm_w = width / w
            norm_h = height / h

            try:
                class_id = int(region_attrs.get("type", 0))
            except ValueError:
                class_id = 0

            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {norm_w:.6f} {norm_h:.6f}")

        if lines:
            out_filename = os.path.splitext(filename)[0] + ".txt"
            out_path = os.path.join(output_labels_dir, out_filename)
            with open(out_path, 'w') as f_out:
                f_out.write("\n".join(lines))

    print(f"\n✅ Conversion complete.\nYOLO labels saved in: '{output_labels_dir}'")


convert_via_to_yolo(
    via_json_path='C:/Projects/FreeZem/NeonatesCounter/Production/Eval_Annotations/test2_VGG_format/neonates_counter_json_merged.json',
    images_dir='C:/Projects/FreeZem/NeonatesCounter/dataset/final_optimization/all_images',
    output_labels_dir='C:/Projects/FreeZem/NeonatesCounter/dataset/final_optimization/all_images',  # e.g., 'labels/train'
)
