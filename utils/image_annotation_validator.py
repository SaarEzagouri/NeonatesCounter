import cv2
import os

# Define class names (update based on your dataset)
class_names = ["Neonate"]

# Folder paths
img_dir = "C:/Projects/FreeZem/NeonatesCounter/dataset/val/images" #"C:/Projects/FreeZem/NeonatesCounter/dataset/train/images"
label_dir = "C:/Projects/FreeZem/NeonatesCounter/dataset/val/labels" #"C:/Projects/FreeZem/NeonatesCounter/dataset/train/labels"

# Loop through images and plot annotations
for img_file in os.listdir(img_dir):
    if img_file.endswith(".jpg"):
        img_path = os.path.join(img_dir, img_file)
        label_path = os.path.join(label_dir, img_file.replace(".jpg", ".txt"))
        
        # Load image
        img = cv2.imread(img_path)
        h, w, _ = img.shape  # Get image size
        
        # Read annotation file
        with open(label_path, "r") as file:
            lines = file.readlines()
        
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])  # Object class
            x_center, y_center, box_w, box_h = map(float, parts[1:])

            # Convert YOLO format to pixel coordinates
            x1 = int((x_center - box_w / 2) * w)
            y1 = int((y_center - box_h / 2) * h)
            x2 = int((x_center + box_w / 2) * w)
            y2 = int((y_center + box_h / 2) * h)

            font_scale = 0.45

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, class_names[class_id], (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)

        # Show image
        cv2.imshow("Annotated Image", img)
        cv2.waitKey(0)  # Press any key to continue

cv2.destroyAllWindows()
