''''' 
Difference between NeonatesCounter v3 and v4: v4 does not output any JSON files or log files; it only outputs the images with bounding boxes. Also, it returns the number of objects detected in each image as a dictionary.

NeonatesCounter version 1.0, all rights reserved to FreezeM.

This script is used to detect neonates in images using YOLOv8m and stitch the detected objects back together.
Patchify is used to split the image into overlapping patches, and the YOLO model is run on each patch.
The detected objects are then overlaid on the reconstructed image.
The final image is saved with bounding boxes around the detected objects.

Patch size and stride can be adjusted to control the overlap between patches. It is hard-coded to 416x416 patches with a stride of 376 in ImagePatcher class.
''''' 

#Imports:
import os
import cv2
import numpy as np
import time
from pathlib import Path
from patchify import patchify
from ultralytics import YOLO

class ImagePatcher:
    def __init__(self, patch_size=416, stride=376):
        self.patch_size = patch_size
        self.stride = stride

    def pad_image(self, image):
        """Pads the image so that it can be evenly divided into patches."""
        h, w, _ = image.shape
        pad_h = (self.stride - (h % self.stride)) if (h % self.stride) != 0 else 0
        pad_w = (self.stride - (w % self.stride)) if (w % self.stride) != 0 else 0
        padded_image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return padded_image

    def create_patches(self, image):
        """Splits an image into overlapping patches."""
        return patchify(image, (self.patch_size, self.patch_size, 3), step=self.stride)

class YOLOModel:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def predict(self, patch):
        """Runs YOLO on a patch."""
        return self.model.predict(source=patch, conf=0.045, iou=0.4, imgsz=416, save=False, line_width=1, verbose=False)

class ImageRepatcher:
    def __init__(self, patch_size=416, stride=376):
        self.patch_size = patch_size
        self.stride = stride

class ImageProcessor:
    def __init__(self, model_path, images_path):
        self.model_path = model_path
        self.images_path = images_path

        self.patcher = ImagePatcher()
        self.model = YOLOModel(model_path)
        self.repatcher = ImageRepatcher()

    def process_images(self):
        """Processes images, detects objects, and returns results as a dictionary."""
        results_dict = {}
        image_files = sorted(Path(self.images_path).glob("*.jpg"))  # Ensure consistent order

        for img_path in image_files:
            if "Overlay" in img_path.name or img_path.stem.endswith("_bboxes"):
                continue

            image = cv2.imread(str(img_path))
            h, w, _ = image.shape

            # Pad the image to ensure it's divisible by stride
            padded_image = self.patcher.pad_image(image)
            padded_h, padded_w, _ = padded_image.shape

            # Create patches from the padded image
            patches = self.patcher.create_patches(padded_image)

            detections = []
            num_detections = 0
            for i in range(patches.shape[0]):
                for j in range(patches.shape[1]):
                    patch = patches[i, j, 0]

                    # Run YOLO on patch
                    results_yolo = self.model.predict(patch)

                    # Check if any predictions are found
                    if results_yolo:
                        for result in results_yolo:
                            for box in result.boxes:
                                num_detections += 1

            results_dict[img_path.name] = num_detections
        
        return results_dict  # Return results as a dictionary

# Find the newest folder inside Nachshonim Neonates Calibration
data_root = Path("C:/Projects/FreeZem/NeonatesCounter/dataset/") #("https://www.dropbox.com/home/FreezeM%20R%26D/Nachshonim%20Neonates%20Calibration")
latest_folder = str(max(data_root.iterdir(), key=lambda d: d.stat().st_mtime))

# Instantiate and run
if __name__ == "__main__":
    processor = ImageProcessor(
        model_path = "NeonatesCounter_v1.0_model.pt",#"https://www.dropbox.com/home/FreezeM%20R%26D/Engineering%20RnD/Projects/Calibtaion%20Procedure/SaarDev/NeonatesCounter/NeonatesCounter_v1.0/NeonatesCounter_v1.0_model.pt",
        images_path = latest_folder
    )
    results = processor.process_images()
