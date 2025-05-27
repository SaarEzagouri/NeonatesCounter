"""""
Neonates Counter v1.2

This script processes images in folders created on the current day, performs object detection using a YOLO model,
and saves the detection results. It performs the following steps:

1. Pads and splits each image into patches to feed into a YOLO model for object detection.
2. Reconstructs the image with bounding boxes drawn on the detected objects.
3. Saves the modified image with bounding boxes in a specified output folder.
4. Updates an Excel file with the detection count for each processed image.

The script ensures that no image containing "Overlay" or ending with "_bboxes" in the filename is processed.

New features in v1.2:

- Conditional analysis of images based on their modification date - run only on images created (not modified) today.
- Bboxes are drawn with a thin line and semi-transparent overlay.
- Analyze all images in each folder except of "Overlay.jpg".
- Update the XLSX file with the number of detected objects for each image and the mean.
- Error handling: If excel / images not found, display message and continue.

"""""

# ------------- Libraries ------------- #

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from patchify import patchify
from openpyxl import load_workbook
from ultralytics import YOLO
from datetime import datetime

# ------------- Utilities ------------- #

class ImagePatcher:
    def __init__(self, patch_size=416, stride=376):
        self.patch_size = patch_size
        self.stride = stride

    def pad_image(self, image):
        """Pads the image so that it can be evenly divided into patches."""
        h, w, _ = image.shape
        pad_h = (self.stride - (h % self.stride)) if (h % self.stride) != 0 else 0
        pad_w = (self.stride - (w % self.stride)) if (w % self.stride) != 0 else 0
        return cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    def create_patches(self, image):
        """Splits an image into overlapping patches."""
        return patchify(image, (self.patch_size, self.patch_size, 3), step=self.stride)

class YOLOModel:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def predict(self, patch):
        """Runs YOLO on a patch."""
        return self.model.predict(source=patch, conf=0.65, iou=0.4, imgsz=416, save=False, line_width=1, verbose=False)

class ImageRepatcher:
    def __init__(self, patch_size=416, stride=376):
        self.patch_size = patch_size
        self.stride = stride

    def repatch_image(self, patches, image_shape):
        """Reconstructs an image from patches using weighted averaging in overlapping regions."""
        h, w, c = image_shape
        reconstructed = np.zeros((h, w, c), dtype=np.float32)
        count_map = np.zeros((h, w, c), dtype=np.float32)

        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                patch = patches[i, j, 0]
                y_start, x_start = i * self.stride, j * self.stride
                y_end, x_end = y_start + self.patch_size, x_start + self.patch_size

                reconstructed[y_start:y_end, x_start:x_end] += patch
                count_map[y_start:y_end, x_start:x_end] += 1

        count_map[count_map == 0] = 1  # Avoid division by zero
        return (reconstructed / count_map).astype(np.uint8)

class ResultSaver:
    def __init__(self, output_path):
        self.output_path = output_path  # Same as test_images_path

    def save_image(self, image, image_name):
        """Saves the final image in the same directory as the original."""
        output_image_path = os.path.join(self.output_path, f"{image_name}_bboxes.jpg")
        cv2.imwrite(output_image_path, image)

class XLSXUpdater:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.file_path = self.find_xlsx_file()

    def find_xlsx_file(self):
        """Finds the first .xlsx file in the given directory."""
        for file in os.listdir(self.folder_path):
            if file.endswith(".xlsx"):
                return os.path.join(self.folder_path, file)
        return None  # Don't raise error

    def update_xlsx(self, filenames, detections_per_image):
        """Adds image filenames and detection counts as new columns to the Excel file, plus a 'mean' row."""
        if not self.file_path:
            print("No Excel file found, skipping Excel update.")
            return  # Skip if no Excel file

        df = pd.read_excel(self.file_path, engine='openpyxl')
        image_filenames = filenames + ['mean']
        detection_values = [detections_per_image[name] for name in filenames]
        mean_detection = int(np.round(np.mean(detection_values)))
        detection_counts = detection_values + [mean_detection]

        df = df.reindex(range(len(image_filenames)))  # Expand DataFrame if needed
        df["image_filename"] = image_filenames
        df["detections_count"] = detection_counts

        df.to_excel(self.file_path, index=False, engine='openpyxl')

# ------------- Main Processor ------------- #

class ImageProcessor:
    def __init__(self, model_path, root_folder):
        self.model_path = model_path
        self.root_folder = root_folder

        self.patcher = ImagePatcher()
        self.model = YOLOModel(model_path)
        self.repatcher = ImageRepatcher()
        self.saver = ResultSaver(root_folder)
        self.xlsx_updater = XLSXUpdater(root_folder)

    def process_images(self):
        detections_per_image = {}
        filenames = []

        image_paths = list(self.root_folder.glob("*.jpg"))
        image_paths = [img_path for img_path in image_paths if "Overlay" not in img_path.stem and not img_path.stem.endswith("_bboxes")]

        if not image_paths:
            print(f"No valid images found in {self.root_folder}, skipping.")
            return  # Nothing to do

        for img_path in image_paths:
            if "Overlay" in img_path.stem or img_path.stem.endswith("_bboxes"):
                continue
            
            image = cv2.imread(str(img_path))
            h, w, _ = image.shape

            # Pad the image
            padded_image = self.patcher.pad_image(image)
            padded_h, padded_w, _ = padded_image.shape

            # Create patches
            patches = self.patcher.create_patches(padded_image)

            detections = []
            for i in range(patches.shape[0]):
                for j in range(patches.shape[1]):
                    patch = patches[i, j, 0]

                    # Run YOLO on patch
                    results = self.model.predict(patch)

                    # Collect detections
                    if results:
                        for result in results:
                            for box in result.boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                detections.append((
                                    x1 + j * self.patcher.stride,
                                    y1 + i * self.patcher.stride,
                                    x2 + j * self.patcher.stride,
                                    y2 + i * self.patcher.stride
                                ))

            # Store detections count per image
            image_name = img_path.stem
            detections_count = len(detections)
            detections_per_image[image_name] = detections_count
            filenames.append(image_name)  # Append the image filename to the list

            # Print result
            print(f"{image_name}: {detections_count} objects detected")

            # Reconstruct image
            reconstructed_image = self.repatcher.repatch_image(patches, (padded_h, padded_w, 3))
            reconstructed_image = reconstructed_image[:h, :w]  # Crop to original size

            # Draw bounding boxes with transparency
            for x1, y1, x2, y2 in detections:
                # Create a semi-transparent overlay
                overlay = reconstructed_image.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 1)
                
                # Blend the overlay with the original image using the alpha value
                cv2.addWeighted(overlay, 0.5, reconstructed_image, 1 - 0.5, 0, reconstructed_image)

            # Save the final image
            self.saver.save_image(reconstructed_image, image_name)

        # Update the Excel file
        self.xlsx_updater.update_xlsx(filenames, detections_per_image)

# ------------- Run ------------- #

def main():
    script_path = Path(__file__).parent
    model_path = script_path / "NeonatesCounter_v1.0_model.pt"
    img_dir = Path("C:/Users/FreezeM Hermetia/FreezeM Dropbox/FreezeM R&D/Neonate Calibration Hermetia")
    
    # Get the directories created today
    today = pd.Timestamp.today().strftime('%Y-%m-%d')
    today_folders = []

    for folder in img_dir.iterdir():
        if folder.is_dir():
            # Get the creation time (on Unix-like systems, this is the birth time)
            creation_time = datetime.fromtimestamp(folder.stat().st_ctime).strftime('%Y-%m-%d')
            if creation_time == today:
                today_folders.append(folder)
    
    # Run the image processing for each folder created today
    for folder in today_folders:
        print(f"Processing folder: {folder}")
       
        # Create an ImageProcessor instance for each folder
        processor = ImageProcessor(model_path=model_path, root_folder=folder)
        processor.process_images()

    print("Processing Completed")

if __name__ == "__main__":
    main()
