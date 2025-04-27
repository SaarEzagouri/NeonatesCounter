"""""
Neonates Counter v1.1
- This version is suitable for calling this py file from within MATLAB
- If more than 10 images are found, only every second image is processed.
- The last image is always processed.
    
"""""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from patchify import patchify
from openpyxl import load_workbook
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
        return cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))

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
        self.file_path = self.find_xlsx_file()  # Automatically find the .xlsx file

    def find_xlsx_file(self):
        """Finds the first .xlsx file in the given directory."""
        for file in os.listdir(self.folder_path):
            if file.endswith(".xlsx"):
                return os.path.join(self.folder_path, file)
        raise FileNotFoundError("No .xlsx file found in the folder.")

    def update_xlsx(self, filenames, detections_per_image):
        """Adds image filenames and detection counts as new columns to the Excel file, plus a 'mean' row."""
        df = pd.read_excel(self.file_path, engine='openpyxl')
    
        # Make sure you append the filenames and detections before checking row count
        image_filenames = filenames + ['mean']
        detection_values = [detections_per_image[name] for name in filenames]
        mean_detection = int(np.round(np.mean(detection_values)))  # You can also use float if you prefer decimals
        detection_counts = detection_values + [mean_detection]
    
        # Add or extend columns
        df = df.reindex(range(len(image_filenames)))  # Expand DataFrame if necessary
        df["image_filename"] = image_filenames
        df["detections_count"] = detection_counts
    
        # Save updated Excel file
        df.to_excel(self.file_path, index=False, engine='openpyxl')

class ImageProcessor:
    def __init__(self, model_path, test_images_path):
        self.model_path = model_path
        self.test_images_path = test_images_path

        self.patcher = ImagePatcher()
        self.model = YOLOModel(model_path)
        self.repatcher = ImageRepatcher()
        self.saver = ResultSaver(test_images_path)
        self.xlsx_updater = XLSXUpdater(test_images_path)  # Use XLSXUpdater instead of CSVUpdater

    def process_images(self):
        """Processes images, detects objects, updates the Excel file, and saves the images."""
        detections_per_image = {}
        filenames = []  # Add this line to initialize the filenames list
        
        # Get all the image paths
        image_paths = list(Path(self.test_images_path).glob("*.jpg"))
    
        # If there are 10 or fewer images, process all of them
        if len(image_paths) <= 10:
            selected_images = image_paths
        else:
            # If there are more than 10 images, process odd-indexed images and the last image
            selected_images = [img for idx, img in enumerate(image_paths) if idx % 2 == 0]  # Odd-indexed images (1, 3, 5, etc.)
            # Ensure the last image is included, but only if it's not already included in the odd-indexed selection
            if image_paths[-1] not in selected_images:
                selected_images.append(image_paths[-1])  # Add the last image if not already included
    
        for img_path in selected_images:
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
    
            # Draw bounding boxes
            for x1, y1, x2, y2 in detections:
                cv2.rectangle(reconstructed_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
            # Save the final image
            self.saver.save_image(reconstructed_image, image_name)
    
        # Update the Excel file
        self.xlsx_updater.update_xlsx(filenames, detections_per_image)