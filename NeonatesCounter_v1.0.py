'''''
NeonatesCounter version 1.0, all rights reserved to FreezeM.

This script detects neonates in images using a custom-trained YOLOv8m model. It employs Patchify to divide the image into overlapping patches, runs the YOLO model on each patch, and reconstructs the image with detected objects overlaid. The final output includes an annotated image with bounding boxes and a log file documenting the detections.  

The patch size and stride, set in the `ImagePatcher` class, are fixed at 416×416 with a 376-pixel stride, allowing control over the overlap between patches.
'''''

#Imports:
import os
import cv2
import numpy as np
import shutil
import time
import json
from pathlib import Path
from datetime import datetime
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
        self.output_path = output_path

    def save_log(self, log_path, num_objects, runtime):
        """Saves detection log to a file."""
        log_data = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "objects_detected": num_objects,
            "runtime_seconds": f"{runtime:.3f}"
        }
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=4)

    def save_image(self, image, output_folder, image_name):
        """Saves the final image with bounding boxes."""
        output_image_path = os.path.join(output_folder, f"{image_name}_reconstructed.jpg")
        cv2.imwrite(output_image_path, image)

class ImageProcessor:
    def __init__(self, model_path, test_images_path, output_path):
        self.model_path = model_path
        self.test_images_path = test_images_path
        self.output_path = output_path

        self.patcher = ImagePatcher()
        self.model = YOLOModel(model_path)
        self.repatcher = ImageRepatcher()
        self.saver = ResultSaver(output_path)

    def process_images(self):
        """Processes images, detects objects, and stitches them back together."""
        start_time = time.time()
        total_detections = 0
        Path(self.output_path).mkdir(parents=True, exist_ok=True)

        for img_path in Path(self.test_images_path).glob("*.jpg"):
            image = cv2.imread(str(img_path))
            h, w, _ = image.shape

            # Pad the image to ensure it's divisible by stride
            padded_image = self.patcher.pad_image(image)
            padded_h, padded_w, _ = padded_image.shape

            # Create patches from the padded image
            patches = self.patcher.create_patches(padded_image)
            patch_preds = np.zeros_like(patches)

            detections = []
            for i in range(patches.shape[0]):
                for j in range(patches.shape[1]):
                    patch = patches[i, j, 0]

                    # Run YOLO on patch
                    results = self.model.predict(patch)

                    # Check if any predictions are found
                    if results:
                        for result in results:
                            for box in result.boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                # Adjust coordinates to match the full image
                                detections.append((x1 + j * self.patcher.stride, y1 + i * self.patcher.stride, x2 + j * self.patcher.stride, y2 + i * self.patcher.stride))
                                total_detections += 1

            # Reconstruct the full image from patches
            reconstructed_image = self.repatcher.repatch_image(patches, (padded_h, padded_w, 3))

            # Crop back to original image size
            reconstructed_image = reconstructed_image[:h, :w]

            # Create the folder for saving predictions specific to this image
            image_name = img_path.stem
            output_folder = os.path.join(self.output_path, f"{image_name}_predictions")

            # Check if folder exists, and handle overwriting
            if os.path.exists(output_folder):
                print(f"Warning: The folder '{output_folder}' already exists. It will be overwritten.")
                shutil.rmtree(output_folder)
            
            Path(output_folder).mkdir(parents=True, exist_ok=True)

            # Draw bounding boxes on the reconstructed image
            for x1, y1, x2, y2 in detections:
                cv2.rectangle(reconstructed_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Save the final image with bounding boxes
            self.saver.save_image(reconstructed_image, output_folder, image_name)

        runtime = time.time() - start_time
        self.saver.save_log(os.path.join(self.output_path, f"{image_name}_results.txt"), total_detections, runtime)
        print(f"Processing complete. {total_detections} objects detected in {runtime:.2f} seconds.")

# Instantiate and run
if __name__ == "__main__":
    processor = ImageProcessor(
        model_path="best_model.pt",
        test_images_path="path/to/images",
        output_path="path/to/output"
    )
    processor.process_images()
