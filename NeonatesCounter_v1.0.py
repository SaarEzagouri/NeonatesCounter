''''' 
Difference between NeonatesCounter v3 and v4: v4 does not output any JSON files or log files; it only outputs the images with bounding boxes. Also, it returns the number of objects detected in each image as a vector.

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

    def save_image(self, image, output_folder, image_name):
        """Saves the final image with bounding boxes."""
        output_image_path = os.path.join(output_folder, f"{image_name}_bboxes.jpg")
        cv2.imwrite(output_image_path, image)

class ImageProcessor:
    def __init__(self, model_path, images_path):
        self.model_path = model_path
        self.images_path = images_path

        self.patcher = ImagePatcher()
        self.model = YOLOModel(model_path)
        self.repatcher = ImageRepatcher()
        self.saver = ResultSaver(images_path)

    def process_images(self):
        """Processes images, detects objects, and stitches them back together."""
        start_time = time.time()
        results = []
        Path(self.images_path).mkdir(parents=True, exist_ok=True)

        for img_path in Path(self.images_path).glob("*.jpg"):
            if "Overlay" in img_path.name or img_path.stem.endswith("_bboxes"):
                continue

            image = cv2.imread(str(img_path))
            h, w, _ = image.shape

            # Pad the image to ensure it's divisible by stride
            padded_image = self.patcher.pad_image(image)
            padded_h, padded_w, _ = padded_image.shape

            # Create patches from the padded image
            patches = self.patcher.create_patches(padded_image)
            patch_preds = np.zeros_like(patches)

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
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                detections.append((x1 + j * self.patcher.stride, y1 + i * self.patcher.stride, x2 + j * self.patcher.stride, y2 + i * self.patcher.stride))
                                num_detections += 1

            results.append(num_detections)

            # Reconstruct the full image from patches
            reconstructed_image = self.repatcher.repatch_image(patches, (padded_h, padded_w, 3))

            # Crop back to original image size
            reconstructed_image = reconstructed_image[:h, :w]

            # Draw bounding boxes on the reconstructed image with transparency
            overlay = reconstructed_image.copy()
            for x1, y1, x2, y2 in detections:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.5, reconstructed_image, 0.5, 0, reconstructed_image)

            # Save the final image with bounding boxes
            self.saver.save_image(reconstructed_image, self.images_path, img_path.stem)
        
        return results

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
