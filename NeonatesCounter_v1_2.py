"""""
Neonates Counter v1.2

This script processes images in folders created on the current day, performs object detection using a YOLO model,
and saves the detection results. It performs the following steps:
1. Loads images from today's folders two levels up from the script's directory.
2. Pads and splits each image into patches to feed into a YOLO model for object detection.
3. Reconstructs the image with bounding boxes drawn on the detected objects.
4. Saves the modified image with bounding boxes in a specified output folder.
5. Updates an Excel file with the detection count for each processed image.

The script ensures that no image containing "Overlay" or ending with "_bboxes" in the filename is processed.

New features in v1.2:

- Conditional analysis of images based on their modification date - run only on images created today.
- Bboxes are drawn with a thin line and semi-transparent overlay.
- Analyze all images in each folder except of "Overlay.jpg".
- Update the XLSX file with the number of detected objects for each image and the mean.

"""""
# ------------- Libraries ------------- #

from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO
import openpyxl

# ------------- Utilities ------------- #

class ImagePatcher:
    def __init__(self, patch_size=640, overlap=0.2):
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = int(patch_size * (1 - overlap))

    def pad_image(self, image):
        h, w, c = image.shape
        pad_h = (self.stride - (h - self.patch_size) % self.stride) % self.stride
        pad_w = (self.stride - (w - self.patch_size) % self.stride) % self.stride
        padded_image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
        return padded_image

    def create_patches(self, image: np.ndarray) -> np.ndarray:
        """Splits an image into overlapping patches."""
        h, w, c = image.shape
        patches = []
        for y in range(0, h - self.patch_size + 1, self.stride):
            row = []
            for x in range(0, w - self.patch_size + 1, self.stride):
                patch = image[y:y+self.patch_size, x:x+self.patch_size]
                row.append(patch)
            patches.append(row)
        return np.array(patches)

class ImageRepatcher:
    def repatch_image(self, patches: np.ndarray, full_shape: tuple) -> np.ndarray:
        """Reconstructs the full image from patches."""
        h, w, c = full_shape
        reconstructed_image = np.zeros((h, w, c), dtype=np.uint8)
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                y = i * patches.shape[2]
                x = j * patches.shape[3]
                reconstructed_image[y:y+patches.shape[2], x:x+patches.shape[3]] = patches[i, j, 0]
        return reconstructed_image

class YOLOModel:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def predict(self, patch: np.ndarray) -> list:
        """Predicts objects in a patch using the YOLO model."""
        return self.model.predict(patch, verbose=False)

class ResultSaver:
    def __init__(self, base_folder: str) -> None:
        self.output_folder = Path(base_folder) / "BoundingBoxes"
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def save_image(self, image: np.ndarray, image_name: str) -> None:
        save_path = self.output_folder / f"{image_name}_bboxes.jpg"
        cv2.imwrite(str(save_path), image)

class XLSXUpdater:
    def __init__(self, base_folder: str) -> None:
        self.xlsx_path = Path(base_folder) / "Results.xlsx"
        if not self.xlsx_path.exists():
            self._create_xlsx()

    def _create_xlsx(self):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Detection Results"
        ws.append(["Filename", "Detection Count"])
        wb.save(self.xlsx_path)

    def update_xlsx(self, filenames, detections_dict):
        wb = openpyxl.load_workbook(self.xlsx_path)
        ws = wb["Detection Results"]

        for filename in filenames:
            detection_count = detections_dict.get(filename, 0)
            ws.append([filename, detection_count])

        wb.save(self.xlsx_path)

# ------------- Main Processor ------------- #

class ImageProcessor:
    def __init__(self, model_path):
        self.model_path = model_path

        # Find today's folders two levels up
        script_dir = Path(__file__).resolve().parent
        two_levels_up = script_dir.parent.parent
        
        today_str = datetime.now().strftime("%Y-%m-%d")
        self.test_images_paths = []

        for folder in two_levels_up.glob("*"):
            if folder.is_dir():
                folder_mtime = datetime.fromtimestamp(folder.stat().st_mtime).strftime("%Y-%m-%d")
                if folder_mtime == today_str:
                    self.test_images_paths.append(folder)

        if not self.test_images_paths:
            raise ValueError("No folders created today were found two levels up.")

        # Initialize utilities
        self.patcher = ImagePatcher()
        self.model = YOLOModel(model_path)
        self.repatcher = ImageRepatcher()
        self.savers = {str(path): ResultSaver(str(path)) for path in self.test_images_paths}
        self.xlsx_updaters = {str(path): XLSXUpdater(str(path)) for path in self.test_images_paths}

    def process_images(self):
        """Processes images in all today's folders."""
        for test_images_path in self.test_images_paths:
            detections_per_image = {}
            filenames = []

            image_paths = list(Path(test_images_path).glob("*.jpg"))
            
            # Process all images without any selection
            for img_path in image_paths:
                # Skip images with "Overlay" in the filename or ending with "_bboxes"
                if "Overlay" in img_path.stem or img_path.stem.endswith("_bboxes"):
                    continue

                image = cv2.imread(str(img_path))
                h, w, _ = image.shape

                padded_image = self.patcher.pad_image(image)
                padded_h, padded_w, _ = padded_image.shape

                patches = self.patcher.create_patches(padded_image)

                detections = []
                for i in range(patches.shape[0]):
                    for j in range(patches.shape[1]):
                        patch = patches[i, j, 0]
                        results = self.model.predict(patch)

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

                image_name = img_path.stem
                detections_count = len(detections)
                detections_per_image[image_name] = detections_count
                filenames.append(image_name)

                print(f"{image_name}: {detections_count} objects detected")

                reconstructed_image = self.repatcher.repatch_image(patches, (padded_h, padded_w, 3))
                reconstructed_image = reconstructed_image[:h, :w]

                # Draw semi-transparent bounding boxes
                overlay = reconstructed_image.copy()
                for x1, y1, x2, y2 in detections:
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 1)
                alpha = 0.5
                reconstructed_image = cv2.addWeighted(overlay, alpha, reconstructed_image, 1 - alpha, 0)

                self.savers[str(test_images_path)].save_image(reconstructed_image, image_name)

            self.xlsx_updaters[str(test_images_path)].update_xlsx(filenames, detections_per_image)


# ------------- Run ------------- #

if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    model_path = current_dir / "model.pt"  # full model path
    processor = ImageProcessor(str(model_path))
    processor.process_images()