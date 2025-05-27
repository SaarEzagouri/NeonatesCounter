"""
NeonatesCounter_V6.py
This script provides a pipeline for evaluating and processing images using a YOLO-based object detection model, 
specifically tailored for counting and analyzing neonates in images.

Has a pdoc file.
...
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from patchify import patchify
from openpyxl import load_workbook
from ultralytics import YOLO
import torch
from torchvision.ops import nms
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from datetime import datetime
import json

class ImagePatcher:
    """
    Handles image padding and patch extraction for tiled inference.

    Attributes:
        patch_size (int): Size of each patch.
        stride (int): Stride for patch extraction.
    """
    def __init__(self, patch_size=416, stride=376):
        """
        Initialize the ImagePatcher.

        Args:
            patch_size (int): Size of each patch.
            stride (int): Stride for patch extraction.
        """
        self.patch_size = patch_size
        self.stride = stride

    def pad_image(self, image):
        """
        Pad the image so its dimensions are multiples of the stride.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Padded image.
        """
        h, w, _ = image.shape
        pad_h = (self.stride - (h % self.stride)) if (h % self.stride) != 0 else 0
        pad_w = (self.stride - (w % self.stride)) if (w % self.stride) != 0 else 0
        return cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    def create_patches(self, image):
        """
        Create patches from the input image.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Array of patches.
        """
        return patchify(image, (self.patch_size, self.patch_size, 3), step=self.stride)

class YOLOModel:
    """
    Loads and runs predictions using a YOLOv8 model.

    Attributes:
        model: YOLO model instance.
    """
    def __init__(self, model_path):
        """
        Initialize the YOLOModel.

        Args:
            model_path (str or Path): Path to the YOLO model file.
        """
        self.model = YOLO(model_path)

    def predict(self, patch):
        """
        Run prediction on a single patch.

        Args:
            patch (np.ndarray): Image patch.

        Returns:
            list: YOLO detection results.
        """
        return self.model.predict(source=patch, conf=0.03, iou=0.3, imgsz=416, save=False, line_width=0, verbose=False, show_conf=False)

class ImageRepatcher:
    """
    Reconstructs images from patches if needed.

    Attributes:
        patch_size (int): Size of each patch.
        stride (int): Stride for patch extraction.
    """
    def __init__(self, patch_size=416, stride=376):
        """
        Initialize the ImageRepatcher.

        Args:
            patch_size (int): Size of each patch.
            stride (int): Stride for patch extraction.
        """
        self.patch_size = patch_size
        self.stride = stride

    def repatch_image(self, patches, image_shape):
        """
        Reconstruct the image from patches.

        Args:
            patches (np.ndarray): Array of patches.
            image_shape (tuple): Shape of the original image.

        Returns:
            np.ndarray: Reconstructed image.
        """
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

        count_map[count_map == 0] = 1
        return (reconstructed / count_map).astype(np.uint8)

class ResultSaver:
    """
    Saves processed images with bounding boxes and overlays.

    Attributes:
        output_path (str or Path): Directory to save results.
    """
    def __init__(self, output_path):
        """
        Initialize the ResultSaver.

        Args:
            output_path (str or Path): Directory to save results.
        """
        self.output_path = output_path

    def save_image(self, image, image_name):
        """
        Save an image with bounding boxes.

        Args:
            image (np.ndarray): Image to save.
            image_name (str): Name for the saved image.
        """
        output_image_path = os.path.join(self.output_path, f"{image_name}_bboxes.jpg")
        cv2.imwrite(output_image_path, image)

    def save_debug_image(self, image, image_name):
        """
        Save a debug image with bounding boxes.

        Args:
            image (np.ndarray): Debug image to save.
            image_name (str): Name for the saved image.
        """
        output_image_path = os.path.join(self.output_path, f"{image_name}_debug_bboxes.jpg")
        cv2.imwrite(output_image_path, image)
    
    def save_evaluation_image(self, image, image_name):
        """
        Save an evaluation overlay image.

        Args:
            image (np.ndarray): Evaluation image to save.
            image_name (str): Name for the saved image.
        """
        output_image_path = os.path.join(self.output_path, f"{image_name}_eval_overlay.jpg")
        cv2.imwrite(output_image_path, image)

class GroundTruthLoader:
    """
    Loads ground truth bounding boxes from VGG Image Annotator (VIA) JSON files.

    Attributes:
        annotation_path (str or Path): Path to the annotation JSON file.
        gt_data (dict): Loaded ground truth data.
    """
    def __init__(self, annotation_path):
        """
        Initialize the GroundTruthLoader.

        Args:
            annotation_path (str or Path): Path to the annotation JSON file.
        """
        self.annotation_path = annotation_path
        self.gt_data = self.load_ground_truth()
    
    def load_ground_truth(self):
        """
        Load ground truth from VGG Image Annotator JSON file.

        Returns:
            dict: Mapping from image stem to list of ground truth boxes.
        """
        gt_data = {}
        if not os.path.exists(self.annotation_path):
            print(f"Warning: Annotation file not found at {self.annotation_path}")
            return gt_data
        try:
            with open(self.annotation_path, 'r') as f:
                via_data = json.load(f)
            for img_id_key, img_info in via_data.items():
                filename = img_info.get('filename')
                if not filename:
                    continue
                image_stem = Path(filename).stem
                gt_boxes = []
                for region in img_info.get('regions', []):
                    shape_attributes = region.get('shape_attributes', {})
                    region_attributes = region.get('region_attributes', {})
                    if (shape_attributes.get('name') == 'rect' and 
                        region_attributes.get('type') == '0'):
                        x = shape_attributes.get('x')
                        y = shape_attributes.get('y')
                        width = shape_attributes.get('width')
                        height = shape_attributes.get('height')
                        if all(v is not None for v in [x, y, width, height]):
                            x1, y1 = int(x), int(y)
                            x2, y2 = int(x + width), int(y + height)
                            gt_boxes.append([x1, y1, x2, y2])
                if gt_boxes:
                    gt_data[image_stem] = gt_boxes
            print(f"Loaded ground truth for {len(gt_data)} images from {self.annotation_path}")
        except Exception as e:
            print(f"Error loading ground truth: {e}")
        return gt_data
    
    def get_ground_truth(self, image_stem):
        """
        Get ground truth boxes for a specific image.

        Args:
            image_stem (str): Image stem (filename without extension).

        Returns:
            list: List of ground truth boxes.
        """
        return self.gt_data.get(image_stem, [])

class EvaluationVisualizer:
    """
    Creates overlay images for visual comparison of detections and ground truth.

    Attributes:
        output_path (str or Path): Directory to save overlays.
    """
    def __init__(self, output_path):
        """
        Initialize the EvaluationVisualizer.

        Args:
            output_path (str or Path): Directory to save overlays.
        """
        self.output_path = output_path
    
    def create_evaluation_overlay(self, image_path, gt_boxes, det_boxes, image_stem):
        """
        Create overlay image with ground truth (green) and detections (red).

        Args:
            image_path (str or Path): Path to the image.
            gt_boxes (list): List of ground truth boxes.
            det_boxes (list): List of detection boxes.
            image_stem (str): Image stem for saving.

        Returns:
            np.ndarray or None: Overlay image or None if error.
        """
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return None
        overlay = image.copy()
        for gt_box in gt_boxes:
            x1, y1, x2, y2 = map(int, gt_box)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(overlay, "GT", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        for det_box in det_boxes:
            x1, y1, x2, y2, conf = det_box[:5]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(overlay, f"Det ({conf:.2f})", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        alpha = 0.7
        result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        output_path = os.path.join(self.output_path, f"{image_stem}_eval_overlay.jpg")
        cv2.imwrite(output_path, result)
        print(f"Evaluation overlay saved to {output_path}")
        return result

class XLSXUpdater:
    """
    Updates Excel files with detection counts per image.

    Attributes:
        folder_path (str or Path): Folder containing the Excel file.
        file_path (str or Path): Path to the Excel file.
    """
    def __init__(self, folder_path):
        """
        Initialize the XLSXUpdater.

        Args:
            folder_path (str or Path): Folder containing the Excel file.
        """
        self.folder_path = folder_path
        self.file_path = self.find_xlsx_file()

    def find_xlsx_file(self):
        """
        Find the first Excel file in the folder.

        Returns:
            str or None: Path to the Excel file, or None if not found.
        """
        for file in os.listdir(self.folder_path):
            if file.endswith(".xlsx"):
                return os.path.join(self.folder_path, file)
        return None

    def update_xlsx(self, filenames, detections_per_image):
        """
        Update the Excel file with detection counts.

        Args:
            filenames (list): List of image filenames.
            detections_per_image (dict): Mapping from filename to detection count.
        """
        if not self.file_path:
            print("No Excel file found, skipping Excel update.")
            return
        df = pd.read_excel(self.file_path, engine='openpyxl')
        image_filenames = filenames + ['mean']
        detection_values = [detections_per_image[name] for name in filenames]
        mean_detection = int(np.round(np.mean(detection_values)))
        detection_counts = detection_values + [mean_detection]
        df = df.reindex(range(len(image_filenames)))
        df["image_filename"] = image_filenames
        df["detections_count"] = detection_counts
        df.to_excel(self.file_path, index=False, engine='openpyxl')

class ImageProcessor:
    """
    Orchestrates the full pipeline, including patch inference, non-maximum suppression (NMS),
    shape-based filtering, visualization, and evaluation metrics (precision, recall).

    Attributes:
        model_path (str or Path): Path to the YOLO model.
        root_folder (str or Path): Folder containing images.
        show_scores (bool): If True, displays detection confidence scores on the output images.
        debugging (bool): If True, enables additional debug output and visualizations for troubleshooting.
        eval_mode (bool): If True, runs the processor in evaluation mode, comparing results to ground truth annotations.
        anno_path (str or Path): Path to annotation file.
    """
    def __init__(self, model_path, root_folder, show_scores=True, debugging=False, eval_mode=False, anno_path=None):
        """
        Initialize the ImageProcessor.

        Args:
            model_path (str or Path): Path to the YOLO model.
            root_folder (str or Path): Folder containing images.
            show_scores (bool): Whether to show confidence scores.
            debugging (bool): Enable debugging mode.
            eval_mode (bool): Enable evaluation mode.
            anno_path (str or Path, optional): Path to annotation file.
        """
        self.model_path = model_path
        self.root_folder = root_folder
        self.show_scores = show_scores
        self.debugging = debugging
        self.eval_mode = eval_mode
        self.patcher = ImagePatcher()
        self.model = YOLOModel(model_path)
        self.repatcher = ImageRepatcher()
        self.saver = ResultSaver(root_folder)
        self.xlsx_updater = XLSXUpdater(root_folder)
        if self.eval_mode:
            self.all_detections = {}
            if anno_path:
                self.gt_loader = GroundTruthLoader(anno_path)
                self.eval_visualizer = EvaluationVisualizer(root_folder)
                self.all_image_detections = []
                self.all_image_ground_truths = []
            else:
                print("Warning: eval_mode=True but no anno_path provided. Evaluation visualization and overall metrics will be disabled.")
                self.gt_loader = None
                self.eval_visualizer = None
                self.all_image_detections = []
                self.all_image_ground_truths = []

    def deduplicate_boxes(self, boxes_with_scores, iou_threshold=0.45, return_scores=False):
        """
        Remove duplicate bounding boxes using Non-Maximum Suppression (NMS).

        Args:
            boxes_with_scores (list): List of boxes with scores.
            iou_threshold (float): IoU threshold for NMS.
            return_scores (bool): Whether to return scores.

        Returns:
            list: Deduplicated boxes.
        """
        if len(boxes_with_scores) == 0:
            return []
        boxes_np = np.array([[x1, y1, x2, y2] for x1, y1, x2, y2, _ in boxes_with_scores])
        scores = np.array([score for *_, score in boxes_with_scores])
        boxes_tensor = torch.tensor(boxes_np, dtype=torch.float32)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        keep_indices = nms(boxes_tensor, scores_tensor, iou_threshold)
        kept = np.array(boxes_with_scores)[keep_indices]
        if return_scores:
            return [(int(x1), int(y1), int(x2), int(y2), float(score)) for x1, y1, x2, y2, score in kept]
        else:
            return [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2, _ in kept]

    def filter_by_shape_metrics_percentile(self, image, detections):
        """
        Filter detections by shape metrics using percentile thresholds calculated
        from all detected contours in the image.

        Args:
            image (np.ndarray): Input image.
            detections (list): List of detection boxes.

        Returns:
            list: Filtered detections.
        """
        circularities = []
        solidities = []
        areas = []
        extents = []
        convexities = []
        contour_data = []
        for det in detections:
            x1, y1, x2, y2, score = det[:5]
            roi = image[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            contours = self._find_contours(gray)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area == 0:
                    continue
                perimeter = cv2.arcLength(cnt, True)
                circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0
                x, y, w, h = cv2.boundingRect(cnt)
                rect_area = w * h
                extent = float(area) / rect_area if rect_area > 0 else 0
                convexity = float(area) / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
                circularities.append(circularity)
                solidities.append(solidity)
                areas.append(area)
                extents.append(extent)
                convexities.append(convexity)
                contour_data.append({
                    "det": det,
                    "circularity": circularity,
                    "solidity": solidity,
                    "area": area,
                    "extent": extent,
                    "convexity": convexity
                })
        if not contour_data:
            return []
        min_circ_thresh = np.percentile(circularities, 1)
        max_circ_thresh = np.percentile(circularities, 100)
        min_solid_thresh = np.percentile(solidities, 0)
        max_solid_thresh = np.percentile(solidities, 100)
        min_area_thresh = np.percentile(areas, 3)
        max_area_thresh = np.percentile(areas, 100)
        min_extent_thresh = np.percentile(extents, 1)
        max_extent_thresh = np.percentile(extents, 100)
        min_convex_thresh = np.percentile(convexities, 0)
        max_convex_thresh = np.percentile(convexities, 100)
        filtered_detections = []
        det_seen = set()
        for cdata in contour_data:
            det = cdata["det"]
            det_tuple = tuple(det)
            if det_tuple in det_seen:
                continue
            if (0.1 <= cdata["circularity"] <= max_circ_thresh and
                0.2 <= cdata["solidity"] <= max_solid_thresh and
                5 <= cdata["area"] <= max_area_thresh and
                0.1 <= cdata["extent"] <= max_extent_thresh and
                0.1 <= cdata["convexity"] <= max_convex_thresh):
                filtered_detections.append(det)
                det_seen.add(det_tuple)
        return filtered_detections

    def _find_contours(self, gray_image):
        """
        Find contours in a grayscale image.

        Args:
            gray_image (np.ndarray): Grayscale image.

        Returns:
            list: List of contours.
        """
        _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def _calculate_iou(self, boxA, boxB):
        """
        Calculates Intersection over Union (IoU) of two bounding boxes.

        Args:
            boxA (list): First box [x1, y1, x2, y2].
            boxB (list): Second box [x1, y1, x2, y2].

        Returns:
            float: IoU value.
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def calculate_metrics(self, all_gt_boxes, all_det_boxes, iou_threshold=0.3):
        """
        Calculates Precision and Recall across all images.

        Args:
            all_gt_boxes (list): List of ground truth boxes.
            all_det_boxes (list): List of detection boxes.
            iou_threshold (float): IoU threshold for matching.

        Returns:
            tuple: (precision, recall, true_positives, false_positives, false_negatives)
        """
        all_det_boxes.sort(key=lambda x: x[4], reverse=True)
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        matched_gt = [False] * len(all_gt_boxes)
        for det_idx, det_box_with_conf in enumerate(all_det_boxes):
            det_box = det_box_with_conf[:4]
            best_iou = 0
            best_gt_idx = -1
            for gt_idx, gt_box in enumerate(all_gt_boxes):
                iou = self._calculate_iou(det_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            if best_iou >= iou_threshold:
                if not matched_gt[best_gt_idx]:
                    true_positives += 1
                    matched_gt[best_gt_idx] = True
                else:
                    false_positives += 1 
            else:
                false_positives += 1
        false_negatives = sum(1 for matched in matched_gt if not matched)
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        return precision, recall, true_positives, false_positives, false_negatives

    def process_images(self, analyze_distributions=False, distribution_plot_path=None):
        """
        Process all images in the root folder.

        Args:
            analyze_distributions (bool): Whether to analyze detection count distributions.
            distribution_plot_path (str or Path, optional): Path to save the distribution plot.
        """
        detections_per_image = {}
        filenames = []
        image_paths = list(self.root_folder.glob("*.jpg"))
        image_paths = [img_path for img_path in image_paths if "Overlay" not in img_path.stem and not img_path.stem.endswith("_bboxes")]
        if not image_paths:
            print(f"No valid images found in {self.root_folder}, skipping.")
            return
        line_thickness = 1
        alpha = 0.4
        all_folder_gt_boxes = []
        all_folder_det_boxes = []
        for img_path in image_paths:
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Could not read image {img_path}, skipping.")
                continue
            padded_image = self.patcher.pad_image(image)
            patches = self.patcher.create_patches(padded_image)
            raw_detections = []
            for i in range(patches.shape[0]):
                for j in range(patches.shape[1]):
                    patch = patches[i, j, 0]
                    results = self.model.predict(patch)
                    if results:
                        for result in results:
                            for box in result.boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                conf = float(box.conf[0]) if hasattr(box, "conf") else 1.0
                                raw_detections.append((
                                    x1 + j * self.patcher.stride,
                                    y1 + i * self.patcher.stride,
                                    x2 + j * self.patcher.stride,
                                    y2 + i * self.patcher.stride,
                                    conf
                                ))
            nms_detections = self.deduplicate_boxes(raw_detections, iou_threshold=0.45, return_scores=True)
            filtered_detections = self.filter_by_shape_metrics_percentile(image, nms_detections)
            detections_per_image[img_path.name] = len(filtered_detections)
            filenames.append(img_path.name)
            print(f"{img_path.name}: {len(filtered_detections)} objects detected after filtering.")
            if self.eval_mode:
                self.all_detections[img_path.stem] = filtered_detections
                gt_boxes = self.gt_loader.get_ground_truth(img_path.stem)
                if gt_boxes:
                    print(f"{img_path.name}: {len(gt_boxes)} ground truth objects found.")
                    self.eval_visualizer.create_evaluation_overlay(
                        img_path, gt_boxes, filtered_detections, img_path.stem
                    )
                    all_folder_gt_boxes.extend(gt_boxes)
                    all_folder_det_boxes.extend(filtered_detections)
                else:
                    print(f"Annotation for the image {img_path.stem} not found. Skipping evaluation visualization for this image.")
            if self.debugging:
                image_debug = image.copy().astype(np.float32)
                overlay_debug = image_debug.copy()
                filtered_set = {tuple(det) for det in filtered_detections}
                for det in nms_detections:
                    x1, y1, x2, y2, conf = det
                    if tuple(det) in filtered_set:
                        color = (0, 255, 0)
                    else:
                        color = (0, 0, 255)
                    cv2.rectangle(overlay_debug, (x1, y1), (x2, y2), color, line_thickness)
                    if self.show_scores:
                        text = f"{conf:.2f}"
                        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        text_y = max(y1 - 5, text_height + 5)
                        cv2.putText(overlay_debug, text, (x1, text_y),
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                image_debug = cv2.addWeighted(overlay_debug, alpha, image_debug, 1 - alpha, 0)
                image_debug = image_debug.astype(np.uint8)
                self.saver.save_debug_image(image_debug, img_path.stem)
            else:
                image_final = image.copy().astype(np.float32)
                overlay_final = image_final.copy()
                for det in filtered_detections:
                    x1, y1, x2, y2, conf = det
                    color = (0, 0, 255)
                    cv2.rectangle(overlay_final, (x1, y1), (x2, y2), color, line_thickness)
                    if self.show_scores:
                        text = f"{conf:.2f}"
                        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        text_y = max(y1 - 5, text_height + 5)
                        cv2.putText(overlay_final, text, (x1, text_y),
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                image_final = cv2.addWeighted(overlay_final, alpha, image_final, 1 - alpha, 0)
                image_final = image_final.astype(np.uint8)
                self.saver.save_image(image_final, img_path.stem)
        self.xlsx_updater.update_xlsx(filenames, detections_per_image)
        if analyze_distributions and distribution_plot_path:
            self.plot_distribution(detections_per_image, distribution_plot_path)
        if self.eval_mode:
            self.print_evaluation_summary(all_folder_gt_boxes, all_folder_det_boxes)

    def print_evaluation_summary(self, all_folder_gt_boxes, all_folder_det_boxes):
        """
        Print summary of evaluation results, including overall Precision and Recall.

        Args:
            all_folder_gt_boxes (list): All ground truth boxes in the folder.
            all_folder_det_boxes (list): All detection boxes in the folder.
        """
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        total_images_processed_for_eval = len(self.all_detections)
        total_detections = sum(len(dets) for dets in self.all_detections.values())
        total_gt_found = len(all_folder_gt_boxes)
        print(f"Total images processed: {total_images_processed_for_eval}")
        print(f"Total detections: {total_detections}")
        print(f"Total ground truth objects found for evaluation: {total_gt_found}")
        print("\nPer-image breakdown:")
        for img_stem, detections in self.all_detections.items():
            gt_count = len(self.gt_loader.get_ground_truth(img_stem))
            det_count = len(detections)
            print(f"  {img_stem}: {det_count} detections, {gt_count} ground truth")
        if all_folder_gt_boxes or all_folder_det_boxes:
            precision, recall, tp, fp, fn = self.calculate_metrics(all_folder_gt_boxes, all_folder_det_boxes)
            print("\n--- Overall Metrics (across all images in folder) ---")
            print(f"True Positives (TP): {tp}")
            print(f"False Positives (FP): {fp}")
            print(f"False Negatives (FN): {fn}")
            print(f"Overall Precision: {precision:.4f}")
            print(f"Overall Recall: {recall:.4f}")
        else:
            print("\nNo ground truth annotations or detections available for overall metric calculation in this folder.")
        print("="*50)

    def plot_distribution(self, detections_dict, save_path):
        """
        Plot and save the distribution of detection counts per image.

        Args:
            detections_dict (dict): Mapping from image name to detection count.
            save_path (str or Path): Path to save the plot.
        """
        counts = np.array(list(detections_dict.values()))
        if len(counts) == 0:
            print("No detection counts to plot distribution.")
            return
        if len(np.unique(counts)) < 2:
            plt.figure(figsize=(10, 6))
            plt.bar(np.unique(counts), len(counts))
            plt.xlabel('Number of Detected Objects')
            plt.ylabel('Count of Images')
            plt.title('Distribution of Object Counts per Image (All counts are same)')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            print(f"Distribution plot saved to {save_path}")
            return
        density = gaussian_kde(counts)
        xs = np.linspace(min(counts), max(counts), 200)
        ys = density(xs)
        plt.figure(figsize=(10, 6))
        plt.plot(xs, ys, label='Density')
        plt.scatter(counts, density(counts), color='r', label='Detections per Image')
        plt.xlabel('Number of Detected Objects')
        plt.ylabel('Density')
        plt.title('Distribution of Object Counts per Image')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Distribution plot saved to {save_path}")

def main():
    """
    This function performs the following steps:
    1. Determines the script and model paths.
    2. Sets the image directory and creates an analysis directory if it does not exist.
    3. Identifies all subfolders in the image directory that were created today (excluding the analysis directory).
    4. For each folder created today:
        - Creates a corresponding analysis directory.
        - Initializes an ImageProcessor with the following options:
            - show_scores (bool): If True, displays detection confidence scores on the output images.
            - debugging (bool): If True, enables additional debug output and visualizations for troubleshooting.
            - eval_mode (bool): If True, runs the processor in evaluation mode, comparing results to ground truth annotations.
        - Processes the images in the folder and saves analysis plots.
    5. Prints a completion message when processing is finished.
    
    """
    script_path = Path(__file__).parent
    model_path = script_path / "NeonatesCounter_v1.0_model.pt"
    img_dir = Path("C:/Projects/FreeZem/NeonatesCounter/Production/testing_images/test1")
    analysis_dir = img_dir / "analysis_plots"
    analysis_dir.mkdir(exist_ok=True)
    today = datetime.now().strftime('%Y-%m-%d')
    today_folders = []
    for folder in img_dir.iterdir():
        if folder.is_dir() and folder.name != "analysis_plots":
            try:
                creation_time = datetime.fromtimestamp(folder.stat().st_ctime).strftime('%Y-%m-%d')
                if creation_time == today:
                    today_folders.append(folder)
            except OSError as e:
                print(f"Could not get creation time for {folder}: {e}")
                continue
    if not today_folders:
        print(f"No folders created today ({today}) found in {img_dir}. Exiting.")
        return
    for folder in today_folders:
        print(f"Processing folder: {folder}")
        folder_analysis_dir = analysis_dir / folder.name
        folder_analysis_dir.mkdir(exist_ok=True)
        eval_mode = True
        anno_path = "C:/Projects/FreeZem/NeonatesCounter/Production/Eval_Annotations/test1_VGG_format/test1_annotations.json"
        processor = ImageProcessor(
            model_path=model_path,
            root_folder=folder,
            show_scores=False,
            debugging=False,
            eval_mode=eval_mode,
            anno_path=anno_path
        )
        processor.process_images(
            analyze_distributions=False,
            distribution_plot_path=folder_analysis_dir / "distribution_plot.png"
        )
    print("Processing Completed")

if __name__ == "__main__":
    main()