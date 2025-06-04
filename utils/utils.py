import os
import cv2
import numpy as np
import torch
from torchvision.ops import nms
from patchify import patchify, unpatchify

class ImagePatcher:
    def __init__(self, patch_size=640, stride=576):
        self.patch_size = patch_size
        self.stride = stride

    def pad_image(self, image):
        h, w, _ = image.shape
        pad_h = (self.stride - (h % self.stride)) if (h % self.stride) != 0 else 0
        pad_w = (self.stride - (w % self.stride)) if (w % self.stride) != 0 else 0
        padded = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return padded

    def create_patches(self, image):
        # returns patches shaped (num_patches_y, num_patches_x, 1, patch_h, patch_w, 3)
        return patchify(image, (self.patch_size, self.patch_size, 3), step=self.stride)

class ImageRepatcher:
    def __init__(self, patch_size=640, stride=576):
        self.patch_size = patch_size
        self.stride = stride

    def repatch_image(self, patches, image_shape):
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

def apply_nms(boxes, scores, iou_threshold=0.5):
    if boxes.shape[0] == 0:
        return torch.empty((0,), dtype=torch.long)
    keep = nms(boxes, scores, iou_threshold)
    return keep

class ModelTrain:
    def __init__(self, model_path=None, imgsz=640, device='cpu'):
        from ultralytics import YOLO
        self.device = device
        if model_path:
            self.model = YOLO(model_path)
        else:
            self.model = YOLO('yolov8n.pt')  # default pre-trained yolov8n
        self.imgsz = imgsz

    def train(self, data_yaml, epochs=50, batch=16, save_dir='runs/train'):
        self.model.train(data=data_yaml, epochs=epochs, batch=batch, imgsz=self.imgsz, save=True, project=save_dir)

    def infer_on_patch(self, patch_img):
        results = self.model(patch_img, imgsz=self.imgsz)
        # Extract boxes and scores as tensors on CPU
        boxes = results[0].boxes.xyxy.cpu()  # (N,4) tensor
        scores = results[0].boxes.conf.cpu()  # (N,) tensor
        return boxes, scores

def load_ground_truth_yolo(labels_folder, imgs_folder):
    gt_dict = {}
    for fname in os.listdir(imgs_folder):
        if not fname.lower().endswith(('.jpg', '.jpeg')):
            continue
        img_path = os.path.join(imgs_folder, fname)
        h, w = cv2.imread(img_path).shape[:2]
        label_path = os.path.join(labels_folder, os.path.splitext(fname)[0] + '.txt')
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    # YOLO format: class_id cx cy w h (normalized)
                    _, cx, cy, bw, bh = map(float, parts)
                    x1 = (cx - bw / 2) * w
                    y1 = (cy - bh / 2) * h
                    x2 = (cx + bw / 2) * w
                    y2 = (cy + bh / 2) * h
                    boxes.append([x1, y1, x2, y2])
        gt_dict[fname] = np.array(boxes) if boxes else np.zeros((0,4))
    return gt_dict

def calculate_precision_recall(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    pred_boxes, gt_boxes: numpy arrays of shape (N,4) with [x1,y1,x2,y2]
    Returns precision and recall for one image.
    """
    def iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea)

    tp = 0
    fp = 0
    matched_gt = set()
    for pb in pred_boxes:
        matched = False
        for i, gb in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            if iou(pb, gb) >= iou_threshold:
                tp += 1
                matched_gt.add(i)
                matched = True
                break
        if not matched:
            fp += 1
    fn = len(gt_boxes) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall

def draw_boxes_on_image(image, boxes, color=(0, 255, 0), alpha=0.5, thickness=1):
    """
    Draw bounding boxes on an image with transparency.
    
    Args:
        image (np.ndarray): Original image.
        boxes (list): List of bounding boxes in [x1, y1, x2, y2, conf, cls].
        color (tuple): BGR color for the boxes.
        alpha (float): Transparency for the rectangles.
        thickness (int): Rectangle border thickness.

    Returns:
        np.ndarray: Image with drawn boxes.
    """
    overlay = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
    
    # Blend the original and overlay images
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def save_visualized_image(output_path, image_name, image):
    os.makedirs(output_path, exist_ok=True)
    out_file = os.path.join(output_path, image_name)
    cv2.imwrite(out_file, image)