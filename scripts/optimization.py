import os
import cv2
import numpy as np
import torch
import yaml
from pathlib import Path
from ultralytics import YOLO
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import optuna
from optuna.samplers import TPESampler
import json

model_size='C:/Projects/FreeZem/NeonatesCounter/models/NeonatesCounter_v1.0_model.pt'

def load_hyperparameters_from_yaml(yaml_filepath):
    """
    Loads hyperparameters from a YAML file into a Python dictionary.
    """
    with open(yaml_filepath, 'r') as file:
        hyperparams_dict = yaml.safe_load(file)
    return hyperparams_dict
params_path = 'C:/Projects/FreeZem/NeonatesCounter/models/NeonatesCounter_v1.0_model_args.yaml'
model_params = load_hyperparameters_from_yaml(params_path)

class YOLOPatchTrainer:
    def __init__(self, 
                 train_images_dir,
                 train_labels_dir,
                 val_images_dir,
                 val_labels_dir,
                 dataset_yaml_path,
                 patch_size=416,
                 stride_ratio=0.5,
                 target_precision=0.99,
                 target_recall=0.99,
                 ):
        """
        Initialize the YOLO patch trainer
        
        Args:
            train_images_dir: Path to training images
            train_labels_dir: Path to training labels
            val_images_dir: Path to validation images  
            val_labels_dir: Path to validation labels
            patch_size: Size of patches (default 640)
            stride_ratio: Stride ratio to prevent overlap artifacts (0.8 means 20% overlap)
            target_precision: Target precision for optimization
            target_recall: Target recall for optimization
        """
        self.train_images_dir = Path(train_images_dir)
        self.train_labels_dir = Path(train_labels_dir)
        self.val_images_dir = Path(val_images_dir)
        self.val_labels_dir = Path(val_labels_dir)
        self.patch_size = patch_size
        self.stride = int(patch_size * stride_ratio)
        self.target_precision = target_precision
        self.target_recall = target_recall
        self.dataset_yaml_path = Path(dataset_yaml_path)
        
        # Create working directories
        self.work_dir = Path("yolo_patch_work")
        self.patches_dir = self.work_dir / "patches"
        self.dataset_dir = self.work_dir / "dataset"
        self.models_dir = self.work_dir / "models"
        
        for dir_path in [self.work_dir, self.patches_dir, self.dataset_dir, self.models_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)
        
        # Initialize tracking variables
        self.best_f1 = 0.0
        self.best_model_path = None
        self.patch_info = {}  # Store patch information for repatching
        
        # Hyperparameter optimization variables
        self.optimization_results = []
        self.best_trial_params = None
        
    def load_yolo_annotations(self, label_file):
        """Load YOLO format annotations"""
        annotations = []
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        annotations.append([class_id, x_center, y_center, width, height])
        return annotations
    
    def convert_yolo_to_absolute(self, annotations, img_width, img_height):
        """Convert YOLO format (normalized) to absolute coordinates"""
        absolute_annotations = []
        for ann in annotations:
            class_id, x_center, y_center, width, height = ann
            
            # Convert to absolute coordinates
            abs_x_center = x_center * img_width
            abs_y_center = y_center * img_height
            abs_width = width * img_width
            abs_height = height * img_height
            
            # Convert to x1, y1, x2, y2
            x1 = abs_x_center - abs_width / 2
            y1 = abs_y_center - abs_height / 2
            x2 = abs_x_center + abs_width / 2
            y2 = abs_y_center + abs_height / 2
            
            absolute_annotations.append([class_id, x1, y1, x2, y2])
        
        return absolute_annotations
    
    def convert_absolute_to_yolo(self, annotations, img_width, img_height):
        """Convert absolute coordinates to YOLO format (normalized)"""
        yolo_annotations = []
        for ann in annotations:
            class_id, x1, y1, x2, y2 = ann
            
            # Calculate center and dimensions
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            
            # Normalize
            norm_x_center = x_center / img_width
            norm_y_center = y_center / img_height
            norm_width = width / img_width
            norm_height = height / img_height
            
            # Ensure values are within bounds
            norm_x_center = max(0, min(1, norm_x_center))
            norm_y_center = max(0, min(1, norm_y_center))
            norm_width = max(0, min(1, norm_width))
            norm_height = max(0, min(1, norm_height))
            
            yolo_annotations.append([class_id, norm_x_center, norm_y_center, norm_width, norm_height])
        
        return yolo_annotations
    
    def get_patch_coordinates(self, img_width, img_height):
        """Get patch coordinates for an image"""
        patches = []
        
        for y in range(0, img_height, self.stride):
            for x in range(0, img_width, self.stride):
                x1 = x
                y1 = y
                x2 = min(x + self.patch_size, img_width)
                y2 = min(y + self.patch_size, img_height)
                
                # Adjust patch to ensure it's exactly patch_size when possible
                if x2 - x1 < self.patch_size and x2 == img_width:
                    x1 = max(0, img_width - self.patch_size)
                if y2 - y1 < self.patch_size and y2 == img_height:
                    y1 = max(0, img_height - self.patch_size)
                
                patches.append((x1, y1, x2, y2))
        
        return patches
    
    def filter_annotations_for_patch(self, annotations, patch_coords):
        """Filter annotations that are within or overlap with a patch"""
        x1_patch, y1_patch, x2_patch, y2_patch = patch_coords
        filtered_annotations = []
        
        for ann in annotations:
            class_id, x1, y1, x2, y2 = ann
            
            # Check if annotation overlaps with patch
            if (x1 < x2_patch and x2 > x1_patch and 
                y1 < y2_patch and y2 > y1_patch):
                
                # Clip annotation to patch boundaries
                clipped_x1 = max(x1, x1_patch)
                clipped_y1 = max(y1, y1_patch)
                clipped_x2 = min(x2, x2_patch)
                clipped_y2 = min(y2, y2_patch)
                
                # Convert to patch-relative coordinates
                rel_x1 = clipped_x1 - x1_patch
                rel_y1 = clipped_y1 - y1_patch
                rel_x2 = clipped_x2 - x1_patch
                rel_y2 = clipped_y2 - y1_patch
                
                # Only keep if the clipped annotation has reasonable size
                if (rel_x2 - rel_x1) > 10 and (rel_y2 - rel_y1) > 10:
                    filtered_annotations.append([class_id, rel_x1, rel_y1, rel_x2, rel_y2])
        
        return filtered_annotations
    
    def create_patches(self, images_dir, labels_dir, output_dir, split_name):
        """Create patches from images and convert annotations"""
        patch_images_dir = output_dir / "images" / split_name
        patch_labels_dir = output_dir / "labels" / split_name
        
        patch_images_dir.mkdir(parents=True, exist_ok=True)
        patch_labels_dir.mkdir(parents=True, exist_ok=True)
        
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        patch_count = 0
        
        print(f"Creating patches for {split_name} set...")
        
        for img_file in tqdm(image_files):
            # Load image
            img = cv2.imread(str(img_file))
            if img is None:
                continue
                
            img_height, img_width = img.shape[:2]
            
            # Load annotations
            label_file = labels_dir / f"{img_file.stem}.txt"
            annotations = self.load_yolo_annotations(label_file)
            
            # Convert to absolute coordinates
            abs_annotations = self.convert_yolo_to_absolute(annotations, img_width, img_height)
            
            # Get patch coordinates
            patch_coords = self.get_patch_coordinates(img_width, img_height)
            
            # Store patch info for repatching
            self.patch_info[img_file.stem] = {
                'original_size': (img_width, img_height),
                'patch_coords': patch_coords
            }
            
            # Create patches
            for i, (x1, y1, x2, y2) in enumerate(patch_coords):
                # Extract patch
                patch = img[y1:y2, x1:x2]
                
                # Resize patch to exact patch_size if needed
                if patch.shape[:2] != (self.patch_size, self.patch_size):
                    patch = cv2.resize(patch, (self.patch_size, self.patch_size))
                
                # Filter annotations for this patch
                patch_annotations = self.filter_annotations_for_patch(abs_annotations, (x1, y1, x2, y2))
                
                # Save patch only if it has annotations or is from training set
                if patch_annotations or split_name == 'train':
                    patch_name = f"{img_file.stem}_patch_{i}"
                    
                    # Save patch image
                    cv2.imwrite(str(patch_images_dir / f"{patch_name}.jpg"), patch)
                    
                    # Convert annotations to YOLO format for patch
                    patch_height, patch_width = patch.shape[:2]
                    yolo_annotations = self.convert_absolute_to_yolo(
                        patch_annotations, patch_width, patch_height
                    )
                    
                    # Save annotations
                    with open(patch_labels_dir / f"{patch_name}.txt", 'w') as f:
                        for ann in yolo_annotations:
                            f.write(f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")
                    
                    patch_count += 1
        
        print(f"Created {patch_count} patches for {split_name} set")
        return patch_count
        
    def repatch_predictions(self, predictions, original_img_name):
        """Repatch predictions back to original image coordinates"""
        if original_img_name not in self.patch_info:
            return []
        
        patch_info = self.patch_info[original_img_name]
        original_width, original_height = patch_info['original_size']
        patch_coords = patch_info['patch_coords']
        
        full_image_predictions = []
        
        for i, (x1_patch, y1_patch, x2_patch, y2_patch) in enumerate(patch_coords):
            patch_name = f"{original_img_name}_patch_{i}"
            
            if patch_name in predictions:
                patch_preds = predictions[patch_name]
                
                for pred in patch_preds:
                    class_id, confidence, x_center, y_center, width, height = pred
                    
                    # Convert from normalized patch coordinates to absolute patch coordinates
                    abs_x_center = x_center * self.patch_size
                    abs_y_center = y_center * self.patch_size
                    abs_width = width * self.patch_size
                    abs_height = height * self.patch_size
                    
                    # Convert to original image coordinates
                    orig_x_center = abs_x_center + x1_patch
                    orig_y_center = abs_y_center + y1_patch
                    
                    # Convert back to normalized coordinates for the original image
                    norm_x_center = orig_x_center / original_width
                    norm_y_center = orig_y_center / original_height
                    norm_width = abs_width / original_width
                    norm_height = abs_height / original_height
                    
                    full_image_predictions.append([
                        class_id, confidence, norm_x_center, norm_y_center, norm_width, norm_height
                    ])
        
        return full_image_predictions
    
    def apply_global_nms(self, predictions, iou_threshold=0.5):
        """Apply Non-Maximum Suppression on full image predictions"""
        if not predictions:
            return []
        
        # Convert to format suitable for NMS
        boxes = []
        scores = []
        classes = []
        
        for pred in predictions:
            class_id, confidence, x_center, y_center, width, height = pred
            
            # Convert to x1, y1, x2, y2 format
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            boxes.append([x1, y1, x2, y2])
            scores.append(confidence)
            classes.append(class_id)
        
        # Apply NMS
        boxes = torch.tensor(boxes, dtype=torch.float32)
        scores = torch.tensor(scores, dtype=torch.float32)
        
        keep_indices = torch.ops.torchvision.nms(boxes, scores, iou_threshold)
        
        # Return filtered predictions
        filtered_predictions = []
        for idx in keep_indices:
            filtered_predictions.append(predictions[idx])
        
        return filtered_predictions
    
    def evaluate_full_image(self, predictions, ground_truth, iou_threshold=0.5):
        """Evaluate predictions on full image"""
        # Apply global NMS
        predictions = self.apply_global_nms(predictions, iou_threshold)
        
        if not predictions and not ground_truth:
            return 1.0, 1.0, 1.0  # Perfect scores for empty image
        
        if not predictions:
            return 0.0, 1.0 if not ground_truth else 0.0, 0.0
        
        if not ground_truth:
            return 0.0, 0.0, 0.0
        
        # Convert predictions and ground truth to comparable format
        pred_boxes = []
        for pred in predictions:
            class_id, confidence, x_center, y_center, width, height = pred
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            pred_boxes.append([x1, y1, x2, y2, class_id, confidence])
        
        gt_boxes = []
        for gt in ground_truth:
            class_id, x_center, y_center, width, height = gt
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            gt_boxes.append([x1, y1, x2, y2, class_id])
        
        # Calculate IoU and match predictions with ground truth
        matches = []
        for i, pred_box in enumerate(pred_boxes):
            best_iou = 0
            best_match = -1
            
            for j, gt_box in enumerate(gt_boxes):
                if pred_box[4] == gt_box[4]:  # Same class
                    iou = self.calculate_iou(pred_box[:4], gt_box[:4])
                    if iou > best_iou and iou >= iou_threshold:
                        best_iou = iou
                        best_match = j
            
            if best_match >= 0:
                matches.append((i, best_match))
        
        # Calculate metrics
        tp = len(matches)
        fp = len(pred_boxes) - tp
        fn = len(gt_boxes) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union of two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def custom_validation(self, model, epoch):
        """Custom validation on full images"""
        print(f"Running custom validation for epoch {epoch}...")
        
        # Get validation images
        val_images = list(self.val_images_dir.glob("*.jpg")) + list(self.val_images_dir.glob("*.png"))
        
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        valid_images = 0
        
        for img_file in tqdm(val_images):
            # Load ground truth
            label_file = self.val_labels_dir / f"{img_file.stem}.txt"
            ground_truth = self.load_yolo_annotations(label_file)
            
            # Get predictions for all patches of this image
            patch_predictions = {}
            patch_info = self.patch_info.get(img_file.stem)
            
            if patch_info:
                for i, patch_coords in enumerate(patch_info['patch_coords']):
                    patch_name = f"{img_file.stem}_patch_{i}"
                    patch_img_path = self.dataset_dir / "images" / "val" / f"{patch_name}.jpg"
                    
                    if patch_img_path.exists():
                        # Run inference on patch
                        results = model(str(patch_img_path), verbose=False,conf=0.25,iou=0.45)
                        
                        if results and len(results) > 0 and results[0].boxes is not None:
                            boxes = results[0].boxes
                            patch_preds = []
                            
                            for box in boxes:
                                class_id = int(box.cls[0])
                                confidence = float(box.conf[0])
                                x_center, y_center, width, height = box.xywhn[0].tolist()
                                
                                patch_preds.append([class_id, confidence, x_center, y_center, width, height])
                            
                            patch_predictions[patch_name] = patch_preds
                
                # Repatch predictions
                full_image_predictions = self.repatch_predictions(patch_predictions, img_file.stem)
                
                # Evaluate
                precision, recall, f1 = self.evaluate_full_image(full_image_predictions, ground_truth)
                
                total_precision += precision
                total_recall += recall
                total_f1 += f1
                valid_images += 1
        
        if valid_images > 0:
            avg_precision = total_precision / valid_images
            avg_recall = total_recall / valid_images
            avg_f1 = total_f1 / valid_images
            
            print(f"Epoch {epoch} - Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}")
            
            # Save best model based on F1 score and target metrics
            target_met = avg_precision >= self.target_precision and avg_recall >= self.target_recall
            if avg_f1 > self.best_f1 or target_met:
                self.best_f1 = avg_f1
                self.best_model_path = self.models_dir / f"best_model_epoch_{epoch}.pt"
                model.save(str(self.best_model_path))
                print(f"New best model saved with F1: {avg_f1:.4f}")
                
                if target_met:
                    print(f"Target metrics achieved! Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}")
            
            return avg_precision, avg_recall, avg_f1
        
        return 0.0, 0.0, 0.0
    
    def train(self, model_size=model_size, epochs=100, **train_kwargs):
        """
        Main training function with full YOLO hyperparameter support
        ...
        """
        print("Starting YOLO patch training pipeline...")

        # Step 1: Create patches for training set
        print("Step 1: Creating training patches...")
        self.create_patches(self.train_images_dir, self.train_labels_dir, self.dataset_dir, "train")

        # Step 2: Create patches for validation set
        print("Step 2: Creating validation patches...")
        self.create_patches(self.val_images_dir, self.val_labels_dir, self.dataset_dir, "val")

        # Step 3: Remove this line if you're providing the YAML path manually
        # print("Step 3: Creating dataset configuration...")
        # self.create_dataset_yaml() # <--- REMOVE THIS LINE

        # Step 4: Initialize YOLO model
        print(f"Step 4: Initializing YOLO model ({model_size})...")
        model = YOLO(model_size)
        
        # Step 5: Prepare training arguments
        print("Step 5: Preparing training configuration...")
        
        # Default training arguments optimized for high precision/recall
        default_args = {
            'data': str(self.dataset_yaml_path),
            'epochs': 1,  # We'll train epoch by epoch for custom validation
            'batch': 16,
            'lr0': 0.001,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            
            # Augmentation settings - moderate for high precision
            'hsv_h': 0.015,
            'hsv_s': 0.7, 
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.2,  # Reduced scale for better precision
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 0.8,  # Reduced mosaic for better precision
            'mixup': 0.0,
            'copy_paste': 0.0,
            
            # Loss settings optimized for precision/recall
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            
            # Validation settings
            'conf': 0.001,  # Low confidence for high recall
            'iou': 0.6,     # Moderate IoU for good precision
            
            # Other settings
            'patience': 50,
            'save': False,  # We handle saving manually
            'verbose': True,
            'project': str(self.work_dir),
            'workers': 8,
            'seed': 0,
            'deterministic': True,
            'amp': True,
            'rect': False,  # Keep False for consistent patch size
            'cos_lr': True,  # Cosine LR for better convergence
            'close_mosaic': 10,
        }
        
        # Override defaults with user-provided arguments
        default_args.update(train_kwargs)
        
        print(f"Training configuration:")
        for key, value in default_args.items():
            if key != 'data':  # Don't print long path
                print(f"  {key}: {value}")
        
        # Step 6: Custom training loop with validation
        print("Step 6: Starting training with custom validation...")
        
        metrics_history = []
        
        for epoch in range(epochs):
            print(f"\n=== Epoch {epoch + 1}/{epochs} ===")
            
            # Update epoch-specific settings
            train_args = default_args.copy()
            train_args['name'] = f"train_epoch_{epoch}"
            
            # Adjust learning rate schedule if using custom scheduler
            if 'lr_scheduler' in train_kwargs:
                current_lr = self._calculate_lr(epoch, epochs, train_args['lr0'], train_args['lrf'])
                train_args['lr0'] = current_lr
                print(f"  Current learning rate: {current_lr:.6f}")
            
            # Train for one epoch
            try:
                print(f"  Training epoch {epoch + 1}...")
                results = model.train(**train_args)
                
                # Extract training metrics if available
                train_metrics = {}
                if hasattr(results, 'results_dict'):
                    train_metrics = results.results_dict
                elif hasattr(results, 'metrics'):
                    train_metrics = results.metrics
                
                print(f"  Training completed for epoch {epoch + 1}")
                
            except Exception as e:
                print(f"  Error during training epoch {epoch + 1}: {e}")
                continue
            
            # Custom validation on full images
            precision, recall, f1 = self.custom_validation(model, epoch)
            
            # Store metrics
            epoch_metrics = {
                'epoch': epoch,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            # Add training metrics if available
            if train_metrics:
                for key, value in train_metrics.items():
                    if isinstance(value, (int, float)):
                        epoch_metrics[f'train_{key}'] = value
            
            metrics_history.append(epoch_metrics)
            
            # Early stopping if target metrics achieved
            if precision >= self.target_precision and recall >= self.target_recall:
                print(f"🎯 Target metrics achieved at epoch {epoch + 1}!")
                print(f"   Precision: {precision:.4f} (target: {self.target_precision})")
                print(f"   Recall: {recall:.4f} (target: {self.target_recall})")
                break
            
            # Check for plateau and suggest hyperparameter adjustments
            if epoch > 10 and len(metrics_history) >= 5:
                recent_f1 = [m['f1'] for m in metrics_history[-5:]]
                if max(recent_f1) - min(recent_f1) < 0.01:
                    print(f"  ℹ️  F1 score plateau detected. Consider adjusting hyperparameters.")
        
        # Save metrics history
        df = pd.DataFrame(metrics_history)
        df.to_csv(self.work_dir / "training_metrics.csv", index=False)
        
        # Plot training metrics
        self.plot_metrics(df)
        
        print(f"\n🎉 Training completed!")
        print(f"   Best model saved at: {self.best_model_path}")
        print(f"   Best F1 score: {self.best_f1:.4f}")
        print(f"   Final Precision: {precision:.4f}")
        print(f"   Final Recall: {recall:.4f}")
        
        return self.best_model_path, df
    
    def _calculate_lr(self, epoch, total_epochs, lr0, lrf):
        """Calculate learning rate for custom scheduler"""
        return lr0 * (lrf/lr0) ** (epoch / total_epochs)
    
    def optimize_hyperparameters(self,
                                 model_size=model_size,
                                 n_trials=50,
                                 epochs_per_trial=20,
                                 space=None,
                                 **base_kwargs):

        print(f"🔍 Starting hyperparameter optimization with {n_trials} trials...")

        if space is None:
            # ... default space
            pass

        print(f"Search space: {json.dumps(space, indent=2)}")

        # Prepare patches if not done already
        if not (self.dataset_dir / "images" / "train").exists():
            print("Preparing dataset patches...")

            self.create_patches(self.train_images_dir, self.train_labels_dir, self.dataset_dir, "train")
            self.create_patches(self.val_images_dir, self.val_labels_dir, self.dataset_dir, "val")

        def objective(trial):
            """Optuna objective function"""
            try:
                # Sample hyperparameters from the defined space
                params = {}
                for param_name, param_range in space.items():
                    if isinstance(param_range, tuple) and len(param_range) == 2:
                        if param_name in ['mosaic', 'mixup', 'hsv_h', 'hsv_s', 'hsv_v']:
                            # Float parameters
                            params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
                        elif param_name in ['degrees', 'translate', 'scale']:
                            # Float parameters
                            params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
                        elif param_name in ['lr0', 'lrf', 'momentum', 'conf', 'iou']:
                            # Float parameters with log scale for some
                            if param_name in ['lr0', 'lrf', 'conf']:
                                params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1], log=True)
                            else:
                                params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
                        else:
                            # Default float
                            params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])

                print(f"\n🧪 Trial {trial.number + 1}/{n_trials}")
                print(f"Testing parameters: {json.dumps(params, indent=2)}")

                # Combine with base parameters
                train_args = {
                    'data': str(self.dataset_yaml_path),
                    'epochs': epochs_per_trial,
                    'batch': 16,
                    'save': False,
                    'verbose': False,
                    'project': str(self.work_dir),
                    'name': f"trial_{trial.number}",
                    'patience': epochs_per_trial,  # No early stopping within trial
                }

                # Update with base kwargs and sampled params
                train_args.update(base_kwargs)
                train_args.update(params)

                # Initialize model for this trial
                model = YOLO(model_size)

                # Train model
                results = model.train(**train_args)

                # Run custom validation on full images
                final_precision, final_recall, final_f1 = self.custom_validation(model, trial.number)

                # Store trial results
                trial_result = {
                    'trial': trial.number,
                    'params': params.copy(),
                    'precision': final_precision,
                    'recall': final_recall,
                    'f1': final_f1,
                    'target_met': final_precision >= self.target_precision and final_recall >= self.target_recall
                }
                self.optimization_results.append(trial_result)

                print(f"Trial {trial.number + 1} results:")
                print(f"  Precision: {final_precision:.4f}")
                print(f"  Recall: {final_recall:.4f}")
                print(f"  F1: {final_f1:.4f}")
                print(f"  Target met: {trial_result['target_met']}")

                # Save best model if this trial is better
                if final_f1 > self.best_f1:
                    self.best_f1 = final_f1
                    self.best_trial_params = params.copy()
                    self.best_model_path = self.models_dir / f"best_model_trial_{trial.number}.pt"
                    model.save(str(self.best_model_path))
                    print(f"  🏆 New best model saved! F1: {final_f1:.4f}")

                # Optuna maximizes the objective, so return F1 score
                # Add bonus if target metrics are met
                objective_value = final_f1
                if trial_result['target_met']:
                    objective_value += 0.1  # Bonus for meeting targets

                return objective_value

            except Exception as e:
                print(f"Trial {trial.number} failed with error: {e}")
                return 0.0  # Return poor score for failed trials

        # Create Optuna study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            study_name='yolo_hyperparameter_optimization'
        )

        # Run optimization
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value

        print(f"\n🎉 Hyperparameter optimization completed!")
        print(f"Best F1 score: {best_value:.4f}")
        print(f"Best parameters: {json.dumps(best_params, indent=2)}")

        # Save optimization results
        results_df = pd.DataFrame(self.optimization_results)
        results_df.to_csv(self.work_dir / "hyperparameter_optimization_results.csv", index=False)

        # Plot optimization results
        self.plot_optimization_results(study, results_df)

        return best_params, study
    
    def plot_optimization_results(self, study, results_df):
            """Plot optimization results"""
            if results_df.empty:
                print("No optimization results to plot. The DataFrame is empty.")
                return
    
            # Check if essential columns exist before plotting
            required_columns = ['precision', 'recall', 'f1']
            if not all(col in results_df.columns for col in required_columns):
                print(f"Missing one or more required columns ({required_columns}) in optimization results DataFrame. Available columns: {results_df.columns.tolist()}")
                return
    
            print("Generating optimization plots...")
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Optuna Hyperparameter Optimization Results', fontsize=16)
    
            # Plot 1: F1 score over trials
            trial_col = 'trial'
            if 'trial_num' in results_df.columns:
                trial_col = 'trial_num'
            axes[0, 0].plot(results_df[trial_col], results_df['f1'], marker='o')
            axes[0, 0].set_title('F1 Score per Trial')
            axes[0, 0].set_xlabel('Trial Number')
            axes[0, 0].set_ylabel('F1 Score')
            axes[0, 0].grid(True)
    
            # Plot 2: Precision vs Recall
            axes[0, 1].scatter(results_df['precision'], results_df['recall'],
                            c=results_df['f1'], cmap='viridis', s=100, alpha=0.7)
            axes[0, 1].set_title('Precision vs. Recall (Color by F1 Score)')
            axes[0, 1].set_xlabel('Precision')
            axes[0, 1].set_ylabel('Recall')
            axes[0, 1].grid(True)
            colorbar = fig.colorbar(axes[0, 1].collections[0], ax=axes[0, 1])
            colorbar.set_label('F1 Score')
    
            # Let's plot some of the parameters vs F1
            param_names = [p for p in results_df.columns if p.startswith('params_')]
            if param_names:
                # Example: Plotting first parameter vs F1
                if len(param_names) > 0:
                    param_to_plot = param_names[0]
                    axes[1, 0].scatter(results_df[param_to_plot], results_df['f1'], alpha=0.7)
                    axes[1, 0].set_title(f'{param_to_plot.replace("params_", "")} vs F1 Score')
                    axes[1, 0].set_xlabel(param_to_plot.replace("params_", ""))
                    axes[1, 0].set_ylabel('F1 Score')
                    axes[1, 0].grid(True)
                else:
                    axes[1, 0].set_visible(False) # Hide subplot if no params to plot
    
                # Example: Plotting another parameter vs F1 or another metric
                if len(param_names) > 1:
                    param_to_plot_2 = param_names[1]
                    axes[1, 1].scatter(results_df[param_to_plot_2], results_df['f1'], alpha=0.7)
                    axes[1, 1].set_title(f'{param_to_plot_2.replace("params_", "")} vs F1 Score')
                    axes[1, 1].set_xlabel(param_to_plot_2.replace("params_", ""))
                    axes[1, 1].set_ylabel('F1 Score')
                    axes[1, 1].grid(True)
                else:
                    axes[1, 1].set_visible(False) # Hide subplot if no more params
            else:
                axes[1, 0].set_visible(False)
                axes[1, 1].set_visible(False)
    
            plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to prevent title overlap
            plt.savefig(self.work_dir / "optimization_results.png")
            plt.close(fig)
            print("Optimization plots saved to 'optimization_results.png'")
    
    def train_with_best_params(self, best_params, model_size=model_size, epochs=100, **additional_kwargs):
        """
        Train model with optimized hyperparameters
        
        Args:
            best_params: Best hyperparameters from optimization
            model_size: Model size to use
            epochs: Number of epochs for final training
            **additional_kwargs: Additional training arguments
        """
        print(f"🚀 Training final model with optimized hyperparameters...")
        print(f"Using parameters: {json.dumps(best_params, indent=2)}")
        
        # Combine optimized parameters with any additional arguments
        final_params = best_params.copy()
        final_params.update(additional_kwargs)
        
        # Train with optimized parameters
        return self.train(model_size=model_size, epochs=epochs, **final_params)
    
    def plot_metrics(self, df):
        """Plot training metrics"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.plot(df['epoch'], df['precision'])
        plt.axhline(y=self.target_precision, color='r', linestyle='--', label=f'Target ({self.target_precision})')
        plt.title('Precision')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.plot(df['epoch'], df['recall'])
        plt.axhline(y=self.target_recall, color='r', linestyle='--', label=f'Target ({self.target_recall})')
        plt.title('Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        plt.plot(df['epoch'], df['f1'])
        plt.title('F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        
        plt.tight_layout()
        plt.savefig(self.work_dir / "training_metrics.png")
        plt.show()

# Example usage
if __name__ == "__main__":
    # Initialize trainer (your current initialization looks good)
    trainer = YOLOPatchTrainer(
        train_images_dir="C:/Projects/FreeZem/NeonatesCounter/dataset/final_optimization/train/images",
        train_labels_dir="C:/Projects/FreeZem/NeonatesCounter/dataset/final_optimization/train/labels",
        val_images_dir="C:/Projects/FreeZem/NeonatesCounter/dataset/final_optimization/val/images",
        val_labels_dir="C:/Projects/FreeZem/NeonatesCounter/dataset/final_optimization/val/labels",
        patch_size=640,
        stride_ratio=0.8,
        target_precision=0.99,
        target_recall=0.99,
        # REMEMBER: You need to add 'dataset_yaml_path' here based on our previous discussion
        dataset_yaml_path="C:/Projects/FreeZem/NeonatesCounter/git/NeonatesCounter/yolo_patch_work/dataset/dataset.yaml"
    )

    output_dataset_root = Path("C:/Projects/FreeZem/NeonatesCounter/git/NeonatesCounter/yolo_patch_work/dataset")

    print("--- Creating patches for TRAIN data ---")
    trainer.create_patches(
        images_dir=trainer.train_images_dir,
        labels_dir=trainer.train_labels_dir,
        output_dir=output_dataset_root,
        split_name='train'
    )

    print("\n--- Creating patches for VALIDATION data ---")
    trainer.create_patches(
        images_dir=trainer.val_images_dir,
        labels_dir=trainer.val_labels_dir,
        output_dir=output_dataset_root,
        split_name='val'
    )
    print("\n--- Patch creation complete ---")

    # Define the search space
    # space = {
    # # 1. Initial Learning Rate (lr0)
    # # This is often the most impactful hyperparameter.
    # # Start with a range that includes common defaults (e.g., 0.01 for YOLOv8)
    # # and allows for some exploration.
    # "lr0": (1e-3, 5e-2),  # From 0.001 to 0.05. Common start is 0.01.

    # # 2. Final Learning Rate Multiplier (lrf) - for cosine annealing
    # # This determines how much the learning rate decays by the end of training.
    # # It works with 'cos_lr=True' (which is often default/recommended for YOLOv8).
    # "lrf": (0.01, 0.1),   # From 1% to 10% of lr0. Common is 0.01-0.05.
    #                       # Make sure to set 'cos_lr=True' in your base parameters!

    # # 3. Weight Decay
    # # Controls L2 regularization, crucial for preventing overfitting.
    # # Important if your dataset isn't massive or if objects are complex.
    # "weight_decay": (1e-5, 1e-3), # From 0.00001 to 0.001. Common is 0.0005.
    # } 
    
    # Example 1: Hyperparameter optimization
    # best_params, study = trainer.optimize_hyperparameters(
    #     model_size= model_size,
    #     n_trials=1, # Increased for better search
    #     epochs_per_trial=100, # Increased for more stable evaluation per trial
    #     # Base parameters that remain constant
    #     batch=16,
    #     patience=50, # Set patience equal to epochs_per_trial for full runs
    #     workers=8,
    #     amp=True,
    #     cos_lr=True
        # Add any other fixed parameters you want from the 'default_args' in your train function
        # Example:
        # cos_lr=True,
        # close_mosaic=10,
        # seed=0,
        # deterministic=True,
    # )
    
    # Example 2: Train final model with best parameters
    final_model_path, final_metrics = trainer.train_with_best_params(
        best_params=model_params,
        model_size=model_size,
        epochs=100,  # Final training epochs
    )
    
    # # Example 3: Custom search space for your specific needs
    # custom_space = {
    #     # Focus on loss parameters for precision/recall
    #     "cls": (0.8, 2.0),      # Higher class loss
    #     "dfl": (2.0, 5.0),      # Higher DFL loss  
    #     "box": (10.0, 20.0),    # Much higher box loss
        
    #     # Conservative augmentation for precision
    #     "mosaic": (0.2, 0.6),   # Lower mosaic
    #     "mixup": (0.0, 0.1),    # Minimal mixup
    #     "degrees": (0.0, 2.0),  # Minimal rotation
    #     "scale": (0.0, 0.1),    # Minimal scaling
        
    #     # Detection thresholds
    #     "conf": (0.00001, 0.001), # Very low confidence
    #     "iou": (0.2, 0.5),        # Lower IoU for more detections
    # }
    
    # precision_focused_params, precision_study = trainer.optimize_hyperparameters(
    #     model_size='yolov8l.pt',
    #     n_trials=50,
    #     epochs_per_trial=20,
    #     space=custom_space
    # )
    
    # # Example 4: Quick optimization with default space
    # quick_params, quick_study = trainer.optimize_hyperparameters(
    #     model_size='yolov8n.pt',
    #     n_trials=10,
    #     epochs_per_trial=10
    #     # Uses default search space
    # )
    
    # print(f"Optimization completed. Best parameters: {best_params}")
    # print(f"Final model saved at: {final_model_path}")