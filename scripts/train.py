import os
import cv2
import numpy as np
import torch
from utils.utils import ImagePatcher, ImageRepatcher, ModelTrain, apply_nms, load_ground_truth, calculate_precision_recall

def draw_boxes(image, boxes, color=(0,255,0), alpha=0.5, thickness=1):
    overlay = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
    # Blend with transparency
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    return image

def main():
    imgs_folder = 'path/to/images'
    annotation_file = 'path/to/annotations.json'  # COCO format assumed
    output_folder = 'path/to/output_predictions'
    os.makedirs(output_folder, exist_ok=True)

    patch_size = 640
    stride = 576

    patcher = ImagePatcher(patch_size=patch_size, stride=stride)
    repatcher = ImageRepatcher(patch_size=patch_size, stride=stride)
    trainer = ModelTrain(model_path=None, imgsz=patch_size, device='cpu')  # load default yolov8n or specify your model

    gt_dict = load_ground_truth(annotation_file)

    precision_list = []
    recall_list = []

    for img_fname in os.listdir(imgs_folder):
        if not img_fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        img_path = os.path.join(imgs_folder, img_fname)
        image = cv2.imread(img_path)
        original_shape = image.shape

        # Pad image to multiples of stride
        padded_img = patcher.pad_image(image)

        # Create patches
        patches = patcher.create_patches(padded_img)  # shape (num_y, num_x, 1, h, w, 3)

        all_boxes = []
        all_scores = []

        # Predict on each patch and map boxes to full image coords
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                patch_img = patches[i, j, 0]
                boxes_patch, scores_patch = trainer.infer_on_patch(patch_img)
                if boxes_patch.shape[0] == 0:
                    continue
                # Shift boxes by patch offset to full image coords
                x_offset = j * stride
                y_offset = i * stride
                boxes_patch[:, 0] += x_offset
                boxes_patch[:, 1] += y_offset
                boxes_patch[:, 2] += x_offset
                boxes_patch[:, 3] += y_offset

                all_boxes.append(boxes_patch)
                all_scores.append(scores_patch)

        if len(all_boxes) == 0:
            print(f"No boxes predicted for {img_fname}")
            continue

        all_boxes = torch.cat(all_boxes)
        all_scores = torch.cat(all_scores)

        # Apply NMS on full image boxes
        keep_idx = apply_nms(all_boxes, all_scores, iou_threshold=0.5)
        boxes_nms = all_boxes[keep_idx].cpu().numpy()

        # Clip boxes to original image size
        boxes_nms[:, [0,2]] = np.clip(boxes_nms[:, [0,2]], 0, original_shape[1])
        boxes_nms[:, [1,3]] = np.clip(boxes_nms[:, [1,3]], 0, original_shape[0])

        # Load ground truth boxes for this image
        gt_boxes = gt_dict.get(img_fname, np.zeros((0,4)))

        # Calculate precision and recall for this image
        precision, recall = calculate_precision_recall(boxes_nms, gt_boxes)
        precision_list.append(precision)
        recall_list.append(recall)

        print(f"{img_fname} - Precision: {precision:.3f}, Recall: {recall:.3f}")

        # Visualize boxes on original image
        vis_img = image.copy()
        vis_img = draw_boxes(vis_img, boxes_nms, color=(0,255,0), alpha=0.5, thickness=1)

        output_path = os.path.join(output_folder, img_fname)
        cv2.imwrite(output_path, vis_img)

    # Average precision and recall across dataset
    avg_precision = np.mean(precision_list) if precision_list else 0.0
    avg_recall = np.mean(recall_list) if recall_list else 0.0
    print(f"Average Precision: {avg_precision:.4f}, Average Recall: {avg_recall:.4f}")

if __name__ == "__main__":
    main()
