import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from pycocotools import mask as mask_util
import json

BRIGHT_COLORS = [
    (255, 0, 0),       # Red
    (0, 255, 0),       # Lime
    (0, 0, 255),       # Blue
    (255, 255, 0),     # Yellow
    (255, 0, 255),     # Magenta
    (0, 255, 255),     # Cyan
    (255, 165, 0),     # Orange
    (255, 105, 180),   # Hot Pink
    (0, 255, 127),     # Spring Green
    (30, 144, 255),    # Dodger Blue
    (255, 20, 147),    # Deep Pink
    (0, 191, 255),     # Deep Sky Blue
    (173, 255, 47),    # Green Yellow
    (255, 69, 0),      # Red-Orange
    (124, 252, 0),     # Lawn Green
]

def load_original_sam2():
    """Load the original SAM2 model"""
    checkpoint = '../checkpoints/sam2.1_hiera_large.pt'
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_model = build_sam2(model_cfg, checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)
    return predictor


def load_finetuned_sam2(checkpoint_path):
    """Load fine-tuned SAM2 model"""
    # load original architecture
    checkpoint = '../checkpoints/sam2.1_hiera_large.pt'
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_model = build_sam2(model_cfg, checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)

    # load fine-tuned weights
    finetuned_state_dict = torch.load(checkpoint_path, map_location='cuda')
    predictor.model.load_state_dict(finetuned_state_dict)

    return predictor


def predict_masks(predictor, image, points, labels):
    """Get mask predictions from a SAM2 predictor"""
    predictor.set_image(image)

    # Predict masks
    masks, scores, logits = predictor.predict(
        point_coords=points,
        point_labels=labels,
        multimask_output=True,
    )

    return masks, scores, logits


def calculate_metrics(pred_mask, gt_mask):
    """Calculate IoU and other metrics"""
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()

    iou = intersection / (union + 1e-8)

    # Dice coefficient
    dice = 2 * intersection / (pred_mask.sum() + gt_mask.sum() + 1e-8)

    # Precision and Recall
    precision = intersection / (pred_mask.sum() + 1e-8)
    recall = intersection / (gt_mask.sum() + 1e-8)

    return {
        'iou': iou,
        'dice': dice,
        'precision': precision,
        'recall': recall
    }


def visualize_comparison(image, gt_masks, orig_masks, ft_masks, points, save_path=None):
    """Visualize the comparison between original and fine-tuned predictions"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Generate distinct colors for each segment
    import matplotlib.cm as cm
    import numpy as np
    n_segments = len(gt_masks)
    cmap = cm.get_cmap('tab10', n_segments)

    # unique_colors = [tuple((np.array(cmap(i)[:3]) * 255).astype(np.uint8)) for i in range(n_segments)]
    unique_colors = [BRIGHT_COLORS[i % len(BRIGHT_COLORS)] for i in range(n_segments)]

    # Original image with points
    axes[0, 0].imshow(image)
    for point in points:
        axes[0, 0].scatter(point[0], point[1], c='red', s=200, marker='*')
    axes[0, 0].set_title('Original Image + Points')
    axes[0, 0].axis('off')

    # Ground truth
    axes[0, 1].imshow(image)
    gt_overlay = np.zeros_like(image, dtype=np.float32)
    for idx, segment in enumerate(gt_masks):
        color = np.array(unique_colors[idx]) / 255.0  # Normalize to [0,1]
        for channel in range(3):
            gt_overlay[:, :, channel] += segment.astype(np.float32) * color[channel]
    # Clip values to avoid overflow
    gt_overlay = np.clip(gt_overlay, 0, 1.0)
    axes[0, 1].imshow(gt_overlay, alpha=0.5)
    axes[0, 1].set_title('Ground Truth')
    axes[0, 1].axis('off')

    # Original SAM2
    axes[1, 0].imshow(image)
    overlay = np.zeros_like(image, dtype=np.float32)
    for i, mask in enumerate(orig_masks):
        color = np.array(unique_colors[i]) / 255.0  # Normalize
        mask_bool = mask.astype(bool)
        for c in range(3):
            overlay[:, :, c] += mask_bool * color[c]
        axes[1, 0].contour(mask_bool.astype(float), colors=[color], linewidths=2)
    overlay = np.clip(overlay, 0, 1.0)
    axes[1, 0].imshow(overlay, alpha=0.7)
    axes[1, 0].set_title('Original SAM2')
    axes[1, 0].axis('off')

    # Fine-tuned SAM2
    axes[1, 1].imshow(image)
    overlay = np.zeros_like(image, dtype=np.float32)
    for i, mask in enumerate(ft_masks):
        color = np.array(unique_colors[i]) / 255.0  # Normalize
        mask_bool = mask.astype(bool)
        for c in range(3):
            overlay[:, :, c] += mask_bool * color[c]
        axes[1, 1].contour(mask_bool.astype(float), colors=[color], linewidths=2)
    overlay = np.clip(overlay, 0, 1.0)
    axes[1, 1].imshow(overlay, alpha=0.7)
    axes[1, 1].set_title('Fine-tuned SAM2')
    axes[1, 1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(f'/ccn2/u/lilianch/external_repos/sam2/distill/vis/{save_path}')

def evaluate_on_entityseg_sample(original_predictor, finetuned_predictor, data_sample):
    """Evaluate both models on a single EntitySeg sample"""
    # Load image
    image = cv2.imread(data_sample["image"])[..., ::-1]  # BGR to RGB

    # Resize image (same as training)
    r = np.min([1024 / image.shape[1], 1024 / image.shape[0]])
    new_h, new_w = int(image.shape[0] * r), int(image.shape[1] * r)
    image = cv2.resize(image, (new_w, new_h))

    # Process annotations
    annotations = data_sample["annotation"]
    gt_masks = []
    points = []

    for annot in annotations:
        # Decode and resize mask
        mask = mask_util.decode(annot['segmentation'])
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        gt_masks.append(mask)
        # Get center point
        coords = np.argwhere(mask > 0)
        if len(coords) > 0:
            # Use center point + small random offset
            center_y, center_x = coords.mean(axis=0)
            center_point = [int(center_x), int(center_y)]
            # Ensure point is still within mask
            if mask[center_point[1], center_point[0]] > 0:
                points.append(center_point)
            else:
                yx = np.array(coords[np.random.randint(len(coords))])  # Fallback to center
                points.append([yx[1], yx[0]])

    points = np.array(points)
    labels = np.ones(len(points))

    # Get predictions for each segment individually
    orig_best_masks = []
    ft_best_masks = []
    orig_best_scores = []
    ft_best_scores = []

    for i, point in enumerate(points):
        # Predict masks for this single point
        orig_masks, orig_scores, _ = predict_masks(original_predictor, image, [point], [1])
        ft_masks, ft_scores, _ = predict_masks(finetuned_predictor, image, [point], [1])

        # Select best mask for this segment (highest confidence)
        orig_best_idx = np.argmax(orig_scores)  # Choose from 3 masks
        ft_best_idx = np.argmax(ft_scores)

        orig_best_masks.append(orig_masks[orig_best_idx])
        ft_best_masks.append(ft_masks[ft_best_idx])
        orig_best_scores.append(orig_scores[orig_best_idx])
        ft_best_scores.append(ft_scores[ft_best_idx])

    # Calculate metrics
    orig_metrics = []
    ft_metrics = []
    for i, gt_mask in enumerate(gt_masks):
        if i < len(orig_best_masks):
            orig_metrics.append(calculate_metrics(orig_best_masks[i], gt_mask))
        if i < len(ft_best_masks):
            ft_metrics.append(calculate_metrics(ft_best_masks[i], gt_mask))

    return {
        'image': image,
        'gt_masks': gt_masks,
        'orig_masks': orig_best_masks,
        'ft_masks': ft_best_masks,
        'points': points,
        'orig_metrics': orig_metrics,
        'ft_metrics': ft_metrics,
        'orig_scores': orig_best_scores,
        'ft_scores': ft_best_scores
    }


def run_comparison():
    """Main function to run the comparison"""

    # Load models
    print("Loading original SAM2...")
    original_predictor = load_original_sam2()

    print("Loading fine-tuned SAM2...")
    finetuned_checkpoint = '/ccn2/u/lilianch/external_repos/sam2/distill/model_step_20000.torch'
    finetuned_predictor = load_finetuned_sam2(finetuned_checkpoint)

    # Load your preprocessed EntitySeg data
    data_json_path = '/ccn2/u/lilianch/external_repos/sam2/distill/entityseg_test_set.json'
    with open(data_json_path, 'r') as f:
        data = json.load(f)

    # Test on a few samples
    num_samples = 200
    all_orig_ious = []
    all_ft_ious = []

    for i in range(min(num_samples, len(data))):
        print(f"\nEvaluating sample {i + 1}/{num_samples}")

        try:
            results = evaluate_on_entityseg_sample(
                original_predictor,
                finetuned_predictor,
                data[i]
            )

            # Extract IoU scores
            orig_ious = [m['iou'] for m in results['orig_metrics']]
            ft_ious = [m['iou'] for m in results['ft_metrics']]

            all_orig_ious.extend(orig_ious)
            all_ft_ious.extend(ft_ious)

            # Print results for this sample
            print(f"Original SAM2 - Mean IoU: {np.mean(orig_ious):.3f}")
            print(f"Fine-tuned SAM2 - Mean IoU: {np.mean(ft_ious):.3f}")

            # Visualize the first few samples
            visualize_comparison(
                    results['image'],
                    results['gt_masks'],
                    results['orig_masks'],
                    results['ft_masks'],
                    results['points'],
                    save_path=f'comparison_sample_{i}.png'
            )

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue

    # Overall comparison
    print(f"\n{'=' * 50}")
    print("OVERALL RESULTS:")
    print(f"Original SAM2 - Mean IoU: {np.mean(all_orig_ious):.3f} ± {np.std(all_orig_ious):.3f}")
    print(f"Fine-tuned SAM2 - Mean IoU: {np.mean(all_ft_ious):.3f} ± {np.std(all_ft_ious):.3f}")
    print(f"Improvement: {np.mean(all_ft_ious) - np.mean(all_orig_ious):.3f}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    run_comparison()