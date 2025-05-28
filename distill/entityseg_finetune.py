import numpy as np
import torch
import cv2
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import json
from pycocotools import mask as mask_util

class EntitySegDataset(Dataset):
    def __init__(self, data, max_objects_per_image=20):
        self.data = data
        self.max_objects = max_objects_per_image

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ent = self.data[idx]

        # Read image
        Img = cv2.imread(ent["image"]) #[..., ::-1]  # read image (BGR to RGB)

        # Resize image
        r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]])  # scaling factor
        new_h, new_w = int(Img.shape[0] * r), int(Img.shape[1] * r)
        Img = cv2.resize(Img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Process EntitySeg annotations (not LabPics format!)
        annotations = ent["annotation"]  # This is a list of COCO-style annotations

        # Limit number of objects to prevent memory issues
        if len(annotations) > self.max_objects:
            annotations = np.random.choice(annotations, self.max_objects, replace=False).tolist()

        points = []
        masks = []

        for annot in annotations:
            # Decode COCO RLE mask
            mask = mask_util.decode(annot['segmentation'])

            # Resize mask to match resized image
            mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_AREA)

            masks.append(mask)

            # Get random point from mask
            coords = np.argwhere(mask > 0)
            if len(coords) > 0:
                # Use center point
                center_y, center_x = coords.mean(axis=0)
                center_point = [int(center_x), int(center_y)]

                # Ensure point is still within mask
                if mask[center_point[1], center_point[0]] > 0:
                    points.append(center_point)
                else:
                    yx = np.array(coords[np.random.randint(len(coords))])  # Fallback to center
                    points.append([yx[1], yx[0]])

        return {
            'image': Img,
            'masks': np.array(masks) if masks else np.zeros((0, new_h, new_w), dtype=np.uint8),
            'points': np.array(points) if points else np.zeros((0, 2)),
            'labels': np.ones(len(masks)) if masks else np.zeros(0)
        }


def collate_fn(batch):
    """Custom collate function to handle variable number of objects per image"""
    # Process one image at a time for SAM2 (it doesn't support true batching)
    return batch[0]  # Return single item

def filter_annotations(image_id, annotations, categories):
    annotations_image_id = [x for x in annotations if x['image_id'] == image_id]
    all_segs = []
    image_area = annotations_image_id[0]['segmentation']['size'][0] * annotations_image_id[0]['segmentation']['size'][1]

    for annot in annotations_image_id:
        id = (annot['iscrowd'], annot['area'], annot['image_id'], annot['category_id'], annot['attribute'], annot['id'])
        seg_mask = mask_util.decode(annot['segmentation'])
        category_id = annot['category_id']

        for category in categories:
            if category['id'] == category_id:
                if category['type'] == 'thing' and category['supercategory'] not in ['table', 'cabinets',
                                                                                     'kitchen_appliances',
                                                                                     'traffic_facility', 'stove'] and category['name'] not in ['kitchen_sink', 'billboard', 'blackboard', 'signboard', 'guideboard', 'rug', 'curtain']:
                    if (seg_mask.sum() / image_area) * 100 > 0.5:
                        all_segs.append(annot)

    return all_segs


def train_sam2_batch():
    # Setup data
    data = []  # list of files in dataset

    ann_path = "/ccn2/u/lilianch/data/entityseg/entityseg_train_01.json"
    images_dir = "/ccn2/u/rmvenkat/data/entity_seg_dataset/entity_seg_images/entity_01_11580"

    entity_seg = json.load(open(ann_path))
    annotations = entity_seg['annotations']
    categories = entity_seg['categories']

    data_json_path = "/ccn2/u/lilianch/external_repos/sam2/distill/preprocessed_entityseg_data.json"

    if os.path.exists(data_json_path):
        print(f"Loading preprocessed dataset from {data_json_path}")
        with open(data_json_path, 'r') as f:
            data = json.load(f)
    else:
        for ct, image_data in enumerate(entity_seg['images']):
            filtered_annotations = filter_annotations(image_data['id'], annotations, categories)
            if len(filtered_annotations) == 0:
                continue
            data.append({
                "image": os.path.join(images_dir, image_data['file_name']),
                "annotation": filtered_annotations
            })
            print(f'Done with ct: {ct}')
        print(f'Done loading dataset of size: {len(data)}')

        with open(data_json_path, 'w') as f:
            json.dump(data, f)
        print(f"Saved preprocessed dataset to {data_json_path}")

    # Create dataset and dataloader
    dataset = EntitySegDataset(data, max_objects_per_image=15)  # Limit objects to manage memory
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=0)

    # Setup model
    sam2_checkpoint = '../checkpoints/sam2.1_hiera_large.pt'
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")  # load model
    predictor = SAM2ImagePredictor(sam2_model)  # load net
    print(f'loaded SAM2 predictor...')

    # Configure training
    predictor.model.image_encoder.requires_grad_(False)  # Freeze image encoder
    predictor.model.sam_mask_decoder.train(True)  # enable training of mask decoder
    predictor.model.sam_prompt_encoder.train(True)  # enable training of prompt encoder

    # Setup optimizer for only trainable parameters
    trainable_params = []
    for param in predictor.model.sam_mask_decoder.parameters():
        if param.requires_grad:
            trainable_params.append(param)
    for param in predictor.model.sam_prompt_encoder.parameters():
        if param.requires_grad:
            trainable_params.append(param)

    optimizer = torch.optim.AdamW(params=trainable_params, lr=1e-5, weight_decay=4e-5)
    scaler = torch.cuda.amp.GradScaler()  # set mixed precision

    # Training loop
    mean_iou = 0
    step = 0

    for epoch in range(100):  # Run for multiple epochs
        for batch_idx, batch in enumerate(dataloader):
            if len(batch['masks']) == 0:  # Skip empty batches
                continue

            with torch.cuda.amp.autocast():  # cast to mixed precision
                image = batch['image']
                masks = batch['masks']
                points = batch['points']
                labels = batch['labels']

                # Reshape points for SAM2 format
                input_points = points.reshape(-1, 1, 2)  # [num_objects, 1, 2]
                input_labels = labels.reshape(-1, 1)     # [num_objects, 1]

                predictor.set_image(image)  # apply SAM image encoder to the image

                # Prepare prompts
                mask_input, unnorm_coords, proc_labels, unnorm_box = predictor._prep_prompts(
                    input_points, input_labels, box=None, mask_logits=None, normalize_coords=True
                )

                # Get embeddings
                sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                    points=(unnorm_coords, proc_labels), boxes=None, masks=None
                )

                # Predict masks
                batched_mode = unnorm_coords.shape[0] > 1  # multi mask prediction
                high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]

                low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                    image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                    image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=True,
                    repeat_image=batched_mode,
                    high_res_features=high_res_features,
                )

                prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])

            # Calculate losses
            prd_mask = torch.sigmoid(prd_masks[:, 0])  # Turn logit map to probability map
            gt_mask = torch.from_numpy(masks).float().to(prd_mask.device)  # Fixed tensor conversion

            # Cross entropy loss
            seg_loss = (-gt_mask * torch.log(prd_mask + 1e-8) -
                       (1 - gt_mask) * torch.log((1 - prd_mask) + 1e-8)).mean()

            # IoU calculation and score loss
            inter = (gt_mask * (prd_mask > 0.5)).sum(dim=(1,2))
            union = gt_mask.sum(dim=(1,2)) + (prd_mask > 0.5).sum(dim=(1,2)) - inter
            iou = inter / (union + 1e-8)  # Add epsilon to prevent division by zero
            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()

            loss = seg_loss + score_loss * 0.05  # mix losses

            # Backward pass
            optimizer.zero_grad()  # Use optimizer.zero_grad() instead
            scaler.scale(loss).backward()  # Backpropagate
            scaler.step(optimizer)
            scaler.update()  # Mixed precision

            # Logging and saving
            if step % 100 == 0:
                current_iou = iou.mean().cpu().detach().numpy()
                mean_iou = mean_iou * 0.99 + 0.01 * current_iou
                print(f"Step {step}, Epoch {epoch}, Loss: {loss.item():.4f}, Mean IoU: {mean_iou:.4f}")

            if step % 1000 == 0 and step > 0:
                torch.save(predictor.model.state_dict(), f"model_step_{step}.torch")
                print(f"Model saved at step {step}")

            # Memory cleanup every 50 steps
            if step % 50 == 0:
                torch.cuda.empty_cache()

            step += 1

            if step >= 100000:  # Your original stopping condition
                break

        if step >= 100000:
            break

    # Final save
    final_path = "/ccn2/u/lilianch/external_repos/sam2/distill/entityseg_finetune_100000_steps.torch"
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    torch.save(predictor.model.state_dict(), final_path)
    print("Training completed!")

if __name__ == "__main__":
    train_sam2_batch()