import os
import argparse
import json
from skimage.filters import threshold_otsu
import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import sys
import h5py
from inv.segmentation.utils import (
    resize_segments_np,
    compute_segment_centroids,
    offset_single_centroid,
    get_baseline_metrics,
    plot_segments
)
from pycocotools import mask as mask_util

# SAM2 imports
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class SAM2_Multimask:
    """
    SAM2 Class
    """
    def __init__(self, device='cuda'):
        # checkpoint = '../../../cwm_mono/cwm/segment/external/SAM2/checkpoints/sam2.1_hiera_large.pt'
        checkpoint = '../checkpoints/sam2.1_hiera_large.pt'
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
        self.device = device

    def segment_at_points(self, image: np.ndarray, points: np.ndarray, multimask=True):
        """
        image: H×W×3 uint8
        points: N×2 float, (x,y) pixel coordinates of positives
        returns: HxW (pick only the highest scored mask)
        """
        self.predictor.set_image(image)

        # label=1 for all as positive point
        pt = [points[0][0], points[0][1]]
        masks, scores, logits = self.predictor.predict(
            point_coords=[pt],
            point_labels=np.array([1,]),
            multimask_output=multimask,
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        # logits = logits[sorted_ind]

        return masks[np.argmax(scores), :, :], pt

def main(args, h5_save_file):
    # Build SAM2 wrapper
    sam_wrapper = SAM2_Multimask(device=args.device)

    os.makedirs(os.path.dirname(h5_save_file), exist_ok=True)

    # Collect all .png images from directory
    dataset = h5py.File(args.annotations_h5, 'r')

    with h5py.File(h5_save_file, "w") as h5f:
        for img_idx, img_name in enumerate(dataset.keys()):
            print(f"Processing {img_name}...")
            # Load image
            image = dataset[img_name]
            I = image['rgb'][:]

            # Load GT segments and centroids
            gt_segs = image['segment'][:]
            centroids = compute_segment_centroids(torch.tensor(gt_segs))

            # Create group for this image
            img_grp = h5f.create_group(f"img{img_idx}")
            img_grp.create_dataset("image_rgb", data=I, compression="gzip")
            img_grp.create_dataset("segments_gt", data=gt_segs, compression="gzip")

            for si, cent in enumerate(centroids):
                offsets = offset_single_centroid(
                    cent, N=args.num_offset_points,
                    min_mag=args.min_mag, max_mag=args.max_mag
                )
                seg_grp = img_grp.create_group(f"seg{si}")
                for pi, pt in enumerate(offsets):
                    mask, pt_mv = sam_wrapper.segment_at_points(I, np.array([pt]))
                    pt_grp = seg_grp.create_group(f"pt{pi}")
                    pt_grp.create_dataset("segment", data=mask, compression="gzip")
                    pt_grp.create_dataset("centroid", data=pt_mv, compression="gzip")

    print(f"Done writing all segments to '{h5_save_file}'.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # dataset
    p.add_argument("--save_dir", default='./test_vis_pointseg/')

    p.add_argument("--annotations_h5", default='/ccn2/u/lilianch/data/entityseg_100.h5')

    # test
    p.add_argument("--test", type=str,  nargs='+',
                        choices=['h5', 'vis', 'metrics'], default =['h5', 'vis', 'metrics'],
                        help='h5 for saving segments, vis for visualizing, metrics to save/print metrics')

    # vis test and metrics test: /ccn2/u/lilianch/data/segments.h5
    p.add_argument("--h5_file", default='./segments.h5',
                   help='location to retrieve segments h5 file (assumes existing)')

    # misc args
    p.add_argument("--num_offset_points", type=int, default=1)
    p.add_argument("--min_mag", type=float, default=10.0)
    p.add_argument("--max_mag", type=float, default=25.0)
    p.add_argument("--device", default="cuda:0")

    args = p.parse_args()

    save_dir = args.save_dir
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)

    h5_save_file = os.path.join(save_dir, 'segments.h5')
    if 'h5' in args.test:
        if os.path.exists(h5_save_file):
            response = input(f"File '{h5_save_file}' already exists. Overwrite? (y/[n]): ").strip().lower()
            if response != 'y':
                print("Aborting.")
                sys.exit(0)
        main(args, h5_save_file)

    if 'vis' in args.test:
        assert os.path.exists(h5_save_file), f"Expected file '{h5_save_file}' to exist, but it was not found. Make sure the file is generated or the correct path is provided."
        img_save_dir = os.path.join(save_dir, 'vis/')
        os.makedirs(os.path.dirname(img_save_dir), exist_ok=True)
        plot_segments(h5_save_file, img_save_dir)

    if 'metrics' in args.test:
        assert os.path.exists(h5_save_file), f"Expected file '{h5_save_file}' to exist, but it was not found. Make sure the file is generated or the correct path is provided."
        metrics_save_json = os.path.join(save_dir, 'metrics.json')
        os.makedirs(os.path.dirname(metrics_save_json), exist_ok=True)
        if not os.path.exists(metrics_save_json):
            with open(metrics_save_json, 'w') as f:
                pass
        get_baseline_metrics(h5_save_file, metrics_save_json)
