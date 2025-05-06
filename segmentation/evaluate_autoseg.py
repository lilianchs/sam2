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
    offset_single_centroid,
    get_baseline_metrics_autoseg,
    plot_segments_autoseg
)
from pycocotools import mask as mask_util

# SAM2 imports
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

class SAM2_Autoseg:
    """
    SAM2 Class
    """
    def __init__(self, device='cuda'):
        checkpoint = '../checkpoints/sam2.1_hiera_large.pt'
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.predictor = SAM2AutomaticMaskGenerator(build_sam2(model_cfg, checkpoint, device=device, apply_postprocessing=False))
        self.device = device

    def segment_at_points(self, image: np.ndarray):
        """
        image: H×W×3 uint8
        points: N×2 float, (x,y) pixel coordinates of positives
        returns: HxW (pick only the highest scored mask)
        """
        masks = self.predictor.generate(image)
        return masks

def main(args, h5_save_file):
    # Build SAM2 wrapper
    sam_wrapper = SAM2_Autoseg(device=args.device)

    os.makedirs(os.path.dirname(h5_save_file), exist_ok=True)

    # Collect all .png images from directory
    dataset = h5py.File(args.annotations_h5, 'r')

    with h5py.File(h5_save_file, "w") as h5f:
        for img_idx, img_name in enumerate(dataset.keys()):
            print(f"Processing {img_name}...")
            # Load image
            image = dataset[img_name]
            I = image['rgb'][:]

            # Load GT segments
            gt_segs = image['segment'][:]

            # Create group for this image
            img_grp = h5f.create_group(f"img{img_idx}")
            img_grp.create_dataset("image_rgb", data=I, compression="gzip")
            img_grp.create_dataset("segments_gt", data=gt_segs, compression="gzip")

            masks = sam_wrapper.segment_at_points(I)
            masks = np.stack([m['segmentation'] for m in masks])
            img_grp.create_dataset("segments_pred", data=masks, compression="gzip")

    print(f"Done writing all segments to '{h5_save_file}'.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # dataset
    p.add_argument("--save_dir", default='./test_vis_autoseg/')

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

        plot_segments_autoseg(h5_save_file, img_save_dir)

    if 'metrics' in args.test:
        assert os.path.exists(h5_save_file), f"Expected file '{h5_save_file}' to exist, but it was not found. Make sure the file is generated or the correct path is provided."
        metrics_save_json = os.path.join(save_dir, 'metrics.json')
        os.makedirs(os.path.dirname(metrics_save_json), exist_ok=True)
        if not os.path.exists(metrics_save_json):
            with open(metrics_save_json, 'w') as f:
                pass
        metrics = get_baseline_metrics_autoseg(h5_save_file, metrics_save_json)
