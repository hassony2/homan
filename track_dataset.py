#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=broad-except,too-many-statements,too-many-branches,logging-fstring-interpolation,import-error
"""
Track hand and object bounding boxes in a video clip, this allows to collect the correct
number of candidate detections in each frame.
Step 1: Detection of hand and object bboxes using 100DOH, Shan et al., CVPR2020 hand-object detector
Step 2: Tracking using simple Kallman filetering implementation from motpy, this allows to recover
    missed detections through interpolation and also results in some level of smoothing

Launch for instance with:
`python track_dataset.py --dataset ho3d --split test --only_missing`
"""
import argparse
import os
import pickle

from tqdm import tqdm

from homan.mocap import get_hand_bbox_detector
from homan.datasets.epic import Epic
from homan.datasets.core50 import Core50
from homan.datasets.ho3d import HO3D
from homan.tracking import trackseq


def get_args():
    parser = argparse.ArgumentParser(
        description="Optimize object meshes w.r.t. human.")
    parser.add_argument("--dataset",
                        default="ho3d",
                        choices=[
                            "ho3d",
                            "core50",
                            "epic",
                        ],
                        help="Dataset name")
    parser.add_argument("--split", default="test")
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--only_missing",
                        action="store_true",
                        help="Resume tracking for missing sequences")
    parser.add_argument(
        "--boxes_folder",
        default="data/boxes",
        help="Location where to save the tracked bounding boxes")
    parser.add_argument("--save_folder", default="tracks")
    args = parser.parse_args()
    return args


def main(args):
    os.makedirs(args.boxes_folder, exist_ok=True)
    # Load the target dataset
    if args.dataset == "ho3d":
        dataset = HO3D(
            split=args.split,
            use_cache=args.use_cache,
            # Options to track frames across the full video sequence
            mode="vid",
            frame_nb=-1,
            track=True)
        image_size = 640
    elif args.dataset == "core50":
        dataset = Core50(
            # objects=[ args.split, ],
            objects=None,
            mode="vid",
            frame_nb=-1,
            use_cache=args.use_cache,
            track=True,
        )
        image_size = 350
    elif args.dataset == "epic":
        dataset = Epic(mode="vid", frame_nb=-1, use_cache=args.use_cache)
    else:
        raise ValueError(f"{args.dataset} not in ['core50','epic','ho3d']")
    hand_detector = get_hand_bbox_detector()
    print(f"Processing dataset of size {len(dataset)}")

    save_path = os.path.join(args.boxes_folder,
                             f"boxes_{args.dataset}_{args.split}.pkl")
    all_boxes = {}
    if args.only_missing:
        with open(save_path, "rb") as p_f:
            all_boxes = pickle.load(p_f)

    print(f"Saving tracking results to {args.save_folder}")

    for sample_idx in tqdm(range(0, len(dataset)), desc="video"):
        annots = dataset[sample_idx]
        images = annots['images']
        seq_idx = annots["seq_idx"]
        setup = annots["setup"]
        # Compute detection for sequence if not already computed
        if seq_idx not in all_boxes:
            seq_boxes = trackseq.track_sequence(
                images,
                image_size,
                hand_detector=hand_detector,
                setup=setup,
                sample_idx=sample_idx,
                save_folder=args.save_folder,
            )

            all_boxes[seq_idx] = seq_boxes

        # Update tracking results in pickle file
        with open(save_path, "wb") as p_f:
            pickle.dump(all_boxes, p_f)


if __name__ == "__main__":
    main(get_args())
