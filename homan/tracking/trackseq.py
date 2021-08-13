#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=broad-except,too-many-statements,too-many-branches,logging-fstring-interpolation,import-error
import os
import warnings

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from libyana.visutils import detect2d, vizlines

from homan.datasets import verify
from homan.tracking import preprocess, trackboxes
from homan.utils.bbox import bbox_wh_to_xy
from homan.mocap import get_hand_detections


def track_sequence(images,
                   image_size,
                   hand_detector,
                   setup,
                   save_folder="tmp",
                   sample_idx=0,
                   viz=True):
    """
    Track hand and object detections in sequence of frames
    """
    # Compute detection for sequence if not already computed
    # Initialize list of boxes for each expected hand and object in the scene
    detected_boxes = {item: [] for item in setup}
    for image in tqdm(images, desc="frame"):
        # Convert image or image path to numpy array
        if isinstance(image, str):
            image = preprocess.get_image(image, image_size)
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        # Detect hands and manipulated objects using 100DOH hand-object detector
        try:
            bboxes = get_hand_detections(hand_detector, image)
            # Check if the number detected hands and objects match the expected number
            # for given dataset
            _, computed_setup, valid_bboxes = verify.check_setup(bboxes, setup)
        except AssertionError:
            # In egocentric mode, detection raises error if more then 1
            # person is detected, we ignore all detections in this case
            warnings.warn("Got more then 1 person")
        # Check if detections match dataset setup
        for item, expected_bbox_nb in setup.items():
            # Only consider detections when the number detected hand or object
            # matches the expected number of detections (typically 0 or 1)
            if (item in computed_setup) and (expected_bbox_nb
                                             == computed_setup[item]):
                if item == "objects":
                    detected_boxes[item].append(valid_bboxes[item][0])
                else:
                    detected_boxes[item].append(valid_bboxes[item])
            else:
                detected_boxes[item].append(None)
    tracked_boxes = {}

    # Display tracked boxes examples for each video
    os.makedirs(save_folder, exist_ok=True)
    img_save_path = os.path.join(save_folder, f"tmptrack{sample_idx}.png")

    show_nb = 5
    if viz:
        fig, axes = plt.subplots(2, show_nb, figsize=(4 * show_nb, 8))
    frame_idxs = np.linspace(0, len(images) - 1, show_nb).astype(np.int)
    for show_idx, frame_idx in enumerate(frame_idxs):
        image = images[frame_idx]
        if isinstance(image, str):
            image = preprocess.get_image(image, image_size)
        axis = axes[0, show_idx]
        axis.imshow(image)
        axis.axis("off")
        axis = axes[1, show_idx]
        axis.imshow(image)
        axis.axis("off")

    # Perform tracking
    for item in setup:
        # Perform tracking in forward and backward time directions
        detected_boxes_wh = detected_boxes[item]
        boxes_fwd = trackboxes.track_boxes(detected_boxes_wh, out_xyxy=True)
        boxes_bwd = trackboxes.track_boxes(detected_boxes_wh[::-1],
                                           out_xyxy=True)[::-1]

        # Average predictions in both directions to get more
        # robust tracks
        tracked_boxes[item] = (boxes_fwd + boxes_bwd) / 2

        if viz:
            # Visualize tracked and detected hand+object bboxes
            for show_idx, frame_idx in enumerate(frame_idxs):
                # Display detected boxes
                orig_box = detected_boxes[item][frame_idx]
                axis = axes[0, show_idx]
                if orig_box is not None:
                    detect2d.visualize_bbox(axis,
                                            bbox_wh_to_xy(orig_box),
                                            label=item)
                if show_idx == show_nb // 2:
                    axis.set_title("Detected boxes")

                # Display tracked boxes
                axis = axes[1, show_idx]
                if show_idx == show_nb // 2:
                    axis.set_title("Tracked boxes")
                detect2d.visualize_bbox(axis,
                                        tracked_boxes[item][frame_idx],
                                        label=item)
    if viz:
        fig.savefig(img_save_path)

    # Show x, y evolution for detections before and after tracking
    if viz:
        fig, axes = plt.subplots(1, len(setup))
        for idx, item in enumerate(setup):
            axis = axes[idx]
            axis.set_title(item)
            inp_boxes = np.array([
                bbox_wh_to_xy(box).tolist()
                if box is not None else [None, None, None, None]
                for box in detected_boxes[item]
            ])
            vizlines.add_lines(axis,
                               tracked_boxes[item].transpose(),
                               over_lines=inp_boxes.transpose(),
                               labels=["x_min", "y_min", "x_max", "y_max"])
            axis.set_xlabel("frame")
            if idx == 0:
                axis.set_ylabel("bbox xy pixel locations")
        img_save_path = os.path.join(save_folder, f"tmptrackl{sample_idx}.png")
        fig.savefig(img_save_path)
    return tracked_boxes
