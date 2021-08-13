#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import torch

from detectron2.structures import BitMasks

from libyana.visutils import imagify
from libyana.conversions import npt
from libyana.lib3d import kcrop

from homan.utils.bbox import bbox_wh_to_xy, make_bbox_square, bbox_xy_to_wh
from homan.constants import REND_SIZE


def add_occlusions(masks, occluder_mask, mask_bboxes):
    """
    Args:
        masks (list[np.ndarray]): list of object masks in [REND_SIZE, REND_SIZE] [(REND_SIZE, REND_SIZE), ...]
        mask_bboxes (list[np.ndarray]): matching list of square xy_wh bboxes [(4,), ...]
        occluder_mask (torch.Tensor): [B, IMAGE_SIZE, IMAGE_SIZE] occluder where B
            dim aggregates different one-hot encodings of occluder
            masks
    """
    occluded_masks = []
    for mask, mask_bbox in zip(masks, mask_bboxes):
        bbox_mask = bbox_wh_to_xy(torch.Tensor(mask_bbox).unsqueeze(0)).to(
            occluder_mask.device)
        occlusions = BitMasks(occluder_mask).crop_and_resize(
            bbox_mask.repeat(occluder_mask.shape[0], 1), REND_SIZE)
        # Remove occlusions
        with_occlusions = occluder_mask.new(mask).float()
        with_occlusions[occlusions.sum(0) > 0] = -1

        # Draw back original object mask in case it was removed by occlusions
        with_occlusions[mask] = 1
        occluded_masks.append(npt.numpify(with_occlusions))
    return occluded_masks


def add_target_hand_occlusions(person_parameters,
                               object_parameters,
                               K,
                               square_expand=0,
                               sample_folder=None,
                               debug=True):
    """
    Args:
        person_parameters (dict): {"bboxes": square xyxy bboxes, "masks", [B, IMAGE_SIZE, IMAGE_SIZE]}
        object_parameters (dict): {masks", [B, IMAGE_SIZE, IMAGE_SIZE]}
    """
    person_masks = BitMasks(person_parameters['masks'])

    # Expand box and bring back to model
    tight_boxes = person_parameters["bboxes"]
    batch_size = tight_boxes.shape[0]
    person_boxes = bbox_wh_to_xy(
        make_bbox_square(bbox_xy_to_wh(tight_boxes),
                         bbox_expansion=square_expand))
    person_boxes = tight_boxes.new(person_boxes)
    target_masks = person_masks.crop_and_resize(person_boxes,
                                                REND_SIZE).float()
    object_masks = BitMasks(object_parameters['full_mask'].repeat(
        batch_size, 1, 1)).crop_and_resize(person_boxes, REND_SIZE)
    target_masks[object_masks > 0] = -1
    # Compute corresponding K_roi
    K_roi = kcrop.get_K_crop_resize(
        person_boxes.new(K).unsqueeze(0).repeat(batch_size, 1, 1),
        person_boxes, [
            REND_SIZE,
        ] * batch_size)
    if debug:
        imagify.viz_imgrow(target_masks,
                           os.path.join(sample_folder, "tmpoccl.png"))
        print(f"Saving occlusion masks to {sample_folder}/tmpoccl.png")
    # Bring crop K to NC rendering space
    K_roi[:, :2] = K_roi[:, :2] / REND_SIZE
    person_parameters['K_roi'] = K_roi
    person_parameters['target_masks'] = target_masks
    person_parameters['square_bboxes'] = person_boxes
    return person_parameters
