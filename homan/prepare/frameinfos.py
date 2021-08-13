#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0411,broad-except,too-many-statements,too-many-branches,logging-fstring-interpolation,import-error,too-many-arguments
# pylint: disable=too-many-locals,missing-function-docstring

import numpy as np
import torch

from homan.lib2d import maskutils
from homan.mocap import process_handmocap_predictions
from homan.prepare.gtmasks import render_gt_masks
from homan.utils.bbox import bbox_wh_to_xy, bbox_xy_to_wh
from homan.viz.vizframeinfo import viz_frame_info

import os
from libyana.verify import checkshape


def process_hand_boxes(image, hand_boxes, hand_preds, mask_extractor,
                       image_size):
    if isinstance(hand_boxes, list):
        hand_boxes = np.stack(hand_boxes)
    hand_annots = mask_extractor.masks_from_bboxes(image,
                                                   hand_boxes,
                                                   class_idx=0,
                                                   pred_classes=None,
                                                   image_size=image_size)
    full_masks = np.stack([annot["full_mask"] for annot in hand_annots])
    hand_parameters = process_handmocap_predictions(
        mocap_predictions=hand_preds,
        bboxes=bbox_wh_to_xy(hand_boxes),
        masks=full_masks,
        image_size=image_size)
    return hand_parameters


def get_frame_infos(images_np,
                    hand_predictor=None,
                    mask_extractor=None,
                    sample_folder=None,
                    hand_bboxes=None,
                    obj_bboxes=None,
                    camintr=None,
                    debug=True,
                    image_size=640):
    """
    Arguments:
        images_np (list[np.ndarray]): List of input images
        hand_bboxes (dict): dictionnary containing {left_hand: [None|frame_nb x 4], right_hand: [None|frame_nb x 4]}
            sequence bounding boxes in xywh format
        obj_bboxes (np.ndarray): (1, frame_nb, 4) xywh object bounding boxes
        camintr (list[np.ndarray]): (frame_nb, 3, 3) intrinsic camera parameters
        image_size (int): image size
    """
    checkshape.check_shape(obj_bboxes, (1, -1, 4), "obj_bboxes")
    checkshape.check_shape(camintr[0], (3, 3), "camintr")

    person_parameters = []
    obj_mask_infos = []
    super2d_imgs = []
    with torch.no_grad():
        for image_idx, image in enumerate(images_np):
            image_hand_boxes = {
                key: boxes[image_idx]
                for key, boxes in hand_bboxes.items() if boxes is not None
            }
            frame_info = get_frame_info(
                image,
                hand_predictor,
                mask_extractor,
                sample_folder=sample_folder,
                hand_bboxes=[image_hand_boxes],
                obj_bboxes=obj_bboxes[:, image_idx],
                camintr=camintr[image_idx],
                # Save visualization of middle frame
                debug=debug and (image_idx == len(images_np) // 2),
                image_size=image_size,
            )
            person_parameters.append(frame_info["person_parameters"])
            obj_mask_infos.append(frame_info["obj_mask_infos"])
            super2d_img = viz_frame_info(frame_info,
                                         sample_folder=sample_folder,
                                         save=False)
            super2d_imgs.append(super2d_img)
        super2d_imgs = np.concatenate(super2d_imgs[::len(super2d_imgs) // 10],
                                      1)
    return person_parameters, obj_mask_infos, super2d_imgs


def get_frame_info(image,
                   hand_predictor=None,
                   mask_extractor=None,
                   sample_folder=None,
                   hand_bboxes=None,
                   obj_bboxes=None,
                   camintr=None,
                   debug=True,
                   image_size=640):
    """
    Regress frame hand pose and hand+object masks

    Arguments:
        image (np.ndarray): hand-object image 
        hand_bboxes (list): [{'left_hand': np.array(4,), 'right_hand': np.array(4,)}, ...] in xywh format
        hand_predictor: Hand pose regressor
        mask_extractor: Instance segmentor
    Returns:
        frame_infos (dict): Contains person parameters and mask information
    """
    person_parameters = {}
    left_boxes = [
        boxes['left_hand'].clip(0, None) for boxes in hand_bboxes
        if (("left_hand" in boxes) and (boxes["left_hand"] is not None))
    ]
    if hand_predictor is not None:
        mocap_predictions = hand_predictor.regress(image[..., ::-1],
                                                   hand_bboxes,
                                                   add_margin=False,
                                                   debug=debug,
                                                   K=camintr,
                                                   viz_path=os.path.join(
                                                       sample_folder,
                                                       "hands.png"))
        left_preds = [pred['left_hand'] for pred in mocap_predictions]
    else:
        left_preds = None
    all_parameters = []
    if len(left_boxes) > 0:
        left_parameters = process_hand_boxes(image,
                                             hand_boxes=left_boxes,
                                             hand_preds=left_preds,
                                             mask_extractor=mask_extractor,
                                             image_size=image_size)
        all_parameters.append(left_parameters)
    if hand_predictor is not None:
        right_preds = [pred['right_hand'] for pred in mocap_predictions]
    else:
        right_preds = None

    right_boxes = [
        boxes['right_hand'].clip(0, None) for boxes in hand_bboxes
        if (("right_hand" in boxes) and (boxes["right_hand"] is not None))
    ]
    if len(right_boxes) > 0:
        right_boxes = np.stack(right_boxes)
        right_parameters = process_hand_boxes(image,
                                              hand_boxes=right_boxes,
                                              hand_preds=right_preds,
                                              mask_extractor=mask_extractor,
                                              image_size=image_size)
        all_parameters.append(right_parameters)

    for key in all_parameters[0].keys():
        if isinstance(all_parameters[0][key], str):
            # Process labels separately
            person_parameters[key] = [param[key] for param in all_parameters]
        else:
            person_parameters[key] = torch.cat(
                [param[key] for param in all_parameters])
    # Handling only 1 object
    obj_mask_infos = mask_extractor.masks_from_bboxes(
        image,
        bbox_xy_to_wh(obj_bboxes),
        pred_classes=None,
        image_size=image_size)[0]

    # Masks with -1 for occluded parts, by merging rendered and segmentation masks
    if (len(person_parameters) > 0) and ("rend" in person_parameters):
        hand_occlusions = (
            (person_parameters["rend"].sum([0, 1]).transpose(1, 0) +
             person_parameters["masks"]) > 0)
    else:
        hand_occlusions = person_parameters["masks"] > 0
    target_masks = maskutils.add_occlusions([obj_mask_infos["crop_mask"]],
                                            hand_occlusions,
                                            [obj_mask_infos["square_bbox"]])[0]
    obj_mask_infos["target_crop_mask"] = target_masks
    frame_infos = dict(
        person_parameters=person_parameters,
        obj_mask_infos=obj_mask_infos,
        image=image,
    )

    return frame_infos


def get_gt_infos(images_np,
                 annots,
                 obj_mask_infos,
                 person_parameters,
                 image_size=None,
                 sample_folder="tmp"):
    render_gt_masks(annots,
                    obj_mask_infos,
                    person_parameters,
                    image_size=image_size)
    gt2d_imgs = []
    for image, obj_params, person_params in zip(images_np, obj_mask_infos,
                                                person_parameters):
        gt2d_img = viz_frame_info(
            {
                "image": image,
                "person_parameters": person_params,
                "obj_mask_infos": obj_params,
            },
            sample_folder=sample_folder,
            save=False)
        gt2d_imgs.append(gt2d_img)
    gt2d_imgs = np.concatenate(gt2d_imgs[::len(gt2d_imgs) // 2], 1)
    return gt2d_imgs
