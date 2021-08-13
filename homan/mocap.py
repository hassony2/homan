# Copyright (c) Facebook, Inc. and its affiliates.
"""
Wrapper for Human Pose Estimator using BodyMocap.
See: https://github.com/facebookresearch/frankmocap
"""
import numpy as np
import torch

from detectron2.structures.masks import BitMasks

from homan.constants import BODY_MOCAP_REGRESSOR_CKPT, BODY_MOCAP_SMPL_PATH, HAND_MOCAP_REGRESSOR_CKPT
from homan.utils.nmr_renderer import OrthographicRenderer
from homan.utils.bbox import bbox_xy_to_wh
from homan.utils.camera import local_to_global_cam
from handmocap.hand_mocap_api import HandMocap
from handmocap.hand_bbox_detector import HandBboxDetector
from libyana.conversions import npt


def get_hand_bbox_detector():
    return HandBboxDetector('ego_centric', device=None)


def get_handmocap_predictor():
    hand_predictor = HandMocap(HAND_MOCAP_REGRESSOR_CKPT, BODY_MOCAP_SMPL_PATH)
    return hand_predictor


def get_hand_detections(hand_detector, image):
    _, _, hand_bbox_list, _ = hand_detector.detect_hand_bbox(image.copy())
    return hand_bbox_list


def process_handmocap_predictions(mocap_predictions,
                                  bboxes,
                                  image_size=640,
                                  masks=None):
    """
    Rescales camera to follow HMR convention, and then computes the camera w.r.t. to
    image rather than local bounding box.

    Args:
        mocap_predictions (list).
        bboxes (N x 4): Bounding boxes in xyxy format.
        image_size (int): Max dimension of image.
        masks (N x H x W): Bit mask of people.

    Returns:
        dict {str: torch.cuda.FloatTensor}
            bbox: Bounding boxes in xyxy format (N x 3).
            cams: Weak perspective camera (N x 3).
            masks: Bitmasks used for computing ordinal depth loss, cropped to image
                space (N x L x L).
            local_cams: Weak perspective camera relative to the bounding boxes (N x 3).
    """
    max_dim = np.max(bbox_xy_to_wh(bboxes)[:, 2:], axis=1)
    inds = np.argsort(
        bboxes[:, 0])  # Sort from left to right to make debugging easy.
    if mocap_predictions is not None:
        verts = np.stack([p["pred_vertices_smpl"] for p in mocap_predictions])
        verts2d = np.stack(
            [p["pred_vertices_img"][:, :2] for p in mocap_predictions])
        # Get perspective parameters
        translations = np.stack(
            [p["perspective_trans"] for p in mocap_predictions])
        # My and PHOSA rotation conversions are transposed !
        rotations = np.stack(
            [p["perspective_rot"].transpose() for p in mocap_predictions])
        # rends = np.stack([p["rend"].transpose() for p in mocap_predictions])
        # All faces are the same, so just need one copy.
        faces = np.expand_dims(mocap_predictions[0]["faces"].astype(np.int32),
                               0)
        local_cams = []
        for b, pred in zip(max_dim, mocap_predictions):
            local_cam = pred["pred_camera"].copy()
            local_cams.append(local_cam)
        local_cams = np.stack(local_cams)
        global_cams = local_to_global_cam(bboxes, local_cams, image_size)

        person_parameters = {
            "bboxes": bboxes[inds].astype(np.float32),
            "cams": global_cams[inds].astype(np.float32),
            "faces": faces,
            "local_cams": local_cams[inds].astype(np.float32),
            "verts": verts[inds].astype(np.float32),
            "verts2d": verts2d[inds].astype(np.float32),
            "rotations": rotations[inds].astype(np.float32),
            "mano_pose": pred["pred_hand_pose"][inds, 3:].astype(np.float32),
            "mano_pca_pose": pred["pred_pca_pose"][inds].astype(np.float32),
            "mano_rot": pred["pred_hand_pose"][inds, :3].astype(np.float32),
            "mano_betas": pred["pred_hand_betas"][inds].astype(np.float32),
            # "rend": rends[inds].astype(np.float32),
            "mano_trans":
            npt.numpify(pred["mano_trans"][inds]).astype(np.float32),
            "translations": translations[inds].astype(np.float32),
        }
        person_parameters["hand_side"] = pred["hand_side"]
    else:
        person_parameters = {
            "bboxes": bboxes[inds].astype(np.float32),
        }
    for k, v in person_parameters.items():
        if isinstance(v, str):
            person_parameters[k] = v
        else:
            person_parameters[k] = torch.from_numpy(v).cuda()
    if masks is not None:
        # full_boxes = torch.tensor([[0, 0, image_size, image_size]] *
        #                           len(bboxes))
        # full_boxes = full_boxes.float().cuda()
        masks = npt.tensorify(masks).cuda()
        person_parameters["masks"] = masks[inds]
    return person_parameters


def process_mocap_predictions(mocap_predictions=None,
                              bboxes=None,
                              image_size=640,
                              masks=None):
    """
    Rescales camera to follow HMR convention, and then computes the camera w.r.t. to
    image rather than local bounding box.

    Args:
        mocap_predictions (list).
        bboxes (N x 4): Bounding boxes in xyxy format.
        image_size (int): Max dimension of image.
        masks (N x H x W): Bit mask of people.

    Returns:
        dict {str: torch.cuda.FloatTensor}
            bbox: Bounding boxes in xyxy format (N x 3).
            cams: Weak perspective camera (N x 3).
            masks: Bitmasks used for computing ordinal depth loss, cropped to image
                space (N x L x L).
            local_cams: Weak perspective camera relative to the bounding boxes (N x 3).
    """
    if mocap_predictions is not None:
        verts = np.stack([p["pred_vertices_smpl"] for p in mocap_predictions])
        # All faces are the same, so just need one copy.
        faces = np.expand_dims(mocap_predictions[0]["faces"].astype(np.int32),
                               0)
        max_dim = np.max(bbox_xy_to_wh(bboxes)[:, 2:], axis=1)
        local_cams = []
        global_cams = []
        for b, pred in zip(max_dim, mocap_predictions):
            local_cam = pred["pred_camera"].copy()
            global_cam = pred["global_cams"].copy()
            scale_o2n = pred["bbox_scale_ratio"] * b / 224
            local_cam[0] /= scale_o2n
            local_cam[1:] /= local_cam[:1]
            local_cams.append(local_cam)
            global_cams.append(global_cam)
        local_cams = np.stack(local_cams)
        global_cams = np.stack(global_cams)
    inds = np.argsort(
        bboxes[:, 0])  # Sort from left to right to make debugging easy.
    if mocap_predictions is not None:
        person_parameters = {
            "cams": global_cams[inds].astype(np.float32),
            "faces": faces,
            "local_cams": local_cams[inds].astype(np.float32),
            "verts": verts[inds].astype(np.float32),
        }
    else:
        person_parameters = {}
    person_parameters["bboxes"] = bboxes[inds].astype(np.float32)

    for k, v in person_parameters.items():
        person_parameters[k] = torch.from_numpy(v).cuda()

    if masks is not None:
        full_boxes = torch.tensor([[0, 0, image_size, image_size]] *
                                  len(bboxes))
        full_boxes = full_boxes.float().cuda()
        masks = BitMasks(masks).crop_and_resize(boxes=full_boxes,
                                                mask_size=image_size)
        person_parameters["masks"] = masks[inds].cuda()
    return person_parameters


def visualize_orthographic(image, human_predictions):
    ortho_renderer = OrthographicRenderer(image_size=max(image.shape))
    new_image = image.copy()
    verts = human_predictions["verts"]
    faces = human_predictions["faces"]
    cams = human_predictions["cams"]
    for i in range(len(verts)):
        v = verts[i:i + 1]
        cam = cams[i:i + 1]
        new_image = ortho_renderer(vertices=v,
                                   faces=faces,
                                   cam=cam,
                                   color_name="blue",
                                   image=new_image)
    return (new_image * 255).astype(np.uint8)
