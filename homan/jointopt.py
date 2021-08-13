# Copyright (c) Facebook, Inc. and its affiliates.
"""
Main functions for doing hand-object optimization.
"""
# pylint: disable=import-error,no-member,too-many-arguments,too-many-locals
from collections import OrderedDict, defaultdict
import os

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from libyana.conversions import npt
from libyana.vidutils import np2vid

from homan.homan import HOMan
from homan.visualize import visualize_hand_object


def optimize_hand_object(
    person_parameters,
    object_parameters,
    class_name="default",
    objvertices=None,
    objfaces=None,
    loss_weights=None,
    num_iterations=400,
    lr=1e-2,
    images=None,
    viz_step=10,
    viz_folder="tmp",
    camintr=None,
    hand_proj_mode="persp",
    optimize_mano=False,
    optimize_mano_beta=True,
    optimize_object_scale=False,
    state_dict=None,
    fps=24,
    viz_len=7,
    image_size=640,
):
    """
    Arguments:
        fps (int): frames per second for video visualization
        viz_len (int): number of frames to show
    """
    os.makedirs(viz_folder, exist_ok=True)

    # Load mesh data.
    verts_object_og = npt.tensorify(objvertices).cuda()
    faces_object = npt.tensorify(objfaces).cuda()

    obj_trans = torch.cat([obj["translations"] for obj in object_parameters])
    obj_rots = torch.cat([obj["rotations"] for obj in object_parameters])
    person_trans = torch.cat(
        [person["translations"] for person in person_parameters])
    hand_sides = [param["hand_side"] for param in person_parameters]

    # Assumes all samples in batch have the same hand configuration
    hand_sides = hand_sides[0]
    faces_hand = person_parameters[0]['faces']

    person_mano_trans = torch.cat(
        [person["mano_trans"] for person in person_parameters])
    person_mano_rot = torch.cat(
        [person["mano_rot"] for person in person_parameters])
    person_mano_betas = torch.cat(
        [person["mano_betas"] for person in person_parameters])
    person_mano_pca_pose = torch.cat(
        [person["mano_pca_pose"] for person in person_parameters])

    person_rots = torch.cat(
        [person["rotations"] for person in person_parameters])
    hand_tar_masks = torch.cat(
        [hand["target_masks"] for hand in person_parameters])
    obj_tar_masks = torch.cat(
        [obj["target_masks"] for obj in object_parameters])
    person_full_masks = torch.cat(
        [person["masks"] for person in person_parameters])
    obj_full_masks = torch.cat(
        [obj["full_mask"].unsqueeze(0) for obj in object_parameters])
    person_verts = torch.cat([person["verts"] for person in person_parameters])
    person_verts2d = torch.cat(
        [person["verts2d"] for person in person_parameters])
    person_camintr_roi = torch.cat(
        [person["K_roi"] for person in person_parameters])
    obj_camintr_roi = torch.cat(
        [obj["K_roi"][:, 0] for obj in object_parameters])
    person_cams = torch.cat([person["cams"] for person in person_parameters])
    model = HOMan(
        hand_sides=hand_sides,
        translations_object=obj_trans,  # [B, 1, 3]
        rotations_object=obj_rots,  # [B, 3, 3]
        verts_object_og=verts_object_og,  # [B, VN, 3]
        faces_object=faces_object,  # [B, FN, 3]
        # Used for silhouette supervision
        target_masks_object=obj_tar_masks,  # [B, REND_SIZE, REND_SIZE]
        target_masks_hand=hand_tar_masks,  # [B, REND_SIZE, REND_SIZE]
        verts_hand_og=person_verts,
        ref_verts2d_hand=person_verts2d,
        mano_trans=person_mano_trans,
        mano_rot=person_mano_rot,
        mano_pca_pose=person_mano_pca_pose,
        mano_betas=person_mano_betas,
        translations_hand=person_trans,
        rotations_hand=person_rots,
        faces_hand=faces_hand,  # [B, FN, 3]
        # Used for ordinal depth loss
        masks_object=obj_full_masks,  # [B, IMAGE_SIZE, IMAGE_SIZE]
        masks_hand=person_full_masks,  # [B, IMAGE_SIZE, IMAGE_SIZE]
        cams_hand=person_cams,
        camintr_rois_object=obj_camintr_roi,
        camintr_rois_hand=person_camintr_roi,
        camintr=camintr,
        class_name=class_name,
        int_scale_init=1,
        hand_proj_mode=hand_proj_mode,
        optimize_mano=optimize_mano,
        optimize_mano_beta=optimize_mano_beta,
        optimize_object_scale=optimize_object_scale,
        image_size=image_size,
    )
    # Resume from state_dict if provided
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=False)
    rigid_parameters = [
        val for key, val in model.named_parameters()
        if "mano" not in key and "rotation" not in key
    ]

    rotation_parameters = [
        # Do not try to refactor an iter !
        val for key, val in model.named_parameters()
        if ("rotation" in key) and ("mano" not in key)
    ]
    optimizer = torch.optim.Adam([{
        "params": rigid_parameters,
        "lr": lr
    }, {
        "params": [
            model.mano_pca_pose,
            model.mano_betas,
        ],
        "lr":
        lr * 10
    }, {
        "params": rotation_parameters,
        "lr": lr * 10
    }])
    # optimizer = torch.optim.Adam(parameters, lr=lr)
    loop = tqdm(range(num_iterations))
    loss_evolution = defaultdict(list)

    imgs = OrderedDict()
    optim_imgs = []
    for step in loop:
        if step % viz_step == 0:
            with torch.no_grad():
                frontal, top_down = visualize_hand_object(model,
                                                          images,
                                                          dist=1,
                                                          viz_len=viz_len)

            file_name = f"{step:08d}.jpg"
            front_top_path = os.path.join(viz_folder, file_name)
            frontal = np.concatenate([img for img in frontal], 1)
            top_down = np.concatenate([img for img in top_down], 1)
            front_top = np.concatenate(
                [frontal, top_down[:frontal.shape[0], :frontal.shape[1]]], 0)
            front_top = cv2.resize(
                front_top, (front_top.shape[1] // 2, front_top.shape[0] // 2))
            Image.fromarray(front_top).save(front_top_path)
            imgs[step] = front_top_path
            optim_imgs.append(front_top)
            print(f"Saved rendered image to {front_top_path}.")
        optimizer.zero_grad()
        loss_dict, metric_dict = model(loss_weights=loss_weights)
        loss_dict_weighted = {
            k: loss_dict[k] * loss_weights[k.replace("loss", "lw")]
            for k in loss_dict
        }
        for k, val in loss_dict.items():
            loss_evolution[k].append(val.item())
        for k, val in metric_dict.items():
            loss_evolution[k].append(val)
        loss = sum(loss_dict_weighted.values())
        loss_evolution["loss"].append(loss.item())
        loop.set_description(f"Loss {loss.item():.4f}")
        loss.backward()
        optimizer.step()
    optim_imgs = [optim_imgs[0] for _ in range(30)
                  ] + optim_imgs + [optim_imgs[-1] for _ in range(50)]
    np2vid.make_video(optim_imgs,
                      front_top_path.replace(".jpg", f"{fps}.gif"),
                      fps=fps)
    video_path = os.path.join(os.path.dirname(viz_folder), "joint_optim.webm")
    np2vid.make_video(optim_imgs, video_path, fps=fps)
    np2vid.make_video(optim_imgs, video_path.replace(".webm", ".mp4"), fps=fps)
    return model, dict(loss_evolution), imgs
