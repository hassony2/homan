#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import torch
import numpy as np

from homan.utils.bbox import bbox_wh_to_xy, bbox_xy_to_wh, make_bbox_square
import neural_renderer as nr
from detectron2.structures import BitMasks

from libyana.visutils import imagify


def render_gt_masks(annots,
                    obj_infos,
                    person_parameters,
                    sample_folder="",
                    debug=True,
                    image_size=640):
    "Replace obj and hand masks with ground truth"
    K = annots['camera']['K']
    K = torch.Tensor(K).cuda()
    bs = K.shape[0]
    R = torch.cuda.FloatTensor([[[1, 0, 0], [0, 1, 0], [0, 0,
                                                        1]]]).repeat(bs, 1, 1)
    trans = torch.Tensor([0, 0, 0]).cuda().unsqueeze(0).repeat(bs, 1, 1)
    renderer = nr.renderer.Renderer(image_size=image_size,
                                    K=K,
                                    R=R,
                                    t=trans,
                                    orig_size=1)
    instance_idx = 0
    all_verts = []
    all_faces = []
    all_textures = []
    faces_off = 0
    hand_nb = len(annots["hands"])
    for hand in annots['hands']:
        verts3d = hand['verts3d']
        faces = hand['faces']
        face_nb = faces.shape[1]
        colors = [0, 0, 0]
        colors[instance_idx] = 1
        instance_idx += 1
        textures = torch.Tensor(colors).view(-1, 1, 1, 1,
                                             3).repeat(face_nb, 1, 1, 1, 1)
        faces = torch.Tensor(faces.astype(np.float32)) + faces_off
        faces_off += verts3d.shape[1]
        verts = torch.Tensor(verts3d)
        all_faces.append(faces)
        all_verts.append(verts)
        all_textures.append(textures)
    for obj in annots['objects']:
        verts3d = obj['verts3d']
        faces = obj['faces']
        face_nb = faces.shape[1]
        colors = [0, 0, 0]
        colors[instance_idx] = 1
        instance_idx += 1
        textures = torch.Tensor(colors).view(-1, 1, 1, 1,
                                             3).repeat(face_nb, 1, 1, 1, 1)
        faces = torch.Tensor(faces.astype(np.float32)) + faces_off
        faces_off += verts3d.shape[1]
        verts = torch.Tensor(verts3d)
        all_faces.append(faces)
        all_verts.append(verts)
        all_textures.append(textures)
    all_faces = torch.cat(all_faces, 1).cuda()
    all_verts = torch.cat(all_verts, 1).cuda()
    all_textures = torch.cat(all_textures, 0)
    all_textures = all_textures.unsqueeze(0).repeat(bs, 1, 1, 1, 1, 1).cuda()
    # 2 * factor to be investigated !
    K_nc = K.clone()
    K_nc[:, :2] = 1 / image_size * K_nc[:, :2]
    renderer.light_intensity_direction = 0
    renderer.light_intensity_ambient = 1
    renders, sil, depth = renderer(all_verts, all_faces, all_textures, K=K_nc)
    if debug:
        sample_path = os.path.join(sample_folder, "rendered_gt.png")
        imagify.viz_imgrow(renders[0].clamp(0, 1), path=sample_path)

    obj_bitmasks = BitMasks(renders[:, hand_nb].cpu().float() > 0)
    obj_square_bboxes = bbox_wh_to_xy(
        torch.Tensor(
            np.stack([obj_info["square_bbox"] for obj_info in obj_infos])))
    # Get crops
    obj_crops_obj_bbox = obj_bitmasks.crop_and_resize(obj_square_bboxes,
                                                      256).float()
    obj_occlusions = []
    hand_crops = []
    for hand_idx in range(hand_nb):
        hand_bitmasks = BitMasks(renders[:, hand_idx].cpu().float() > 0)
        hand_bboxes = [
            person_param["bboxes"] for person_param in person_parameters
        ]
        hand_bboxes = torch.stack(hand_bboxes)
        hand_crops_obj_bbox = hand_bitmasks.crop_and_resize(
            obj_square_bboxes, 256)
        obj_occlusions.append(hand_crops_obj_bbox)
        hand_crops_hand_bbox = hand_bitmasks.crop_and_resize(
            hand_bboxes[:, hand_idx].cpu(), 256)
        hand_crops.append(hand_crops_hand_bbox)
    hand_crops = torch.stack(hand_crops, 1)
    obj_occlusions = torch.stack(obj_occlusions, 1)
    full_obj_occlusions = obj_occlusions.sum(1) > 0
    gt_obj_occlusions = obj_crops_obj_bbox.float() - full_obj_occlusions.float(
    )
    for time_idx, obj_info in enumerate(obj_infos):
        obj_info['target_crop_mask'][:] = gt_obj_occlusions[time_idx].numpy()
        obj_info['crop_mask'][:] = obj_crops_obj_bbox[time_idx] > 0
        obj_info['full_mask'][:] = (
            renders[time_idx, hand_nb, :obj_info["full_mask"].
                    shape[0], :obj_info["full_mask"].shape[1]] > 0)
    for time_idx, person_param in enumerate(person_parameters):
        hand_render_mask = renders[
                # time_idx, :hand_nb, :person_param["masks"].
                # shape[1], :person_param["masks"].shape[2]]
                time_idx, :hand_nb, :obj_info["full_mask"].
                shape[0], :obj_info["full_mask"].shape[1]]
        if "masks" in person_param:
            person_param['masks'][:] = hand_render_mask
        else:
            person_param["masks"] = hand_render_mask
