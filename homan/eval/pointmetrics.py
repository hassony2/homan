#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from scipy import spatial

from homan.interactions import scenesdf

import trimesh
from collections import defaultdict
from libyana import distutils
from libyana.conversions import npt
from pytorch3d.loss import chamfer

# pylint: disable=no-member,too-many-locals,missing-function-docstring,wrong-import-order


def get_point_metrics(gt_points, pred_points):
    results = defaultdict(list)
    # dist_mat = distutils.batch_pairwise_dist(gt_points, pred_points)
    # Checked against pytorch3d chamfer loss

    with torch.no_grad():
        chamfer_dists = chamfer.chamfer_distance(gt_points.cuda(),
                                                 pred_points.cuda(),
                                                 batch_reduction=None)[0]
    # chamfer_dists = (dist_mat.min(1)[0] + dist_mat.min(2)[0]).mean(-1)
    results["chamfer_dists"].extend(npt.numpify(chamfer_dists).tolist())
    # ADD-S from
    # https://github.com/thodan/bop_toolkit/blob/53150b649467976b4f619fbffb9efe525c7e11ca/
    # bop_toolkit_lib/pose_error.py#L164
    adis = []
    for pts_gt, pts_est in zip(npt.numpify(gt_points),
                               npt.numpify(pred_points)):
        nn_index = spatial.cKDTree(pts_est)
        nn_dists, _ = nn_index.query(pts_gt, k=1)
        adis.append(nn_dists.mean())
    results["add-s"].extend(adis)
    if gt_points.shape[1] == pred_points.shape[1]:
        # Use vertex assignments
        vert_mean_dists = (gt_points - pred_points).norm(2, -1).mean(-1)
        results["verts_dists"].extend(npt.numpify(vert_mean_dists).tolist())
    else:
        # Else, repeat symmetric term
        results["verts_dists"].extend(adis)
    return dict(results)


def repeat_hand_nb(tens, hand_nb):
    sample_nb = tens.shape[0]
    if tens.dim() == 1:
        tens = tens.unsqueeze(1)
    if tens.dim() == 2:
        tens = tens.unsqueeze(2)
    channel_dim = tens.shape[-1]

    tens_reps = tens.repeat(1, hand_nb, 1).view(hand_nb * sample_nb, -1,
                                                channel_dim)
    return tens_reps


def get_align_metrics(gt_hand_verts, pred_hand_verts, gt_obj_verts,
                      pred_obj_verts):
    # Recover number of hands in image
    hand_size_tot = gt_hand_verts.shape[0]
    obj_size_tot = gt_obj_verts.shape[0]
    hand_nb = hand_size_tot // obj_size_tot

    gt_cent = gt_hand_verts[::hand_nb].mean(1, keepdim=True)
    pred_cent = gt_hand_verts[::hand_nb].mean(1, keepdim=True)
    # Center GT and pred using hand information
    gt_hand_verts_c = gt_hand_verts - repeat_hand_nb(gt_cent, hand_nb)
    gt_obj_verts_c = gt_obj_verts - gt_cent
    pred_hand_verts_c = pred_hand_verts - repeat_hand_nb(pred_cent, hand_nb)
    pred_obj_verts_c = pred_obj_verts - pred_cent

    # Get scale by comparing gt and pred hands
    gt_scale = torch.sqrt((gt_hand_verts_c[::hand_nb].norm(2, -1)**2).sum(1) /
                          gt_hand_verts[::hand_nb].shape[1])
    pred_scale = torch.sqrt(
        (pred_hand_verts_c[::hand_nb].norm(2, -1)**2).sum(1) /
        pred_hand_verts[::hand_nb].shape[1])

    pred_hand_verts_cs = pred_hand_verts_c / repeat_hand_nb(
        pred_scale, hand_nb) * repeat_hand_nb(gt_scale, hand_nb)
    pred_obj_verts_cs = pred_obj_verts_c / pred_scale.unsqueeze(1).unsqueeze(
        1) * gt_scale.unsqueeze(1).unsqueeze(1)
    hand_vert_mean_dists = (gt_hand_verts_c - pred_hand_verts_cs).norm(
        2, -1).mean(-1)
    with torch.no_grad():
        chamfer_dists = chamfer.chamfer_distance(pred_obj_verts_cs.cuda(),
                                                 gt_obj_verts_c.cuda(),
                                                 batch_reduction=None)[0]
    # dist_mat = distutils.batch_pairwise_dist(pred_obj_verts_cs, gt_obj_verts_c)
    # Same results as pytorch3d
    # chamfer_dists = (dist_mat.min(1)[0] + dist_mat.min(2)[0]).mean(-1)
    results = {}
    results["hand_mean_aligned"] = npt.numpify(hand_vert_mean_dists).tolist()
    results["obj_chamfer_aligned"] = npt.numpify(chamfer_dists).tolist()
    return results


def get_inter_metrics(verts_person, verts_object, faces_person, faces_object):
    hand_nb = verts_person.shape[0] // verts_object.shape[0]
    if hand_nb == 2:
        import pudb
        verts_person = verts_person.view(-1, hand_nb, verts_person.shape[1],
                                         3).view(verts_object.shape[0], -1, 3)
        faces_person = torch.cat(
            [faces_person[0], faces_person[1] + verts_person.shape[1]],
            0).unsqueeze(0)
    elif hand_nb > 3:
        raise ValueError(f"Invalid hand nb {hand_nb}")
    sdfl = scenesdf.SDFSceneLoss([faces_person[0], faces_object[0]])
    sdf_loss, sdf_meta = sdfl([verts_person, verts_object])
    # Penetration of hand into object
    max_depths = sdf_meta['dist_values'][(1, 0)].max(1)[0]
    # If at least 1 vertex with contact, config is considered in contact
    # max_depths_obj_in_hand = sdf_meta['dist_values'][(0, 1)].max(1)[0]
    has_contact = (max_depths > 0)
    # check penetration of object into hand
    return {
        "pen_depths": npt.numpify(max_depths).tolist(),
        "has_contact": npt.numpify(has_contact).tolist()
    }
