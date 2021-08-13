#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch

from libyana.conversions import npt
from libyana.camutils import project


def transform_pts(T, pts):
    bsz = T.shape[0]
    n_pts = pts.shape[1]
    assert pts.shape == (bsz, n_pts, 3)
    if T.dim() == 4:
        pts = pts.unsqueeze(1)
        assert T.shape[-2:] == (4, 4)
    elif T.dim() == 3:
        assert T.shape == (bsz, 4, 4)
    else:
        raise ValueError('Unsupported shape for T', T.shape)
    pts = pts.unsqueeze(-1)
    T = T.unsqueeze(-3)
    pts_transformed = T[..., :3, :3] @ pts + T[..., :3, [-1]]
    return pts_transformed.squeeze(-1)


def compute_bbox_proj(verts, f, img_size=256):
    """
    Computes the 2D bounding box of vertices projected to the image plane.

    Args:
        verts (B x N x 3): Vertices.
        f (float): Focal length.
        img_size (int): Size of image in pixels.

    Returns:
        Bounding box in xywh format (Bx4).
    """
    xy = verts[:, :, :2]
    z = verts[:, :, 2:]
    proj = f * xy / z + 0.5  # [0, 1]
    proj = proj * img_size  # [0, img_size]
    u, v = proj[:, :, 0], proj[:, :, 1]
    x1, x2 = u.min(1).values, u.max(1).values
    y1, y2 = v.min(1).values, v.max(1).values
    return torch.stack((x1, y1, x2 - x1, y2 - y1), 1)


def compute_optimal_translation(bbox_target, vertices, f=1, img_size=256):
    """
    Computes the optimal translation to align the mesh to a bounding box using
    least squares.

    Args:
        bbox_target (list): bounding box in xywh.
        vertices (B x V x 3): Batched vertices.
        f (float): Focal length.
        img_size (int): Image size in pixels.

    Returns:
        Optimal 3D translation (B x 3).
    """
    bbox_mask = np.array(bbox_target)
    mask_center = bbox_mask[:2] + bbox_mask[2:] / 2
    diag_mask = np.sqrt(bbox_mask[2]**2 + bbox_mask[3]**2)
    B = vertices.shape[0]
    x = torch.zeros(B).cuda()
    y = torch.zeros(B).cuda()
    z = 2.5 * torch.ones(B).cuda()
    for _ in range(50):
        translation = torch.stack((x, y, z), -1).unsqueeze(1)
        v = vertices + translation
        bbox_proj = compute_bbox_proj(v, f=f, img_size=img_size)
        diag_proj = torch.sqrt(torch.sum(bbox_proj[:, 2:]**2, 1))
        delta_z = z * (diag_proj / diag_mask - 1)
        z = z + delta_z
        proj_center = bbox_proj[:, :2] + bbox_proj[:, 2:] / 2
        x += (mask_center[0] - proj_center[:, 0]) * z / f / img_size
        y += (mask_center[1] - proj_center[:, 1]) * z / f / img_size
    return torch.stack((x, y, z), -1).unsqueeze(1)


def TCO_init_from_boxes_zup_autodepth(boxes_2d, model_points_3d, K):
    # User in BOP20 challenge
    model_points_3d = npt.tensorify(model_points_3d)
    bsz = model_points_3d.shape[0]
    device = model_points_3d.device
    K = npt.tensorify(K).to(device)
    boxes_2d = npt.tensorify(boxes_2d).to(device)
    if boxes_2d.dim() == 1:
        boxes_2d = boxes_2d.unsqueeze(0)
    if boxes_2d.shape[0] != bsz:
        boxes_2d = boxes_2d.repeat(bsz, 1)
    if K.dim() == 2:
        K = K.unsqueeze(0)
    if K.shape[0] != bsz:
        K = K.repeat(bsz, 1, 1)

    assert boxes_2d.shape[-1] == 4
    assert boxes_2d.dim() == 2
    # xywh to xyxy
    boxes_2d = torch.stack([
        boxes_2d[:, 0], boxes_2d[:, 1], boxes_2d[:, 0] + boxes_2d[:, 2],
        boxes_2d[:, 1] + boxes_2d[:, 3]
    ], 1)
    # Get length of reference bbox diagonal
    diag_bb = (boxes_2d[:, [2, 3]] - boxes_2d[:, [0, 1]]).norm(2, -1)
    # Get center of reference bbox
    bb_xy_centers = (boxes_2d[:, [0, 1]] + boxes_2d[:, [2, 3]]) / 2
    fxfy = K[:, [0, 1], [0, 1]]
    cxcy = K[:, [0, 1], [2, 2]]
    z = fxfy.new_ones(bsz, 1)
    xy_init = ((bb_xy_centers - cxcy) * z) / fxfy
    trans = torch.cat([xy_init, z], 1)
    for _ in range(10):
        C_pts_3d = model_points_3d + trans.unsqueeze(1)
        proj_pts = project.batch_proj2d(C_pts_3d, K)
        diag_proj = (proj_pts.min(1)[0] - proj_pts.max(1)[0]).norm(2, -1)
        proj_xy_centers = (proj_pts.min(1)[0] + proj_pts.max(1)[0]) / 2

        # Update z to increase/decrease size of projected bbox
        delta_z = z * (diag_proj / diag_bb - 1).unsqueeze(-1)
        z = z + delta_z
        # Update xy to shift center of projected bbox
        xy_init += ((bb_xy_centers - proj_xy_centers) * z) / fxfy
        trans = torch.cat([xy_init, z], 1)
    return trans
