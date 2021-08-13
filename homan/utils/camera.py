# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch

from homan.utils.bbox import bbox_xy_to_wh, make_bbox_square
from libyana.camutils import camconvs


def local_to_global_cam(bboxes, cams, L):
    """
    Converts a weak-perspective camera w.r.t. a bounding box to a weak-perspective
    camera w.r.t. to the entire image.

    Args:
        bboxes (N x 4): Bounding boxes in xyxy format.
        cams (N x 3): Weak perspective camera.
        L (int): Max of height and width of image.
    """
    square_bboxes = make_bbox_square(bbox_xy_to_wh(bboxes))
    global_cams = []
    for cam, bbox in zip(cams, square_bboxes):
        x, y, b, _ = bbox
        X = np.stack((x, y))
        # Bbox space [0, b]
        s_crop = b * cam[0] / 2
        t_crop = cam[1:] + 1 / cam[0]

        # Global image space [0, 1]
        s_og = s_crop / L
        t_og = t_crop + X / s_crop

        # Normalized global space [-1, 1]
        s = s_og * 2
        t = t_og - 0.5 / s_og
        global_cams.append(np.concatenate((np.array([s]), t)))
    return np.stack(global_cams)


def compute_K_roi(upper_left, b, img_size, focal_length=1.0):
    """
    Computes the intrinsics matrix for a cropped ROI box.

    Args:
        upper_left (tuple): Top left corner (x, ytorch.Tensor(gt_verts).unsqueeze(0)).
        b (float): Square box size.
        img_size (int): Size of image in pixels.

    Returns:
        Intrinsic matrix (1 x 3 x 3).
    """
    x1, y1 = upper_left
    f = focal_length * img_size / b
    px = (img_size / 2 - x1) / b
    py = (img_size / 2 - y1) / b
    K = torch.cuda.FloatTensor([[[f, 0, px], [0, f, py], [0, 0, 1]]])
    return K


def compute_transformation_ortho(meshes,
                                 cams,
                                 rotations=None,
                                 intrinsic_scales=None,
                                 K=None,
                                 img=None,
                                 image_size=640):
    """
    Computes the 3D transformation from a scaled orthographic camera model.

    Args:
        meshes (V x 3 or B x V x 3): Vertices.
        cams (B x 3): Scaled orthographic camera [s, tx, ty].
        rotations (B x 3 x 3): Rotation matrices.
        intrinsic_scales (B).
        focal_length (float): Should be 2x object focal length due to scaling.
        K (B x 3 x 3)

    Returns:
        vertices (B x V x 3).
    """
    B = len(cams)
    device = cams.device
    if meshes.ndimension() == 2:
        meshes = meshes.repeat(B, 1, 1)
    if rotations is None:
        rotations = torch.eye(3).repeat(B, 1, 1).to(device)
    if intrinsic_scales is None:
        intrinsic_scales = torch.ones(B).to(device)
    persp_scale = (cams[:, :1] / 2 * image_size
                   )  # scale of verts_pixel / verts_origin
    persp_trans = (cams[:, 1:] + 1 / cams[:, :1]) * persp_scale
    # perspective camera in pixel space
    orthocams_pixels = torch.cat([persp_scale, persp_trans], 1)
    # K in pixel coordinates
    K_pixels = K.clone()
    K_pixels[:, :2] = K_pixels[:, :2] * image_size
    trans = camconvs.batch_weakcam2persptrans(orthocams_pixels, K_pixels,
                                              1).unsqueeze(1)  # B x 1 x 3
    verts_rot_det = torch.matmul(meshes.detach().clone(),
                                 rotations)  # B x V x 3
    verts_rot = torch.matmul(meshes, rotations)  # B x V x 3
    verts_trans = verts_rot + trans
    verts_trans_det = verts_rot_det + trans
    verts_final = intrinsic_scales.view(-1, 1, 1) * verts_trans
    verts_final_det = intrinsic_scales.view(-1, 1, 1) * verts_trans_det
    return verts_final, verts_final_det


def compute_transformation_persp(meshes,
                                 translations,
                                 rotations=None,
                                 intrinsic_scales=None):
    """
    Computes the 3D transformation.

    Args:
        meshes (V x 3 or B x V x 3): Vertices.
        translations (B x 1 x 3).
        rotations (B x 3 x 3).
        intrinsic_scales (B).

    Returns:
        vertices (B x V x 3).
    """
    B = translations.shape[0]
    device = meshes.device
    if meshes.ndimension() == 2:
        meshes = meshes.repeat(B, 1, 1)
    if rotations is None:
        rotations = torch.FloatTensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        rotations = rotations.to(device)
    if intrinsic_scales is None:
        intrinsic_scales = torch.ones(B).to(device)
    meshes_scaled = intrinsic_scales.view(-1, 1, 1) * meshes
    verts_rot = torch.matmul(meshes_scaled, rotations)
    verts_rot_det = torch.matmul(meshes_scaled.detach().clone(),
                                 rotations)  # B x V x 3
    verts_trans = verts_rot + translations
    verts_trans_det = verts_rot_det + translations
    return verts_trans, verts_trans_det
