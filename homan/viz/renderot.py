#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from libyana.conversions import npt
from libyana.lib3d import trans3d
from libyana.renderutils import catmesh, py3drendutils
from libyana.verify.checkshape import check_shape
from libyana.vidutils import np2vid


def seq_render(
    hand_verts,
    hand_faces,
    obj_verts,
    obj_faces,
    camintr,
    hand_color=(199 / 255, 224 / 255, 224 / 255),
    obj_color=(255 / 255, 215 / 255, 224 / 255),
    img_sizes=(640, 480),
    imgs=None,
    video_path="tmp.mp4",
    shading="hard",
    draw_edges=False,
    fps=12,
    loop=True,
    last_idx=40,
):
    """
    Expects lists of verts, faces
    """
    if imgs is not None:
        img_sizes = (imgs[0].shape[1], imgs[0].shape[0])
        check_shape(imgs, (-1, -1, -1, 3), "imgs")
    check_shape(camintr, (3, 3), "camintr")

    # Remove part of video with wrong vertices
    vert_nb = obj_verts[0].shape[0]
    for idx, obj_v in enumerate(obj_verts):
        if obj_v.shape[0] != vert_nb:
            break_last = True
            obj_verts = obj_verts[:idx]
            obj_faces = obj_faces[:idx]
            hand_verts = hand_verts[:idx]
            hand_faces = hand_faces[:idx]
            break
    obj_verts = torch.stack(
        [npt.tensorify(obj_v).cuda() for obj_v in obj_verts])
    hand_verts = torch.stack(
        [npt.tensorify(hand_v).cuda() for hand_v in hand_verts])
    hand_faces = torch.stack(
        [npt.tensorify(hand_f).cuda() for hand_f in hand_faces])
    obj_faces = torch.stack(
        [npt.tensorify(obj_f).cuda() for obj_f in obj_faces])

    check_shape(hand_verts, (-1, -1, 3), "hand_verts")
    check_shape(hand_faces, (-1, -1, 3), "hand_faces")
    check_shape(obj_verts, (-1, -1, 3), "obj_verts")
    check_shape(obj_faces, (-1, -1, 3), "obj_faces")
    hand_colors = torch.Tensor(hand_color).unsqueeze(0).unsqueeze(0).repeat(
        hand_verts.shape[0], hand_verts.shape[1], 1).cuda()
    obj_colors = torch.Tensor(obj_color).unsqueeze(0).unsqueeze(0).repeat(
        obj_verts.shape[0], obj_verts.shape[1], 1).cuda()
    verts, faces, colors = catmesh.batch_cat_meshes([hand_verts, obj_verts],
                                                    [hand_faces, obj_faces],
                                                    [hand_colors, obj_colors])
    camintr = npt.tensorify(camintr).cuda().unsqueeze(0).repeat(
        obj_verts.shape[0], 1, 1)
    with torch.no_grad():
        rends = py3drendutils.batch_render(verts.float(),
                                           faces,
                                           K=camintr.float(),
                                           colors=colors,
                                           image_sizes=[img_sizes],
                                           shading=shading)
        rot_verts = trans3d.rot_points(verts.float(), axisang=(0, 1, 1))
        rot_rends = py3drendutils.batch_render(rot_verts,
                                               faces,
                                               K=camintr.float(),
                                               colors=colors,
                                               image_sizes=[img_sizes],
                                               shading=shading)
        if draw_edges:
            rends = py3drendutils.batch_render(verts.float(),
                                               faces,
                                               K=camintr.float(),
                                               colors=colors,
                                               image_sizes=[img_sizes],
                                               shading=shading)
    # wires = ((rends[:, :, :, -1] > 0.3) & (rends[:, :, :, -1] < 0.9))
    # rends[wires.unsqueeze(-1).repeat(1, 1, 1, 4)] = 0

    if imgs is None:
        imgs = npt.numpify(rends)[:, :, :, :3] * 255
    else:
        imgs = np.concatenate([
            imgs[:last_idx, :, :, :3],
            npt.numpify(rends)[:last_idx, :, :, :3] * 255,
            npt.numpify(rot_rends)[:last_idx, :, :, :3] * 255,
        ], 2)
    if loop:
        vid_imgs = [img for img in imgs] + [img for img in imgs[::-1]]
    else:
        vid_imgs = imgs
    np2vid.make_video([img for img in vid_imgs], video_path, fps=fps)
    return imgs


def rot_render(hand_verts,
               hand_faces,
               obj_verts,
               obj_faces,
               video_path="tmp.mp4",
               hand_color=(0.65, 0.74, 0.85),
               obj_color=(0.9, 0.7, 0.7),
               focal=240,
               img_size=240,
               rot_nb=24,
               fps=12):
    hand_verts = npt.tensorify(hand_verts).cuda().unsqueeze(0)
    obj_verts = npt.tensorify(obj_verts).cuda().unsqueeze(0)
    hand_faces = npt.tensorify(hand_faces).cuda().unsqueeze(0)
    obj_faces = npt.tensorify(obj_faces).cuda().unsqueeze(0)
    check_shape(hand_verts, (1, -1, 3), "hand_verts")
    check_shape(hand_faces, (1, -1, 3), "hand_faces")
    check_shape(obj_verts, (1, -1, 3), "obj_verts")
    check_shape(obj_faces, (1, -1, 3), "obj_faces")
    hand_colors = torch.Tensor(hand_color).unsqueeze(0).repeat(
        hand_verts.shape[1], 1).unsqueeze(0)
    obj_colors = torch.Tensor(obj_color).unsqueeze(0).repeat(
        obj_verts.shape[1], 1).unsqueeze(0)
    verts, faces, colors = catmesh.batch_cat_meshes([hand_verts, obj_verts],
                                                    [hand_faces, obj_faces],
                                                    [hand_colors, obj_colors])
    rot_verts = torch.cat([
        trans3d.rot_points(verts, axisang=(0, rot, 0))
        for rot in np.linspace(0, 2 * np.pi, rot_nb)
    ])
    faces = faces.repeat(rot_nb, 1, 1)
    colors = colors.repeat(rot_nb, 1, 1).to(hand_verts.device)
    camintr = torch.Tensor([[focal, 0,
                             img_size // 2], [0, focal, img_size // 2],
                            [0, 0,
                             1]]).cuda().unsqueeze(0).repeat(rot_nb, 1, 1)
    rends = py3drendutils.batch_render(rot_verts.float(),
                                       faces,
                                       K=camintr.float(),
                                       colors=colors,
                                       image_sizes=[
                                           (img_size, img_size),
                                       ],
                                       shading="hard")

    imgs = npt.numpify(rends)[:, :, :, :3] * 255
    np2vid.make_video([img for img in imgs], video_path, fps=fps)
