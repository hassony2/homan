#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from homan.utils.nmr_renderer import colors as COLORS


def get_faces_and_textures(verts_list,
                           faces_list,
                           color_names=None,
                           colors_list=None):
    """

    Args:
        verts_list (List[Tensor(B x V x 3)]).
        faces_list (List[Tensor(f x 3)]).

    Returns:
        faces: (1 x F x 3)
        textures: (1 x F x 1 x 1 x 1 x 3)
    """
    if colors_list is None:
        if len(color_names) != len(verts_list):
            raise ValueError(
                f"Invalid number of colors {len(color_names)} for {len(verts_list)} verts"
            )
        if color_names is None:
            colors_list = [
                [251 / 255.0, 128 / 255.0, 114 / 255.0],  # red
                [0.65098039, 0.74117647, 0.85882353],  # blue
                [0.9, 0.7, 0.7],  # pink
            ]
        else:
            colors_list = [COLORS[color_name] for color_name in color_names]
    all_faces_list = []
    all_textures_list = []
    offset = 0
    for verts, faces, colors in zip(verts_list, faces_list, colors_list):
        B = len(verts)
        index_offset = torch.arange(B).to(
            verts.device) * verts.shape[1] + offset
        offset += verts.shape[1] * B
        faces_repeat = faces.clone().repeat(B, 1, 1)
        faces_repeat += index_offset.view(-1, 1, 1)
        faces_repeat = faces_repeat.reshape(-1, 3)
        all_faces_list.append(faces_repeat.long())
        textures = torch.FloatTensor(colors).to(verts.device)
        all_textures_list.append(
            textures.repeat(faces_repeat.shape[0], 1, 1, 1, 1))
    all_faces_list = torch.cat(all_faces_list).unsqueeze(0)
    all_textures_list = torch.cat(all_textures_list).unsqueeze(0)
    return all_faces_list, all_textures_list
