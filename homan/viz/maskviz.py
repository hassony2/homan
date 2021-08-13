#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch.nn import functional as F
from libyana.verify import checkshape


def get_show_mask(mask,
                  border_factor=4,
                  color=(1, 0, 0),
                  border_color=(1, 1, 1),
                  alpha=0.5):
    # Downscale and upscale to get borders
    checkshape.check_shape(mask, (-1, -1), "mask")
    reduced = F.interpolate(
        mask.unsqueeze(0).unsqueeze(0).float(),
        (mask.shape[0] // border_factor, mask.shape[1] // border_factor),
        mode="bilinear")
    inflated = (F.interpolate(reduced, (mask.shape[0], mask.shape[1]),
                              mode="bilinear")[0, 0] > 0)
    show_mask = mask.new_zeros(mask.shape[0], mask.shape[1], 4).float()
    for color_idx in range(3):
        border_idxs = torch.where(inflated)
        show_mask[border_idxs[0], border_idxs[1], color_idx *
                  torch.ones_like(border_idxs[0])] = border_color[color_idx]
        color_idxs = torch.where(mask)
        show_mask[color_idxs[0], color_idxs[1], color_idx *
                  torch.ones_like(color_idxs[0])] = color[color_idx]
    show_mask[border_idxs[0], border_idxs[1], 3] = 1
    show_mask[color_idxs[0], color_idxs[1], 3] = alpha
    return show_mask
