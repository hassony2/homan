#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from matplotlib import pyplot as plt
from libyana.visutils import detect2d
from libyana.conversions import npt
from libyana.visutils import vizmp

from homan.utils.bbox import bbox_xy_to_wh, bbox_wh_to_xy, make_bbox_square


def viz_frame_info(frame_info, sample_folder="tmp", save=True):
    plt.close()
    fig, axes = plt.subplots(3, 1, figsize=(3, 8))
    image = frame_info['image']
    obj_infos = frame_info["obj_mask_infos"]
    if "person_parameters" in frame_info and len(
            frame_info["person_parameters"]):
        person_params = frame_info["person_parameters"]
    else:
        person_params = None
    mask_bboxes = obj_infos["bbox"]

    # Image with hand/object detections
    ax = axes[0]
    ax.axis("off")
    ax.imshow(image)
    ax.set_title("detections")
    detect2d.visualize_bbox(ax,
                            bbox_wh_to_xy(mask_bboxes),
                            label="obj",
                            color="g",
                            linewidth=1)
    if person_params is not None:
        show_hands = True
        detect2d.visualize_bboxes(ax,
                                  npt.numpify(person_params["bboxes"]),
                                  linewidth=1,
                                  labels=["hand"] *
                                  len(person_params["bboxes"]),
                                  color="r")
    # Object mask overlay
    ax = axes[1]
    ax.axis("off")
    ax.imshow(image)
    ax.imshow(npt.numpify(obj_infos["full_mask"]), alpha=0.5)
    ax.set_title("object masks")
    # Human mask overlay
    if person_params is not None:
        human_masks = person_params['masks'].sum(0)
        ax = axes[2]
        ax.axis("off")
        ax.imshow(image)
        ax.imshow(npt.numpify(human_masks), alpha=0.5)
    ax.set_title("hand masks")
    if save:
        super2d_img_path = os.path.join(sample_folder, "detections_masks.png")
        fig.savefig(super2d_img_path, bbox_inches="tight")
        plt.close()
        print(f"Saved frame viz to {super2d_img_path}")
        return super2d_img_path
    else:
        img = vizmp.fig2np(fig)
        plt.close()
        return img
