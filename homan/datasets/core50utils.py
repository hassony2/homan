#!/usr/bin/env python
# -*- coding: utf-8 -*-
# See https://github.com/hassony2/obman_train/blob/master/handobjectdatasets/core50.py
# pylint: disable=broad-except,too-many-statements,too-many-branches,logging-fstring-interpolation,import-error,missing-function-docstring
import os

import numpy as np
import pandas as pd
from scipy.io import loadmat
from tqdm import tqdm

from homan.datasets.core50constants import SIDES


def load_annot(annot_path, scale_factor=1.2):
    annot_raw = loadmat(annot_path)
    hand_root2d = annot_raw['annot']['hand'][0, 0]['root2d'][0, 0]
    hand_root_depth_png = annot_raw['annot']['hand'][0, 0]['root_depth_png'][0,
                                                                             0]
    hand_depth = 8000 * (255 - hand_root_depth_png) / 1000 / 256

    obj_root2d = annot_raw['annot']['object'][0, 0]['root2d'][0, 0]
    obj_root_depth = annot_raw['annot']['object'][0, 0]['root_depth_png'][0, 0]
    # x_min, y_min, x_max, y_max hand + object bbox
    bbox = annot_raw['annot']['crop'][0, 0]
    side_code = annot_raw['annot']['hand'][0, 0]['side'][0, 0][0]
    if side_code == 'R':
        side = 'right'
    elif side_code == 'L':
        side = 'left'
    center = np.array([(bbox[0, 0] + bbox[0, 2]) / 2,
                       (bbox[0, 1] + bbox[0, 3]) / 2])
    scale = scale_factor * np.array(
        [bbox[0, 2] - bbox[0, 0], bbox[0, 3] - bbox[0, 1]])
    annot_name = os.path.basename(annot_path)
    frame_idx = int(annot_name.split(".")[0].split("_")[3])
    prefix = '_'.join(annot_name.split('.')[0].split('_')[1:])
    rgb_path = os.path.join(os.path.dirname(annot_path.replace("_Annot", "")),
                            f"C_{prefix}.png")
    annot = {
        "scale": scale,
        "center": center,
        "bbox": bbox,
        "side": side,
        "frame_idx": frame_idx,
        "hand_root2d": hand_root2d,
        "hand_depth": hand_depth,
        "obj_root2d": obj_root2d,
        "obj_root_depth": obj_root_depth,
        "img": rgb_path,
        "prefix": prefix,
    }
    return annot


def build_frame_index(sessions, annot_folder, objects=None):
    all_annots = {}
    vid_index = []
    frame_index = []
    with open(
            "/gpfsscratch/rech/tan/usk19gv/datasets/core50/core50_350x350/tmp.html",
            "w") as t_f:
        # One session per background
        for session in tqdm(sessions, desc="session"):
            # Different objects
            for obj in tqdm(objects, desc="object"):
                if objects is not None and obj not in objects:
                    continue
                sess_path = os.path.join(annot_folder, session)
                img_folder = sess_path.replace("_Annot", "")
                obj_path = os.path.join(sess_path, obj)
                vid_key = (session, obj)
                # Count image number
                img_folder_obj = os.path.join(img_folder, obj)
                frame_nb = len(os.listdir(img_folder_obj))
                if os.path.exists(obj_path):
                    obj_annots = sorted([
                        annot for annot in os.listdir(obj_path)
                        if annot.endswith(".mat")
                    ])
                    for obj_annot in obj_annots:
                        annot_path = os.path.join(obj_path, obj_annot)
                        annot_info = load_annot(annot_path)
                        annot_info["frame_nb"] = frame_nb
                        annot_info["obj"] = obj
                        annot_info["session"] = session
                        frame_key = (session, obj, annot_info["frame_idx"])
                        frame_index.append({
                            "frame_idx": annot_info["frame_idx"],
                            "obj": obj,
                            "session": session,
                            "frame_nb": frame_nb,
                            "seq_idx": vid_key
                        })
                        all_annots[frame_key] = annot_info
                else:
                    prefix = f"{int(session[1:]):02d}_{int(obj[1:]):02d}_000"
                    img_path = f"{session}/{obj}/C_{prefix}.png"
                    hand_side = SIDES[session]
                    t_f.write(f"<img src='{img_path}'>")
                    annot_info = {"prefix": prefix, "side": hand_side}
                vid_index.append({
                    "frame_nb": frame_nb,
                    "obj": obj,
                    "session": session,
                    "hand_side": annot_info["side"],
                    "seq_idx": vid_key,
                    "prefix": annot_info["prefix"]
                })
    frame_index = pd.DataFrame(frame_index)
    vid_index = pd.DataFrame(vid_index)
    return frame_index, vid_index, all_annots
