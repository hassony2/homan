#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable=wrong-import-order,too-many-locals
import numpy as np

import json
import os
import shutil
import subprocess
from collections import defaultdict
from libyana.conversions import npt
from libyana.renderutils import py3drendutils


def dump(pred_out_path, xyz_pred_list, verts_pred_list, codalab=True):
    """ Save predictions into a json file for official ho3dv2 evaluation. """

    xyz_pred_list = [x.round(4).tolist() for x in xyz_pred_list]
    verts_pred_list = [x.round(4).tolist() for x in verts_pred_list]

    # save to a json
    print(f"Dumping json results to {pred_out_path}")
    with open(pred_out_path, "w") as fo:
        json.dump([xyz_pred_list, verts_pred_list], fo)
    print("Dumped %d joints and %d verts predictions to %s" %
          (len(xyz_pred_list), len(verts_pred_list), pred_out_path))
    if codalab:
        save_zip_path = pred_out_path.replace(".json", ".zip")
        subprocess.call(["zip", "-j", save_zip_path, pred_out_path])
        print(f"Saved results to {save_zip_path}")
        shutil.copy(save_zip_path, "pred.zip")
        print("Copied results to pred.zip")


def extend_res(seq_res,
               frame_nb,
               keys=[
                   "hand_verts3d", "hand_joints3d", "obj_verts3d",
                   "hand_roots", "obj_faces"
               ]):
    seq_keys = sorted(list(seq_res.keys()))
    img_root = os.path.dirname(seq_res[0]["img_path"])
    full_res = defaultdict(list)
    for frame_idx in range(frame_nb):
        for key_idx, key in enumerate(keys):
            full_res[key].append(seq_res[frame_idx][key])
            full_res["img_paths"].append(
                os.path.join(img_root, f"{frame_idx:04d}.png"))
    return dict(full_res)


def interpolate_res(
        seq_res,
        frame_nb,
        keys=["hand_verts3d", "hand_joints3d", "obj_verts3d", "hand_roots"]):
    """
    Get interpolated results from seq_res
    """
    interp_res = defaultdict(list)
    seq_keys = sorted(list(seq_res.keys()))
    img_root = os.path.dirname(seq_res[0]["img_path"])
    for key_idx, key in enumerate(keys):
        frame_count = 0
        for key_start, key_end in zip(seq_keys[:-1], seq_keys[1:]):
            interp_weights = np.linspace(0, 1, key_end - key_start + 1)
            start_val = seq_res[key_start][key]
            end_val = seq_res[key_end][key]
            interp_vals = (start_val +
                           ((end_val - start_val) *
                            interp_weights[:, np.newaxis, np.newaxis]))
            for interp_idx in range(key_end - key_start):
                interp_res[key].append(interp_vals[interp_idx])
                if key_idx == 0:
                    interp_res["img_paths"].append(
                        os.path.join(img_root, f"{frame_count:04d}.png"))
                    frame_count += 1

        # Pad with final value
        key_end = seq_keys[-1]
        for _ in range(key_end, frame_nb):
            interp_res[key].append(end_val)
        if key_idx == 0:
            interp_res["img_paths"].append(
                os.path.join(img_root, f"{frame_count:04d}.png"))
            frame_count += 1
    for key in keys:
        try:
            if len(interp_res[key]) != frame_nb:
                raise RuntimeError(
                    f"Expected to get as many interpolation results {len(interp_res[key])} for key {key} as frames {frame_nb}"
                )
        except RuntimeError:
            import pudb
            pu.db
    return dict(interp_res)
