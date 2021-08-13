#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
from pathlib import Path
import pandas as pd
from functools import lru_cache

from homan.datasets import hoaio


def row2box(row):
    return [row.left, row.top, row.right, row.bottom]


def framedet2dicts(det,
                   obj_thresh=0.5,
                   hand_thresh=0.5,
                   height=1080,
                   width=1920):
    res_dict = {"video_id": det.video_id, "frame": det.frame_number}
    dicts = []
    for obj_det in det.objects:
        det_dict = deepcopy(res_dict)
        score = obj_det.score
        if score > obj_thresh:
            det_dict["score"] = score
            det_dict["left"] = obj_det.bbox.left * width
            det_dict["right"] = obj_det.bbox.right * width
            det_dict["top"] = obj_det.bbox.top * height
            det_dict["bottom"] = obj_det.bbox.bottom * height
            det_dict["det_type"] = "object"
            dicts.append(det_dict)
    for hand_det in det.hands:
        det_dict = deepcopy(res_dict)
        score = hand_det.score
        if score > hand_thresh:
            det_dict["score"] = score
        det_dict["score"] = hand_det.score
        det_dict["left"] = hand_det.bbox.left * width
        det_dict["right"] = hand_det.bbox.right * width
        det_dict["top"] = hand_det.bbox.top * height
        det_dict["bottom"] = hand_det.bbox.bottom * height
        det_dict["det_type"] = "hand"
        det_dict["hoa_link"] = str(hand_det.state).split(".")[-1].lower()
        det_dict["side"] = str(hand_det.side).split(".")[-1].lower()
        det_dict["obj_offx"] = hand_det.object_offset.x
        det_dict["obj_offy"] = hand_det.object_offset.y
        dicts.append(det_dict)
    return dicts


@lru_cache(maxsize=128)
def load_video_hoa(video_id, hoa_root):
    """
    Args:
        video_id (str): PXX_XX video id
        hoa_root (str): path to hand-objects folder
            hand-objects
                \\ P01
                   \\ P01_01.pkl
                \\ ...
    """
    hoa_root = Path(hoa_root)
    hoa_list = hoaio.load_detections(hoa_root / video_id[:3] /
                                     f"{video_id}.pkl")
    all_hoa_dicts = []
    for hoa_det in hoa_list:
        hoa_dicts = framedet2dicts(hoa_det)
        all_hoa_dicts.extend(hoa_dicts)
    dat = pd.DataFrame(all_hoa_dicts)
    return dat
