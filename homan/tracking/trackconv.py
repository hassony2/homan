#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def track2dicts(tracks, frame_idx, video_id, det_type="object", side=np.nan):
    track_dicts = []
    for track in tracks:
        track_dict = dict(
            track_id=track.id,
            left=track.box[0],
            top=track.box[1],
            right=track.box[2],
            bottom=track.box[3],
            det_type=det_type,
            side=side,
            frame=frame_idx,
            video_id=video_id,
            hoa_link=np.nan,
            score=1,
        )
        track_dicts.append(track_dict)
    return track_dicts
