#!/usr/bin/env python
# -*- coding: utf-8 -*-
from copy import deepcopy
import pandas as pd


def chunk_vid_index(vid_index,
                    chunk_size=10,
                    chunk_spacing=200,
                    chunk_step=4,
                    frame_col_name="frame_nb",
                    use_frame_start=False):
    """
    Samples subsets of frames from video.

    Collect 'chunks' of frames by sampling chunk_size frames spaced by chunk_step
    every chunk_spacing.
    """
    chunk_extent = chunk_size * chunk_step
    print(f"Chunking {len(vid_index)} videos in chunks of size {chunk_size}, "
          f"skipping {chunk_step}, every {chunk_spacing} frames")
    chunk_dicts = []
    for _, row in vid_index.iterrows():
        row_dict = row.to_dict()
        frame_nb = row_dict[frame_col_name]
        if use_frame_start:
            frame_start = row_dict["frame_start"]
        else:
            frame_start = 0
        start_idxs = list(
            range(frame_start, frame_nb - chunk_extent, chunk_spacing))
        # Make sure end of video is also covered
        start_idxs.append(frame_nb - chunk_extent + chunk_step - 1)
        for start_idx in start_idxs:
            frame_idxs = [
                start_idx + chunk_step * idx for idx in range(chunk_size)
            ]
            chunk_dict = deepcopy(row_dict)
            chunk_dict["frame_idxs"] = frame_idxs
            chunk_dicts.append(chunk_dict)
    new_vid_index = pd.DataFrame(chunk_dicts)
    return new_vid_index
