#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2


def get_frame_by_idx(video_path, frame, cap=None, invert_channels=True):
    if not os.path.exists(video_path):
        raise ValueError(f"{video_path} not found")
    if cap is None:
        cap = cv2.VideoCapture(video_path)
        if cap is None:
            raise ValueError(
                f"Could not read {video_path} with cv2.VideoCapture")
    cv2_frame_nb = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, image = cap.read()
    if not ret:
        raise ValueError(
            f"Couldn't read frame {frame} of {video_path} with cv2_frame_nb {cv2_frame_nb}"
        )
    if invert_channels:
        image = image[:, :, ::-1]
    return image, cap


def get_frames_by_idxs(video_path, frames, invert_channels=True, strict=False):
    images = []
    if not os.path.exists(video_path):
        raise ValueError(f"{video_path} not found")
    cap = cv2.VideoCapture(video_path)
    if cap is None:
        raise ValueError(f"Could not read {video_path} with cv2.VideoCapture")
    cv2_frame_nb = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    for frame in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, image_read = cap.read()
        if strict and (not ret):
            raise ValueError(
                f"Couldn't read frame {frame} of {video_path} with cv2_frame_nb {cv2_frame_nb}"
            )
        elif (not ret):
            pass
        else:
            image = image_read

        if invert_channels:
            image_new = image[:, :, ::-1].copy()
        else:
            image_new = image
        images.append(image_new)
    return images, cap
