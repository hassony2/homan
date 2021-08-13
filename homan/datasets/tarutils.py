#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import tarfile

import cv2
import numpy as np


class TarReader():
    def __init__(self):
        self.last_tar_path = None
        self.tarf = None

    def read_tar_frame(self, frame_path):
        base_tar_path, filename = tar_from_frame_path(frame_path)

        # Lazy loading of tar file
        if base_tar_path != self.last_tar_path:
            load_tarf = tarfile.open(base_tar_path)
            self.tarf = load_tarf
        tarf = self.tarf

        self.last_tar_path = base_tar_path

        # Read image
        img = cv2_imread_tar(tarf, './' + filename)
        return img


def tar_from_frame_path(frame_path):
    base_path = os.path.dirname(frame_path)
    filename = os.path.basename(frame_path)
    folders = base_path.split('/')
    folders = folders[:-3] + folders[-2:-1] + ["rgb_frames"] + folders[-1:]
    base_path = '/'.join(folders)
    return f"{base_path}.tar", filename


def get_np_array_from_tar_object(tar_extractfl):
    """converts a buffer from a tar file in np.array"""
    return np.asarray(bytearray(tar_extractfl.read()), dtype=np.uint8)


def cv2_imread_tar(tar_ref, filename):
    np_dec = get_np_array_from_tar_object(tar_ref.extractfile(filename))
    frame = cv2.imdecode(np_dec, cv2.IMREAD_COLOR)
    return frame
