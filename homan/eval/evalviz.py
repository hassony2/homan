#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
from moviepy import editor


def make_gif(img_paths, gif_path, fps=2):
    img_paths = [str(path) for path in img_paths]
    clip = editor.ImageSequenceClip(img_paths, fps=fps)
    clip.write_gif(gif_path)


def make_video(img_paths, video_path, fps=2, resize_factor=1):
    img_paths = [str(path) for path in img_paths]
    if resize_factor != 1:
        imgs = []
        for img_path in img_paths:
            img = cv2.imread(img_path)[:, :, ::-1]
            img = cv2.resize(
                img,
                (
                    int(img.shape[1] * resize_factor),
                    int(img.shape[0] * resize_factor),
                ),
            )
            imgs.append(img)
    else:
        imgs = img_paths
    clip = editor.ImageSequenceClip(imgs, fps=fps)
    clip.write_videofile(str(video_path))


def make_video_np(imgs, video_path, resize_factor=1, fps=4):
    if resize_factor != 1:
        resize_imgs = []
        for img in imgs:
            img = cv2.resize(
                img,
                (
                    int(img.shape[1] * resize_factor),
                    int(img.shape[0] * resize_factor),
                ),
            )
            resize_imgs.append(img)
        imgs = resize_imgs
    clip = editor.ImageSequenceClip(imgs, fps=fps)
    clip.write_videofile(str(video_path))
