#!/usr/bin/env python
# -*- coding: utf-8 -*-


def pix2nc(points, img_size):
    proj = points + 0.5  # [0, 1]
    proj = proj * img_size  # [0, img_size]
    retunr proj
