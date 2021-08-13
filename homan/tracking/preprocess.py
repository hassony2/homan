#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=broad-except,too-many-statements,too-many-branches,logging-fstring-interpolation,import-error
from PIL import Image
import numpy as np


def get_image(image, image_size):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        pass
    else:
        raise TypeError(
            f"Expected type {type(image)} of image {image} to be in [str|np.ndarray|PIL.Image.Image]"
        )

    width, height = image.size
    radius = min(image_size / width, image_size / height)
    new_width = int(radius * width)
    new_height = int(radius * height)
    new_image = np.array(image.resize((new_width, new_height)))
    return new_image
