#!/usr/bin/env python
# -*- coding: utf-8 -*-


def check_setup(bboxes, setup):
    valid_setup = True
    computed_setup = {}
    valid_boxes = {}
    for item, item_nb in setup.items():
        if bboxes[0][item] is None:
            valid_setup = False
        else:
            if bboxes[0][item].ndim == 2:
                box_nb = bboxes[0][item].shape[0]
                if box_nb != item_nb:
                    valid_setup = False
            else:
                box_nb = 1
            computed_setup[item] = box_nb
            valid_boxes[item] = bboxes[0][item]
    return valid_setup, computed_setup, valid_boxes
