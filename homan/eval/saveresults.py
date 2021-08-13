#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pickle


def dump(args, metrics, path):
    res_bundle = {
        "opts": vars(args),
        "metrics": metrics,
    }
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    with open(path, "wb") as p_f:
        pickle.dump(res_bundle, p_f)
    print(f"Saved results to {path}")
