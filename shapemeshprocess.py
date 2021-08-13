#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simplify a set of meshes by first transforming to watertight mesh
and then performing subsampling to get a smaller number of vertices

Conversion to a watertight mesh is obtained using Manifold2
(see https://github.com/hjwdzh/ManifoldPlus)
Subsampling is performed by AnisotropicRemeshing (ACVD)
(see https://github.com/valette/ACVD)
Both Manifold2 and ACVD need to be **separately** compiled on your platform
following the instructions to build/compile the programs.

Launch:
    python --manifold_path /path/to/Manifold2/build/manifold
       --acvd_path /path/to/ACVD/bin/ACVD --file_path extra_data/objmodels.txt
"""
import numpy as np
from tqdm import tqdm

from meshprocess import simplifymesh

# pylint: disable=broad-except,too-many-statements,too-many-branches,logging-fstring-interpolation,import-error
import argparse
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--manifold_path",
                    type=str,
                    default="Path to Manifold2 executable",
                    required=True)
parser.add_argument("--acvd_path",
                    type=str,
                    help="Path to ACVD executable",
                    required=True)
parser.add_argument(
    "--vertex_nb",
    type=int,
    help="Target number of vertices when subsampling using ACVD",
    default=1000)
parser.add_argument(
    "--file_path",
    type=str,
    help="Path to file containing list of mesh files to process, where file "
    "contains the path to a mesh in each line",
    default="extra_data/objmodels.txt")
args = parser.parse_args()

# Check that executables exist
if not os.path.exists(args.manifold_path):
    raise RuntimeError(
        f"Couldn't find path to Manifold2 executable at {args.manifold_path}! "
        "Please compile it following the instructions in "
        "https://github.com/hjwdzh/ManifoldPlus")
if not os.path.exists(args.acvd_path):
    raise RuntimeError(
        f"Couldn't find path to ACVD executable at {args.acvd_path}! "
        "Please compile it following the instructions in "
        "https://github.com/valette/ACVD")

with open(args.file_path, "rt") as t_f:
    lines = t_f.readlines()
srcs = np.unique(lines)
srcs = [src.strip() for src in srcs]

for obj_path in tqdm(srcs):
    tar_path = obj_path.replace(".obj", "_proc.obj")
    if not os.path.exists(tar_path.replace(".obj", ".pkl")):
        obj_pkl_path = obj_path.replace(".obj", ".pkl")
        if not os.path.exists(obj_path):
            raise FileNotFoundError(obj_path)
        simplifymesh.simplify_mesh(obj_path,
                                   tar_path,
                                   manifold_path=args.manifold_path,
                                   acvd_path=args.acvd_path,
                                   vert_nb=args.vertex_nb)
