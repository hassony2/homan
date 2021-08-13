import os
import pickle

import pandas as pd
from tqdm import tqdm

from libyana.meshutils import meshio


def load_objects(obj_root):
    object_names = [
        obj_name for obj_name in os.listdir(obj_root) if ".tgz" not in obj_name
    ]
    objects = {}
    for obj_name in object_names:
        obj_path = os.path.join(obj_root, obj_name, "textured_simple_2000.obj")
        with open(obj_path) as m_f:
            mesh = meshio.fast_load_obj(m_f)[0]
        objects[obj_name] = {"verts": mesh["vertices"], "faces": mesh["faces"]}
    return objects


def build_frame_index(seqs, root, subfolder="train", use_cache=True, cache_folder="local_data/datasets/cache"):
    cache_path = os.path.join(cache_folder, f"{subfolder}.pkl")
    if not os.path.exists(cache_path):
        os.makedirs(cache_folder, exist_ok=True)
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "rb") as p_f:
            pd_frame_index, annotations_map = pickle.load(p_f)
    else:
        annotations_map = {}
        annotations_list = []
        seq_lengths = {}
        # for seq in tqdm(sorted(seqs), desc="seq"):
        scene_id = 0
        for seq in tqdm(sorted(seqs)):
            seq_folder = os.path.join(root, subfolder, seq)
            meta_folder = os.path.join(seq_folder, "meta")
            rgb_folder = os.path.join(seq_folder, "rgb")

            img_nb = len(os.listdir(meta_folder))
            seq_lengths[seq] = img_nb
            # for frame_idx in tqdm(range(img_nb), desc="img"):
            for frame_idx in range(img_nb):
                meta_path = os.path.join(meta_folder, f"{frame_idx:04d}.pkl")
                with open(meta_path, "rb") as p_f:
                    annot = pickle.load(p_f)
                img_path = os.path.join(rgb_folder, f"{frame_idx:04d}.png")
                annot["img"] = img_path
                annot["seq_idx"] = seq
                annot["frame_idx"] = frame_idx
                annot["frame_nb"] = img_nb
                annot["scene_id"] = scene_id
                annot["view_id"] = 0
                scene_id += 1
                annotations_map[(seq, frame_idx)] = annot
                annotations_list.append(annot)
        pd_frame_index = pd.DataFrame(
                annotations_list).loc[:, ["img", "seq_idx", "frame_idx", "frame_nb"]]
        with open(cache_path, "wb") as p_f:
            pickle.dump((pd_frame_index, annotations_map), p_f)
    return pd_frame_index, annotations_map
