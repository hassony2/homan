#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=broad-except,too-many-statements,too-many-branches,logging-fstring-interpolation,import-error,missing-function-docstring
import os
import pickle

import cv2
import numpy as np
import trimesh

from manopth import manolayer

from homan.datasets import collate, core50utils
from homan.datasets.core50constants import MODELS
from homan.datasets.chunkvids import chunk_vid_index


def load_models(models=MODELS, model_root=None, normalize=True):
    models = {}
    for obj_name, obj_info in MODELS.items():
        if "path" in obj_info:
            obj_path = os.path.join(model_root, obj_info["path"])
            obj = trimesh.load(obj_path, force="mesh")
        elif "form" in obj_info:
            obj = trimesh.creation.icosphere(subdivisions=3, radius=1)
            obj_path = f"{obj_info['form']}/models/model_normalized_proc.obj"
            obj_path_create = os.path.join(
                model_root,
                f"{obj_info['form']}/models/model_normalized_proc.obj")
            os.makedirs(os.path.dirname(obj_path_create), exist_ok=True)
            obj.export(obj_path_create)
        verts = np.array(obj.vertices)
        scale = obj_info["scale"]
        if normalize:
            # center
            verts = verts - verts.mean(0)
            # inscribe in 1-radius sphere
            verts = verts / np.linalg.norm(verts, 2, 1).max() * scale / 2
        models[obj_name] = {
            "verts": verts,
            "faces": np.array(obj.faces),
            "path": obj_path,
            "scale": scale,
        }
    return models


class Core50:
    def __init__(
        self,
        root="local_data/datasets",
        use_cache=False,
        mano_root="extra_data/mano",
        objects=None,
        mode="frame",
        load_img=True,
        chunk_step=4,
        frame_nb=10,
        box_folder="data/boxes",
        track=True,
        data_root="local_data/datasets/core50",
        # obj_root="local_data/datasets/ShapeNetCore.v2"):
        obj_root="local_data/datasets/shapenetmodels"):
        """
        Arguments:
            track: used when bbox tracks are collected
        """
        super().__init__()
        self.name = "core50"
        self.frame_nb = frame_nb
        self.mode = mode
        self.object_models = load_models(MODELS, model_root=obj_root)
        self.img_folder = os.path.join(data_root, "core50_350x350")
        self.load_img = load_img
        self.annot_folder = os.path.join(data_root, "core50_350x350_Annot")
        class_dict = {
            'mobile_phone': ['o{}'.format(idx) for idx in range(6, 11)],
            'light_bulb': ['o{}'.format(idx) for idx in range(16, 21)],
            'can': ['o{}'.format(idx) for idx in range(21, 26)],
            'ball': ['o{}'.format(idx) for idx in range(31, 36)],
            'marker': ['o{}'.format(idx) for idx in range(36, 41)],
            'cups': ['o{}'.format(idx) for idx in range(41, 46)],
            'remote_control': ['o{}'.format(idx) for idx in range(46, 51)],
        }
        self.sessions = ['s{}'.format(idx) for idx in range(1, 12)]
        cache_folder = os.path.join("data", "cache")
        os.makedirs(cache_folder, exist_ok=True)
        cache_path = os.path.join(cache_folder, f"{self.name}")
        if os.path.exists(cache_path) and use_cache:
            with open(cache_path, "rb") as p_f:
                annots = pickle.load(p_f)
        else:
            frame_index, vid_index, all_annots = core50utils.build_frame_index(
                self.sessions,
                self.annot_folder,
                objects=list(self.object_models.keys()))
            annots = {
                "frame_index": frame_index,
                "vid_index": vid_index,
                "all_annots": all_annots
            }
            with open(cache_path, "wb") as p_f:
                pickle.dump(annots, p_f)
        self.frame_index = annots["frame_index"]
        self.vid_index = annots["vid_index"]
        self.annotations = annots["all_annots"]

        self.image_size = (350, 350)
        self.track = track
        self.root = os.path.join(root, self.name)
        left_faces = manolayer.ManoLayer(mano_root=mano_root,
                                         side="left").th_faces.numpy()
        right_faces = manolayer.ManoLayer(mano_root=mano_root,
                                          side="right").th_faces.numpy()
        self.faces = {"left": left_faces, "right": right_faces}

        self.chunk_index = chunk_vid_index(self.vid_index,
                                           chunk_size=frame_nb,
                                           chunk_step=chunk_step,
                                           chunk_spacing=chunk_step * frame_nb)
        if not track:
            track_path = os.path.join(box_folder, f"boxes_core50_test.pkl")
            if not os.path.exists(track_path):
                raise ValueError(
                    f"Missing {track_path} file, generate it with"
                    f"python track_boxes.py --dataset core50 --split {objects} --frame_nb -1"
                )
            with open(track_path, "rb") as p_f:
                self.tracked_boxes = pickle.load(p_f)

    def __getitem__(self, idx):
        if self.mode == "vid":
            return self.get_vid_info(idx, mode="full_vid")
        elif self.mode == "chunk":
            return self.get_vid_info(idx, mode="chunk")
        else:
            raise ValueError(f"self.mode {self.mode} not in [vid|chunk]")

    def get_vid_info(self, idx, mode="full_vid"):
        # Use all frames if frame_nb is -1
        if mode == "full_vid":
            vid_info = self.vid_index.iloc[idx]
            # Use all frames if frame_nb is -1
            if self.frame_nb == -1:
                frame_nb = vid_info.frame_nb
            else:
                frame_nb = self.frame_nb

            frame_idxs = np.linspace(0, vid_info.frame_nb - 1,
                                     frame_nb).astype(np.int)
        else:
            vid_info = self.chunk_index.iloc[idx]
            frame_idxs = vid_info.frame_idxs
        # Read images from tar file
        images = []
        seq_obj_info = []
        seq_hand_infos = []
        seq_cameras = []
        sess, obj = vid_info.seq_idx
        for frame_idx in frame_idxs:
            img_path = os.path.join(
                self.img_folder, sess, obj,
                f"C_{int(sess[1:]):02d}_{int(obj[1:]):02d}_{frame_idx:03d}.png"
            )
            if self.load_img:
                try:
                    img = cv2.imread(img_path)[:, :, ::-1]
                except TypeError:
                    print(f"Failed to open {img_path}")
            else:
                img = img_path
            images.append(img)

            obj_info, hand_infos, camera, setup = self.get_hand_obj_info(
                vid_info,
                frame_idx,
                res=self.image_size,
            )
            seq_obj_info.append(obj_info)
            seq_hand_infos.append(hand_infos)
            seq_cameras.append(camera)
        hand_nb = len(seq_hand_infos[0])
        collated_hand_infos = []
        for hand_idx in range(hand_nb):
            collated_hand_info = collate.collate(
                [hand[hand_idx] for hand in seq_hand_infos])
            collated_hand_info['label'] = collated_hand_info['label'][0]
            collated_hand_infos.append(collated_hand_info)

        obs = dict(hands=collated_hand_infos,
                   objects=[collate.collate(seq_obj_info)],
                   camera=collate.collate(seq_cameras),
                   setup=setup,
                   frame_idxs=frame_idxs,
                   images=images,
                   seq_idx=vid_info.seq_idx)

        return obs

    def get_hand_obj_info(self, frame_info, frame, res=350):
        hand_infos = []
        setup = {"objects": 1}

        if frame_info.hand_side == "right":
            verts = (np.random.rand(778, 3) * 0.2) + np.array([0, 0, 0.6])
            faces = self.faces["right"]
            hand_info = dict(
                verts3d=verts.astype(np.float32),
                faces=faces,
                label="right_hand",
            )
            if not self.track:
                hand_info['bbox'] = self.tracked_boxes[
                    frame_info.seq_idx]["right_hand"][frame]
            hand_infos.append(hand_info)
            setup["right_hand"] = 1
        else:
            verts = (np.random.rand(778, 3) * 0.2) + np.array([0, 0, 0.6])
            faces = self.faces["left"]
            hand_info = dict(
                verts3d=verts.astype(np.float32),
                faces=faces,
                label="left_hand",
            )
            if not self.track:
                hand_info['bbox'] = self.tracked_boxes[
                    frame_info.seq_idx]["left_hand"][frame]
            hand_infos.append(hand_info)
            setup["left_hand"] = 1

        camintr = self.get_camintr()
        # Load sequence models
        obj_info = self.object_models[frame_info.obj]
        verts3d = obj_info["verts"] + np.array([0, 0, 0.6])

        obj_info = dict(
            verts3d=verts3d.astype(np.float32),
            faces=obj_info['faces'],
            path=obj_info['path'],
            scale=obj_info['scale'],
            canverts3d=obj_info["verts"].astype(np.float32),
            name=frame_info.obj,
        )
        if not self.track:
            obj_info['bbox'] = self.tracked_boxes[
                frame_info.seq_idx]["objects"][frame]
        camera = dict(
            resolution=[res, res],  # WH
            K=camintr.astype(np.float32),
        )
        return obj_info, hand_infos, camera, setup

    def get_camintr(self):
        focal = 480
        cam_intr = np.array([
            [focal, 0, 350 // 2],
            [0, focal, 350 // 2],
            [0, 0, 1],
        ])
        return cam_intr

    def get_focal_nc(self):
        cam_intr = self.get_camintr()
        return (cam_intr[0, 0] + cam_intr[1, 1]) / 2 / max(self.image_size)

    def __len__(self):
        if self.mode == "vid":
            return len(self.vid_index)
        elif self.mode == "chunk":
            return len(self.chunk_index)
        else:
            raise ValueError(f"{self.mode} mode not in [frame|vid|chunk]")

    def project(self, points3d, cam_intr, camextr=None):
        if camextr is not None:
            points3d = np.array(camextr[:3, :3]).dot(
                points3d.transpose()).transpose()
        hom_2d = np.array(cam_intr).dot(points3d.transpose()).transpose()
        points2d = (hom_2d / hom_2d[:, 2:])[:, :2]
        return points2d.astype(np.float32)
