#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from homan.datasets import collate, epichoa, tarutils
from homan.datasets.chunkvids import chunk_vid_index
from homan.tracking import trackhoa as trackhoadf
from homan.utils import bbox as bboxutils

import os
import pandas as pd
import pickle
import trimesh
import warnings
from libyana.lib3d import kcrop
from libyana.transformutils import handutils
from manopth import manolayer

MODELS = {
    "bottle": {
        "path": "/gpfsscratch//rech/tan/usk19gv/datasets/ShapeNetCore.v2/"
        "02876657/d851cbc873de1c4d3b6eb309177a6753/models/model_normalized_proc.obj",
        "scale": 0.2,
    },
    "jug": {
        "path":
        "local_data/datasets/ho3dv2/processmodels/019_pitcher_base/textured_simple_400.obj",
        "scale": 0.25,
    },
    "pitcher": {
        "path":
        "local_data/datasets/ho3dv2/processmodels/019_pitcher_base/textured_simple_400.obj",
        "scale": 0.25,
    },
    "plate": {
        "path": "/gpfsscratch//rech/tan/usk19gv/datasets/ShapeNetCore.v2/"
        "02880940/95ac294f47fd7d87e0b49f27ced29e3/models/model_normalized_proc.obj",
        "scale": 0.3,
    },
    "cup": {
        "path": "/gpfsscratch//rech/tan/usk19gv/datasets/ShapeNetCore.v2/"
        "03797390/d75af64aa166c24eacbe2257d0988c9c/models/model_normalized_proc.obj",
        "scale": 0.12,
    },
    "phone": {
        "path": "/gpfsscratch//rech/tan/usk19gv/datasets/ShapeNetCore.v2/"
        "02992529/7ea27ed05044031a6fe19ebe291582/models/model_normalized_proc.obj",
        "scale": 0.07
    },
    "can": {
        "path": "/gpfsscratch//rech/tan/usk19gv/datasets/ShapeNetCore.v2/"
        "02946921/3fd8dae962fa3cc726df885e47f82f16/models/model_normalized_proc.obj",
        "scale": 0.2
    }
}


def apply_bbox_transform(bbox, affine_trans):
    x_min, y_min = handutils.transform_coords(
        [bbox[:2]],
        affine_trans,
    )[0]
    x_max, y_max = handutils.transform_coords(
        [bbox[2:]],
        affine_trans,
    )[0]
    new_bbox = np.array([x_min, y_min, x_max, y_max])
    return new_bbox


def load_models(MODELS, normalize=True):
    models = {}
    for obj_name, obj_info in MODELS.items():
        obj_path = obj_info["path"]
        scale = obj_info["scale"]
        obj = trimesh.load(obj_path)
        verts = np.array(obj.vertices)
        if normalize:
            # center
            verts = verts - verts.mean(0)
            # inscribe in 1-radius sphere
            verts = verts / np.linalg.norm(verts, 2, 1).max() * scale / 2
        models[obj_name] = {
            "verts": verts,
            "faces": np.array(obj.faces),
            "path": obj_path,
        }
    return models


class Epic:
    def __init__(
        self,
        root="local_data/datasets",
        joint_nb=21,
        use_cache=False,
        mano_root="extra_data/mano",
        mode="frame",
        ref_idx=0,
        valid_step=-1,
        frame_step=1,
        frame_nb=10,
        verbs=[  # "take", "put",
            "take", "put"
            "open", "close"
        ],
        nouns=[
            # "can",
            # "cup",
            # "phone",
            "plate",
            # "pitcher",
            # "jug",
            # "bottle",
        ],
        box_folder="data/boxes",
        track_padding=10,
        min_frame_nb=20,
        epic_root="local_data/datasets/epic",
    ):
        """
        Arguments:
            min_frame_nb (int): Only sequences with length at least min_frame_nb are considered
            track_padding (int): Number of frames to include before and after extent of action clip
                provided by the annotations
            frame_step (int): Number of frames to skip between two selected images
            verbs (list): subset of action verbs to consider
            nouns (list): subset of action verbs to consider
        """
        super().__init__()
        self.name = "epic"
        self.mode = mode
        self.object_models = load_models(MODELS)
        self.frame_template = os.path.join(epic_root, "frames",
                                           "{}/{}/{}/frame_{:010d}.jpg")

        self.resize_factor = 3
        self.frame_nb = frame_nb
        self.image_size = (640, 360)
        cache_folder = os.path.join("data", "cache")
        os.makedirs(cache_folder, exist_ok=True)

        self.root = os.path.join(root, self.name)
        left_faces = manolayer.ManoLayer(mano_root="extra_data/mano",
                                         side="left").th_faces.numpy()
        right_faces = manolayer.ManoLayer(mano_root="extra_data/mano",
                                          side="right").th_faces.numpy()
        self.faces = {"left": left_faces, "right": right_faces}

        cache_path = 'data/cache/epic_take_putopen_close_can_cup_phone_plate_pitcher_jug_bottle_20.pkl'
        if os.path.exists(cache_path) and use_cache:
            with open(cache_path, "rb") as p_f:
                dataset_annots = pickle.load(p_f)
            vid_index = dataset_annots["vid_index"]
            annotations = dataset_annots["annotations"]
        else:
            with open("local_data/datasets/epic/EPIC_100_train.pkl",
                      "rb") as p_f:
                annot_df = pickle.load(p_f)

            # annot_df = annot_df[annot_df.video_id.str.len() == 6]

            annot_df = annot_df[annot_df.noun.isin(nouns)]
            print(f"Processing {annot_df.shape[0]} clips for nouns {nouns}")
            vid_index = []
            annotations = {}
            for annot_idx, (annot_key,
                            annot) in enumerate(tqdm(annot_df.iterrows())):
                try:
                    hoa_dets = epichoa.load_video_hoa(
                        annot.video_id,
                        hoa_root="local_data/datasets/epic/hoa")
                    frame_idxs, bboxes = trackhoadf.track_hoa_df(
                        hoa_dets,
                        video_id=annot.video_id,
                        start_frame=max(1, annot.start_frame - track_padding),
                        end_frame=(min(annot.stop_frame + track_padding,
                                       hoa_dets.frame.max() - 1)),
                        dt=frame_step / 60,
                    )
                    if len(frame_idxs) > min_frame_nb:
                        annot_full_key = (annot.video_id, annot_idx, annot_key)
                        vid_index.append({
                            "seq_idx": annot_full_key,
                            "frame_nb": len(frame_idxs),
                            "start_frame": min(frame_idxs),
                            "object": annot.noun,
                            "verb": annot.verb,
                        })
                        annotations[annot_full_key] = {
                            "bboxes_xyxy": bboxes,
                            "frame_idxs": frame_idxs
                        }
                except Exception:
                    print(f"Skipping idx {annot_idx}")
            vid_index = pd.DataFrame(vid_index)
            dataset_annots = {
                "vid_index": vid_index,
                "annotations": annotations,
            }
            with open(cache_path, "wb") as p_f:
                pickle.dump(dataset_annots, p_f)

        self.annotations = annotations
        self.tareader = tarutils.TarReader()
        self.vid_index = vid_index
        self.chunk_index = chunk_vid_index(self.vid_index,
                                           chunk_size=frame_nb,
                                           chunk_step=frame_step,
                                           chunk_spacing=frame_step * frame_nb)
        self.chunk_index = self.chunk_index[self.chunk_index.object.isin(
            nouns)]
        print(f"Working with {len(self.chunk_index)} chunks for {nouns}")

        # Get paired links as neighboured joints
        self.links = [
            (0, 1, 2, 3, 4),
            (0, 5, 6, 7, 8),
            (0, 9, 10, 11, 12),
            (0, 13, 14, 15, 16),
            (0, 17, 18, 19, 20),
        ]

    def get_roi(self, video_annots, frame_ids, res=640):
        """
        Get square ROI in xyxy format
        """
        # Get all 2d points and extract bounding box with given image
        # ratio
        annots = self.annotations[video_annots.seq_idx]
        bboxes = [bboxs[frame_ids] for bboxs in annots["bboxes_xyxy"].values()]
        all_vid_points = np.concatenate(list(bboxes)) / self.resize_factor
        xy_points = np.concatenate(
            [all_vid_points[:, :2], all_vid_points[:, 2:]], 0)
        mins = xy_points.min(0)
        maxs = xy_points.max(0)
        roi_box_raw = np.array([mins[0], mins[1], maxs[0], maxs[1]])
        roi_bbox = bboxutils.bbox_wh_to_xy(
            bboxutils.make_bbox_square(bboxutils.bbox_xy_to_wh(roi_box_raw),
                                       bbox_expansion=0.2))
        roi_center = (roi_bbox[:2] + roi_bbox[2:]) / 2
        # Assumes square bbox
        roi_scale = roi_bbox[2] - roi_bbox[0]
        affine_trans = handutils.get_affine_transform(roi_center, roi_scale,
                                                      [res, res])[0]
        return roi_bbox, affine_trans

    def __getitem__(self, idx):
        if self.mode == "vid":
            return self.get_vid_info(idx, mode="full_vid")
        elif self.mode == "chunk":
            return self.get_vid_info(idx, mode="chunk")

    def get_vid_info(self, idx, res=640, mode="full_vid"):
        # Use all frames if frame_nb is -1
        if mode == "full_vid":
            vid_info = self.vid_index.iloc[idx]
            # Use all frames if frame_nb is -1
            if self.frame_nb == -1:
                frame_nb = vid_info.frame_nb
            else:
                frame_nb = self.frame_nb

            frame_ids = np.linspace(0, vid_info.frame_nb - 1,
                                    frame_nb).astype(np.int)
        else:
            vid_info = self.chunk_index.iloc[idx]
            frame_ids = vid_info.frame_idxs
        seq_frame_idxs = [self.annotations[vid_info.seq_idx]["frame_idxs"]][0]
        frame_idxs = [seq_frame_idxs[frame_id] for frame_id in frame_ids]
        video_id = vid_info.seq_idx[0]

        # Read images from tar file
        images = []
        seq_obj_info = []
        seq_hand_infos = []
        seq_cameras = []
        roi, affine_trans = self.get_roi(vid_info, frame_ids)
        for frame_id in frame_ids:
            frame_idx = seq_frame_idxs[frame_id]
            img_path = self.frame_template.format("train", video_id[:3],
                                                  video_id, frame_idx)
            img = self.tareader.read_tar_frame(img_path)
            img = cv2.resize(self.tareader.read_tar_frame(img_path),
                             self.image_size)
            img = Image.fromarray(img[:, :, ::-1])

            img = handutils.transform_img(img, affine_trans, [res, res])
            images.append(img)

            obj_info, hand_infos, camera, setup = self.get_hand_obj_info(
                vid_info,
                frame_id,
                roi=roi,
                res=res,
                affine_trans=affine_trans)
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

    def get_hand_obj_info(self,
                          frame_info,
                          frame,
                          res=640,
                          roi=None,
                          affine_trans=None):
        hand_infos = []
        video_annots = self.annotations[frame_info.seq_idx]
        bbox_names = video_annots['bboxes_xyxy'].keys()
        bbox_infos = video_annots['bboxes_xyxy']
        setup = {"objects": 1}
        for bbox_name in bbox_names:
            setup[bbox_name] = 1
        has_left = "left_hand" in bbox_names
        has_right = "right_hand" in bbox_names

        if has_right:
            bbox = bbox_infos['right_hand'][frame] / self.resize_factor
            bbox = apply_bbox_transform(bbox, affine_trans)
            verts = np.random.random()
            verts = (np.random.rand(778, 3) * 0.2) + np.array([0, 0, 0.6])
            faces = self.faces["right"]
            hand_info = dict(
                verts3d=verts.astype(np.float32),
                faces=faces,
                label="right_hand",
                bbox=bbox.astype(np.float32),
            )
            hand_infos.append(hand_info)
        if has_left:
            bbox = bbox_infos['left_hand'][frame] / self.resize_factor
            bbox = apply_bbox_transform(bbox, affine_trans)
            verts = (np.random.rand(778, 3) * 0.2) + np.array([0, 0, 0.6])
            faces = self.faces["left"]
            hand_info = dict(
                verts3d=verts.astype(np.float32),
                faces=faces,
                label="left_hand",
                bbox=bbox.astype(np.float32),
            )
            hand_infos.append(hand_info)

        K = self.get_camintr()
        K = kcrop.get_K_crop_resize(
            torch.Tensor(K).unsqueeze(0), torch.Tensor([roi]),
            [res])[0].numpy()
        obj_info = self.object_models[frame_info.object]
        obj_bbox = bbox_infos["objects"][frame] / self.resize_factor
        obj_bbox = apply_bbox_transform(obj_bbox, affine_trans)
        verts3d = obj_info["verts"] + np.array([0, 0, 0.6])

        obj_info = dict(verts3d=verts3d.astype(np.float32),
                        faces=obj_info['faces'],
                        path=obj_info['path'],
                        canverts3d=obj_info["verts"].astype(np.float32),
                        bbox=obj_bbox)
        camera = dict(
            resolution=[res, res],  # WH
            K=K.astype(np.float32),
        )
        return obj_info, hand_infos, camera, setup

    def get_camintr(self):
        focal = 200
        cam_intr = np.array([
            [focal, 0, 640 // 2],
            [0, focal, 360 // 2],
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
            points3d = np.array(self.camextr[:3, :3]).dot(
                points3d.transpose()).transpose()
        hom_2d = np.array(cam_intr).dot(points3d.transpose()).transpose()
        points2d = (hom_2d / hom_2d[:, 2:])[:, :2]
        return points2d.astype(np.float32)
