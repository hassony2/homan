import os
import pickle
import warnings

import cv2
import numpy as np
from PIL import Image
import torch

from manopth import manolayer

from homan.datasets import ho3dfullutils, ho3dutils, collate, ho3dconstants
from homan.datasets.chunkvids import chunk_vid_index
from homan.datasets import manoutils


class HO3D:
    def __init__(
        self,
        split,
        root="local_data/datasets",
        joint_nb=21,
        use_cache=False,
        mano_root="extra_data/mano",
        mode="frame",
        ref_idx=0,
        valid_step=-1,
        frame_nb=1,
        track=False,
        chunk_size=10,
        chunk_step=4,
        box_mode="gt",
        box_folder="data/boxes",
        ycb_root="local_data/datasets/ycbmodels",
        load_img=True,
    ):
        """
        Args:
            track: used when bbox tracks are collected
        """
        super().__init__()

        self.name = "ho3d"
        self.mode = mode
        # if track and (frame_nb != -1):
        #     raise ValueError(f"Tracking should be performed on all the frames")

        self.box_mode = box_mode
        track_path = os.path.join(box_folder, f"boxes_ho3d_{split}.pkl")
        if not track and not os.path.exists(track_path):
            raise ValueError(
                f"Missing {track_path} file, generate it with "
                f"python track_dataset.py --dataset ho3d --split {split}")
        elif not track:
            with open(track_path, "rb") as p_f:
                self.tracked_boxes = pickle.load(p_f)

        self.load_img = load_img
        self.frame_nb = frame_nb
        self.image_size = (640, 480)
        self.track = track
        self.setup = {"right_hand": 1, "objects": 1}
        cache_folder = os.path.join("data", "cache")
        os.makedirs(cache_folder, exist_ok=True)
        cache_path = os.path.join(cache_folder, f"{self.name}_{split}.pkl")

        self.root = os.path.join(root, self.name)
        if not os.path.exists(self.root):
            raise RuntimeError(
                f"HO3D dataset not found at {self.root}, please follow instructions"
                "at https://github.com/hassony2/homan/tree/master#ho-3d")
        self.joint_nb = joint_nb
        self.reorder_idxs = np.array([
            0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8,
            9, 20
        ])

        # Fix dataset split
        valid_splits = ["train", "trainval", "val", "test"]
        assert split in valid_splits, "{} not in {}".format(
            split, valid_splits)
        self.split = split
        self.camextr = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0],
                                 [0, 0, 0, 1]])
        self.layer = manolayer.ManoLayer(
            joint_rot_mode="axisang",
            use_pca=False,
            mano_root=mano_root,
            center_idx=None,
            flat_hand_mean=True,
        )
        self.ref_idx = ref_idx
        self.cent_layer = manolayer.ManoLayer(
            joint_rot_mode="axisang",
            use_pca=False,
            mano_root=mano_root,
            center_idx=ref_idx,
            flat_hand_mean=True,
        )
        self.obj_meshes = ho3dfullutils.load_objects(ycb_root)
        self.can_obj_models = self.obj_meshes
        self.obj_paths = ho3dfullutils.load_object_paths(ycb_root)
        self.hand_faces = self.layer.th_faces
        if self.split == "train":
            seqs = ho3dconstants.TRAIN_SEQS
            subfolder = "train"
        elif self.split == "trainval":
            seqs = ho3dconstants.TRAINVAL_SEQS
            subfolder = "train"
        elif self.split == "val":
            seqs = ho3dconstants.VAL_SEQS
            subfolder = "train"
        elif self.split == "test":
            seqs = ho3dconstants.TEST_SEQS
            subfolder = "evaluation"
            warnings.warn(f"Using seqs {seqs} for evaluation")
            print(f"Using seqs {seqs} for evaluation")
        self.subfolder = subfolder

        if os.path.exists(cache_path) and use_cache:
            with open(cache_path, "rb") as p_f:
                dataset_annots = pickle.load(p_f)
            frame_index = dataset_annots["frame_index"]
            annotations = dataset_annots["annotations"]
        else:
            frame_index, annotations = ho3dutils.build_frame_index(
                seqs, self.root, subfolder, use_cache=use_cache)
            dataset_annots = {
                "frame_index": frame_index,
                "annotations": annotations,
            }
            with open(cache_path, "wb") as p_f:
                pickle.dump(dataset_annots, p_f)

        self.frame_index = frame_index
        self.annotations = annotations
        self.vid_index = self.frame_index.groupby(
            'seq_idx').first().reset_index()
        if split == "test":
            self.chunk_index = chunk_vid_index(self.vid_index,
                                               chunk_size=frame_nb,
                                               chunk_step=chunk_step,
                                               chunk_spacing=frame_nb *
                                               chunk_step)
        else:
            self.chunk_index = chunk_vid_index(self.vid_index,
                                               chunk_size=frame_nb,
                                               chunk_step=chunk_step,
                                               chunk_spacing=frame_nb *
                                               chunk_step)

        # Get paired links as neighboured joints
        self.links = [
            (0, 1, 2, 3, 4),
            (0, 5, 6, 7, 8),
            (0, 9, 10, 11, 12),
            (0, 13, 14, 15, 16),
            (0, 17, 18, 19, 20),
        ]

    def get_root(self, seq_idx, frame_idx):
        hand_root = self.annotations[(seq_idx, frame_idx)]['handJoints3D']
        hand_joints = hand_root[np.newaxis].repeat(21, 0)
        return hand_joints.dot(self.camextr[:3, :3])

    def get_frame_info(self, row, frame_idx, load_img=True):
        seq_idx = row.seq_idx

        # Get image information
        if load_img is False or (self.load_img is False):
            rgb = self.get_image_path(seq_idx, frame_idx)
        else:
            rgb = self.get_image(seq_idx, frame_idx)

        # Gather hand information
        hand_bbox = self.get_hand_bbox(seq_idx, frame_idx)
        obj_bbox = self.get_obj_bbox(seq_idx, frame_idx)
        hand_verts, hand_joints = self.get_hand_verts3d(seq_idx, frame_idx)

        # For test split, all hand joints are initialized to GT root locaton
        if self.split == "test":
            hand_root = self.annotations[(seq_idx, frame_idx)]['handJoints3D']
            hand_joints = hand_root[np.newaxis].repeat(21, 0)

        objpoints3d = self.get_obj_verts_trans(seq_idx, frame_idx)
        objpointscan = self.get_obj_verts_can(seq_idx, frame_idx)
        objfaces = self.get_obj_faces(seq_idx, frame_idx)
        objpath = self.get_obj_path(seq_idx, frame_idx)
        hand_info = dict(
            verts3d=hand_verts.dot(self.camextr[:3, :3]),
            faces=self.hand_faces.numpy(),
            joints3d=hand_joints.dot(self.camextr[:3, :3]),
            label="right_hand",
            bbox=hand_bbox,
        )
        obj_info = dict(verts3d=objpoints3d,
                        faces=objfaces,
                        path=objpath,
                        bbox=obj_bbox,
                        scale=1,
                        canverts3d=objpointscan)
        camintr = self.get_camintr(seq_idx, frame_idx)
        focal_nc = self.get_focal_nc(seq_idx, frame_idx)
        camera = dict(
            resolution=self.image_size,
            TWC=torch.eye(4).float(),
            K=camintr.astype(np.float32),
            focal_nc=focal_nc,
        )
        return rgb, camera, hand_info, obj_info

    def __getitem__(self, idx):
        if self.mode == "frame":
            row = self.frame_index.iloc[idx]
            frame_idx = row.frame_idx
            rgb, camera, hand_info, object_info = self.get_frame_info(
                row, frame_idx)
            obs = dict(img=rgb,
                       hands=[hand_info],
                       objects=[object_info],
                       camera=camera,
                       setup=self.setup)
            return obs
        elif self.mode == "vid":
            return self.get_vid_info(idx, mode="full_vid")
        elif self.mode == "chunk":
            return self.get_vid_info(idx, mode="chunk")

    def get_vid_info(self, idx, mode="full_vid"):
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
        seq_obj_info = []
        seq_hand_infos = []
        seq_cameras = []
        images = []
        for frame_idx in frame_idxs:
            rgb, camera, hand_info, object_info = self.get_frame_info(
                vid_info,
                frame_idx,
                load_img=not (self.track) and self.load_img)
            images.append(rgb)
            seq_obj_info.append(object_info)
            seq_hand_infos.append(hand_info)
            seq_cameras.append(camera)

        collated_hand_info = collate.collate(seq_hand_infos)
        collated_hand_info['label'] = collated_hand_info["label"][0]
        obs = dict(images=images,
                   hands=[collated_hand_info],
                   objects=[collate.collate(seq_obj_info)],
                   camera=collate.collate(seq_cameras),
                   setup=self.setup,
                   frame_idxs=frame_idxs,
                   seq_idx=vid_info.seq_idx)

        return obs

    def get_image_path(self, seq_idx, frame_idx):
        annot = self.annotations[(seq_idx, frame_idx)]
        img_path = annot["img"]
        return img_path

    def get_image(self, seq_idx, frame_idx, load_rgb=True):
        img_path = self.get_image_path(seq_idx, frame_idx)
        img = Image.open(img_path).convert("RGB")
        return img

    def get_joints2d(self, seq_idx, frame_idx):
        joints3d = self.get_joints3d(seq_idx, frame_idx)
        cam_intr = self.get_camintr(seq_idx, frame_idx)
        return self.project(joints3d, cam_intr)

    def get_joints3d(self, seq_idx, frame_idx):
        annot = self.annotations[(seq_idx, frame_idx)]
        joints3d = annot["handJoints3D"]
        joints3d = self.camextr[:3, :3].dot(joints3d.transpose()).transpose()
        if joints3d.ndim == 1:
            joints3d = joints3d[np.newaxis].repeat(21, 0)
        joints3d = joints3d[self.reorder_idxs]
        return joints3d.astype(np.float32)

    def get_obj_textures(self, seq_idx, frame_idx):
        annot = self.annotations[(seq_idx, frame_idx)]
        obj_id = annot["objName"]
        textures = self.obj_meshes[obj_id]["textures"]
        return textures

    def get_hand_ref(self, seq_idx, frame_idx):
        annot = self.annotations[(seq_idx, frame_idx)]
        # Retrieve hand info
        if "handPose" in annot:
            handpose = annot["handPose"]
            hand_trans = annot["handTrans"]
            hand_shape = annot["handBeta"]
            trans = hand_trans
        else:
            handpose = np.zeros(48)
            trans = annot["handJoints3D"]
            hand_shape = np.zeros(10)
        return handpose, trans, hand_shape

    def get_hand_verts3d(self, seq_idx, frame_idx):
        handpose, hand_trans, hand_shape = self.get_hand_ref(
            seq_idx, frame_idx)
        handpose_th = torch.Tensor(handpose).unsqueeze(0)
        hand_joint_rots = handpose_th[:, self.cent_layer.rot:]
        hand_root_rot = handpose_th[:, :self.cent_layer.rot]
        hand_pca = manoutils.pca_from_aa(hand_joint_rots, rem_mean=True)
        handverts, handjoints, center_c = self.cent_layer(
            handpose_th,
            torch.Tensor(hand_shape).unsqueeze(0))
        hand_trans = hand_trans
        if center_c is not None:
            hand_trans = hand_trans + center_c.numpy()[0] / 1000
        handverts = handverts[0].numpy() / 1000 + hand_trans
        handjoints = handjoints[0].numpy() / 1000 + hand_trans
        return handverts, handjoints

    def get_hand_verts2d(self, seq_idx, frame_idx):
        verts3d, _ = self.get_hand_verts3d(seq_idx, frame_idx)
        cam_intr = self.get_camintr(seq_idx, frame_idx)
        verts2d = self.project(verts3d, cam_intr, self.camextr)
        return verts2d

    def get_obj_path(self, seq_idx, frame_idx):
        annot = self.annotations[(seq_idx, frame_idx)]
        obj_id = annot["objName"]
        obj_path = self.obj_paths[obj_id]
        return obj_path

    def get_obj_faces(self, seq_idx, frame_idx):
        annot = self.annotations[(seq_idx, frame_idx)]
        obj_id = annot["objName"]
        objfaces = self.obj_meshes[obj_id]["faces"]
        objfaces = np.array(objfaces).astype(np.int16)
        return objfaces

    def get_obj_rot(self, seq_idx, frame_idx):
        annot = self.annotations[(seq_idx, frame_idx)]
        rot = cv2.Rodrigues(annot["objRot"])[0]
        return rot

    def get_obj_verts_can(self, seq_idx, frame_idx):
        annot = self.annotations[(seq_idx, frame_idx)]
        obj_id = annot["objName"]
        verts = self.obj_meshes[obj_id]["verts"]
        return (verts - verts.mean(0)).astype(np.float32)

    def get_obj_verts2d(self, seq_idx, frame_idx):
        verts3d = self.get_obj_verts_trans(seq_idx, frame_idx)
        cam_intr = self.get_camintr(seq_idx, frame_idx)
        verts2d = self.project(verts3d, cam_intr)
        return verts2d

    def get_obj_verts_trans(self, seq_idx, frame_idx):
        rot = self.get_obj_rot(seq_idx, frame_idx)
        annot = self.annotations[(seq_idx, frame_idx)]
        trans = annot["objTrans"]
        obj_id = annot["objName"]
        verts = self.obj_meshes[obj_id]["verts"]
        trans_verts = rot.dot(verts.transpose()).transpose() + trans
        trans_verts = self.camextr[:3, :3].dot(
            trans_verts.transpose()).transpose()
        obj_verts = np.array(trans_verts).astype(np.float32)
        return obj_verts

    def get_obj_pose(self, seq_idx, frame_idx):
        annot = self.annotations[(seq_idx, frame_idx)]
        rot = cv2.Rodrigues(annot["objRot"])[0]
        trans = annot["objTrans"]
        rot = self.camextr[:3, :3].dot(rot.dot(self.camextr[:3, :3]))
        trans = trans * np.array([1, -1, -1])
        return rot, trans

    def get_obj_corners3d(self, seq_idx, frame_idx):
        annot = self.annotations[(seq_idx, frame_idx)]
        rot = cv2.Rodrigues(annot["objRot"])[0]
        trans = annot["objTrans"]
        corners = annot["objCorners3DRest"]
        trans_corners = rot.dot(corners.transpose()).transpose() + trans
        trans_corners = self.camextr[:3, :3].dot(
            trans_corners.transpose()).transpose()
        obj_corners = np.array(trans_corners).astype(np.float32)
        return obj_corners

    def get_objcorners2d(self, seq_idx, frame_idx):
        corners3d = self.get_obj_corners3d(seq_idx, frame_idx)
        cam_intr = self.get_camintr(seq_idx, frame_idx)
        return self.project(corners3d, cam_intr)

    def get_objverts2d(self, seq_idx, frame_idx):
        objpoints3d = self.get_obj_verts_trans(seq_idx, frame_idx)
        cam_intr = self.get_camintr(seq_idx, frame_idx)
        verts2d = self.project(objpoints3d, cam_intr)
        return verts2d

    def get_sides(self, idx):
        return "right"

    def get_camintr(self, seq_idx, frame_idx):
        annot = self.annotations[(seq_idx, frame_idx)]
        cam_intr = annot["camMat"]
        return cam_intr

    def get_focal_nc(self, seq_idx, frame_idx):
        annot = self.annotations[(seq_idx, frame_idx)]
        cam_intr = annot["camMat"]
        return (cam_intr[0, 0] + cam_intr[1, 1]) / 2 / max(self.image_size)

    def __len__(self):
        if self.mode == "frame":
            return len(self.frame_index)
        elif self.mode == "vid":
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

    def get_hand_bbox(self, seq_idx, frame_idx):
        if self.box_mode in ["track"]:
            bbox = np.array(
                self.tracked_boxes[seq_idx]['right_hand'][frame_idx])
            # bbox = np.array(self.tracked_boxes[seq_idx]['hands'][0][frame_idx])
        elif self.box_mode == "gt":
            annot = self.annotations[(seq_idx, frame_idx)]
            if "handBoundingBox" in annot:
                bbox = np.array(annot["handBoundingBox"])
                warnings.warn(f"Problem with gt bboxes ?")
            else:
                verts2d = self.get_hand_verts2d(seq_idx, frame_idx)
                bbox = np.concatenate([verts2d.min(0), verts2d.max(0)], 0)
        else:
            raise ValueError(
                f"Invalid box mode {self.box_mode}, not in ['track'|'gt']")
        return bbox

    def get_obj_bbox(self, seq_idx, frame_idx):
        if self.box_mode in ["track"]:
            bbox = np.array(self.tracked_boxes[seq_idx]['objects'][frame_idx])
            # bbox = np.array(
            #     self.tracked_boxes[seq_idx]['objects'][0][frame_idx])
        elif self.box_mode == "gt":
            verts2d = self.get_obj_verts2d(seq_idx, frame_idx)
            bbox = np.concatenate([verts2d.min(0), verts2d.max(0)], 0)
        else:
            raise ValueError(
                f"Invalid box mode {self.box_mode}, not in ['track'|'gt']")
        return bbox
