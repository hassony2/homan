import numpy as np
import torch

from homan.manomodel import ManoModel
from homan.utils.camera import compute_transformation_persp
from homan.utils.geometry import combine_verts, matrix_to_rot6d, rot6d_to_matrix

from libyana.conversions import npt
from manopth import manolayer
from manopth import manolayer, rodrigues_layer

# pylint: disable=missing-module-docstring,broad-except,too-many-statements,too-many-branches,logging-fstring-interpolation
# pylint: disable=no-member,wrong-import-order


def post_process(sample_info,
                 state_dict,
                 mano_root="extra_data/mano",
                 frame_nb=10,
                 add_mean=False):
    mano_trans = state_dict["mano_trans"]  # batch_sizex3
    # person_trans = state_dict["translations_hand"]  # batch_sizex3
    # mano_betas = state_dict["mano_betas"]  # batch_sizex10 full of 0
    mano_rots = state_dict["mano_rot"]  # batch_sizex3
    hand_sides = [hand["label"].split("_")[0] for hand in sample_info['hands']]
    hand_scales = state_dict["int_scales_hand"]
    obj_scales = state_dict["int_scales_object"]
    all_hand_verts = []
    hand_nb = len(hand_sides)
    if len(hand_sides) == 2:
        hand_sides = ["left", "right"]
    mano_model = ManoModel("extra_data/mano", pca_comps=16)
    for hand_idx, hand_side in enumerate(hand_sides):
        mano_pca_pose = state_dict["mano_pca_pose"][hand_idx::hand_nb]
        mano_rot = state_dict["mano_rot"][hand_idx::hand_nb]
        mano_posewrot = torch.cat([mano_rot, mano_pca_pose[:, :15]], 1)
        mano_res = mano_model.forward_pca(
            mano_pca_pose,
            rot=mano_rot,
            betas=state_dict["mano_betas"][hand_idx::hand_nb],
            side=hand_side)
        vertices = mano_res["verts"]
        verts_hand_og = vertices + state_dict["mano_trans"][
            hand_idx::hand_nb].unsqueeze(1)
        rotations_hand = rot6d_to_matrix(
            state_dict["rotations_hand"][hand_idx::hand_nb])
        translations_hand = state_dict["translations_hand"][hand_idx::hand_nb]
        hand_vertss, _ = compute_transformation_persp(
            meshes=verts_hand_og,
            translations=translations_hand,
            rotations=rotations_hand,
            intrinsic_scales=hand_scales,
        )
        all_hand_verts.append(hand_vertss)

        joints = mano_res["joints"]
        # import pudb
        # pu.db
        hand_jointss, _ = compute_transformation_persp(
            meshes=joints +
            state_dict["mano_trans"][hand_idx::hand_nb].unsqueeze(1),
            translations=translations_hand,
            rotations=rotations_hand,
            intrinsic_scales=hand_scales,
        )
        hand_jointss = torch.cat(
            [hand_jointss, hand_vertss[:, [745, 317, 444, 556, 673]]], 1)[:, [
                0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7,
                8, 9, 20
            ]]

    # Recover transformed object model
    obj_trans = state_dict["translations_object"]  # batch_sizex3
    obj_rots = state_dict[
        "rotations_object"]  # batch_sizex3x2 continuous rotation representation
    obj_rot_mats = rot6d_to_matrix(obj_rots)  # batch_sizex3x3 hand rotation
    obj_verts = state_dict["verts_object_og"]  #batch_sizexvertex_nbx3
    img_path = sample_info['images']
    # from libyana.visutils import imagify
    # from matplotlib import pyplot as plt
    # import cv2
    # obj_faces = npt.numpify(state_dict["faces_object"])[0]
    # obj_verts_trans = torch.matmul(obj_verts, obj_rot_mats) + obj_trans
    # from libyana.camutils import project
    # from libyana.visutils import viz2d
    # fig, axes = plt.subplots(2)
    # ax = axes[0]
    # v2d = project.proj2d(hand_vertss[0], sample_info["camera"]["K"][0])[0]
    # j2d = project.proj2d(hand_jointss[0], sample_info["camera"]["K"][0])[0]
    # # img = cv2.imread(img_path[0])
    # ax.imshow(img_path[0])
    # ax.scatter(v2d[:, 0], v2d[:, 1], s=1)
    # ax.scatter(j2d[:, 0], j2d[:, 1], s=1)
    # viz2d.visualize_joints_2d(ax, j2d)
    # ax = axes[1]
    # viz2d.visualize_joints_2d(ax, j2d)
    # fig.savefig("tmp.png")

    hand_vertss = npt.numpify(hand_vertss)
    hand_jointss = npt.numpify(hand_jointss)
    camintrs = npt.numpify(sample_info["camera"]["K"])
    train_infos = []
    if "scale" in sample_info["objects"][0]:
        scale = sample_info["objects"][0]["scale"]
        if isinstance(scale, (list, tuple, torch.tensor, np.ndarray)):
            scale = scale[0]
    else:
        scale = 1
    for frame_idx in range(frame_nb):
        obj_path = sample_info['objects'][0]['path']
        train_info = {
            "all_hand_verts3d":
            [npt.numpify(hand_v[frame_idx]) for hand_v in all_hand_verts],
            "hand_verts3d":
            npt.numpify(hand_vertss[frame_idx]),
            "hand_joints3d":
            npt.numpify(hand_jointss[frame_idx]),
            "camintr":
            camintrs[frame_idx],
            "img_path":
            img_path[frame_idx],
            "side":
            hand_side,
            "obj_path":
            obj_path[frame_idx],
            "obj_rot":
            npt.numpify(obj_rot_mats[frame_idx]),
            "obj_trans":
            npt.numpify(obj_trans[frame_idx]),
            "obj_scale":
            scale * obj_scales.item(),
            "hand_sides":
            hand_sides,
        }
        train_infos.append(train_info)
    return train_infos, sample_info["seq_idx"], sample_info["frame_idxs"]
