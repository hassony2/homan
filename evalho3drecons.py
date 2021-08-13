#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable=wrong-import-order,no-member,missing-module-docstring,import-error
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from homan import getdataset, postprocess
from homan.eval import analyze, ho3devalutils, pointmetrics
from homan.viz import renderot

import argparse
import json
import os
import pandas as pd
import trimesh
from collections import OrderedDict, defaultdict
from libyana.camutils import project
from libyana.visutils import viz2d

parser = argparse.ArgumentParser()
parser.add_argument("--root")
parser.add_argument("--frame_nb", default=10, type=int)
parser.add_argument(
    "--dataset",
    default="ho3d",
    choices=["contactpose", "ho3d", "fhb", "epic", "core50", "inhandycb"],
    help="Dataset name")
parser.add_argument("--split",
                    default="test",
                    choices=["train", "val", "trainval", "test"],
                    help="Dataset name")
parser.add_argument("--box_mode", choices=["gt", "track"], default="gt")
parser.add_argument("--display_freq", default=1000, type=int)
parser.add_argument("--chunk_step", default=1, type=int)
args = parser.parse_args()

os.makedirs(args.root, exist_ok=True)

samples_root = os.path.join(args.root, "samples")
vid_folder = os.path.join(args.root, "test_vids")
os.makedirs(vid_folder, exist_ok=True)
all_samples = os.listdir(samples_root)

missing = []
for sample in all_samples:
    sample_path = os.path.join(samples_root, sample)
    if not os.path.exists(os.path.join(sample_path, "joint_fit.pt")):
        missing.append(sample)

print(f"Missing {len(missing)} samples {missing} at location {samples_root}")
dataset, _ = getdataset.get_dataset(args.dataset,
                                    split=args.split,
                                    frame_nb=args.frame_nb,
                                    box_mode=args.box_mode,
                                    load_img=False,
                                    chunk_step=args.chunk_step)

mano_faces = np.array(
    trimesh.load("extra_data/mano/closed_phosa.obj", process=False).faces)
mano_faces_pt = torch.Tensor(mano_faces).cuda().unsqueeze(0)

sequences = [
    "SM1", "MPM10", "MPM11", "MPM12", "MPM13", "MPM14", "SB11", "SB13", "AP10",
    "AP11", "AP12", "AP13", "AP14"
]
vid_index = dataset.vid_index
seq_lens = {}
for seq in sequences:
    seq_frame_nb = (vid_index[vid_index["seq_idx"] == seq].frame_nb).values[0]
    seq_lens[seq] = seq_frame_nb

seq_res = defaultdict(OrderedDict)
# Collect all fitted results
for sample_str in tqdm(sorted(all_samples)):
    sample_root = os.path.join(samples_root, sample_str)
    res_path = os.path.join(sample_root, "joint_fit.pt")
    if os.path.exists(res_path):
        sample_info = dataset[int(sample_str)]
        state_dict = torch.load(res_path)["state_dict"]
    samples_res, seq_idx, frame_idxs = postprocess.post_process(
        sample_info, state_dict, frame_nb=args.frame_nb, add_mean=False)
    for sample_idx, (frame_idx,
                     sample_res) in enumerate(zip(frame_idxs, samples_res)):
        objvertscan = dataset.obj_meshes[sample_res["obj_path"].split('/')
                                         [-2]]['verts']
        objvertscan = objvertscan - objvertscan.mean(0)
        pred_objverts = objvertscan.dot(sample_res["obj_rot"]) * sample_res[
            "obj_scale"] + sample_res["obj_trans"]
        sample_res["obj_verts3d"] = pred_objverts

        hand_roots = sample_info["hands"][0]["joints3d"][:, :1]
        sample_res["hand_roots"] = hand_roots[sample_idx]
        seq_res[seq_idx][frame_idx] = sample_res

all_verts, all_gt_objverts, all_joints, all_img_paths = [], [], [], []
all_hand_roots, all_pred_objverts = [], []
camextr = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
camextrfull = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0],
                        [0, 0, 0, 1]])
# Ours -> HO3D
unorder_idxs = [
    0, 5, 6, 7, 10, 11, 12, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 8, 12, 16, 20
]
full_html_data = []
loss_errors = defaultdict(list)
full_idx = 0
for seq in tqdm(sequences, desc="seq"):
    print(seq)
    seq_frame_nb = seq_lens[seq]
    interp_res = ho3devalutils.interpolate_res(seq_res[seq], seq_frame_nb)
    obj_faces = dataset.get_obj_faces(seq, 0)
    obj_faces_pt = torch.Tensor(obj_faces).cuda().unsqueeze(0)
    seq_html_data = {}
    seq_errors = defaultdict(list)
    seq_frame_idx = 0
    for frame_idx in tqdm(range(seq_frame_nb), desc="frame"):
        # Get ground truth object vertex locations
        gt_objverts = dataset.get_obj_verts_trans(seq, frame_idx).dot(camextr)
        all_gt_objverts.append(gt_objverts)

        # Get predicted object vertices
        pred_objverts = interp_res["obj_verts3d"][frame_idx].dot(
            camextr).astype(np.float32)
        all_pred_objverts.append(pred_objverts)

        # Compute object metric_verts2d
        obj_metrics = pointmetrics.get_point_metrics(
            torch.Tensor(pred_objverts).unsqueeze(0).float(),
            torch.Tensor(gt_objverts).unsqueeze(0).float())
        obj_dist = obj_metrics["verts_dists"][0]
        obj_add = obj_metrics["add-s"][0]
        loss_errors["obj_dist"].append(obj_dist)
        loss_errors["obj_add-s"].append(obj_add)
        seq_errors["obj_dist"].append(obj_dist)
        seq_errors["obj_add-s"].append(obj_add)
        # AP objects start at index 7694
        if full_idx >= 7694:
            loss_errors["obj_dist_unseen"].append(obj_dist)
            loss_errors["add-s_unseen"].append(obj_add)
        else:
            loss_errors["obj_dist_seen"].append(obj_dist)
            loss_errors["add-s_seen"].append(obj_add)
        full_idx += 1

        # Get hand root info
        hand_root = dataset.get_root(seq, frame_idx).dot(camextr)
        all_hand_roots.append(hand_root)

        # Get hand predicted info
        pred_handverts = interp_res["hand_verts3d"][frame_idx].dot(
            camextr).astype(np.float32)
        all_verts.append(pred_handverts)
        pred_handjoints = interp_res["hand_joints3d"][frame_idx].dot(
            camextr)[unorder_idxs].astype(np.float32)
        all_joints.append(pred_handjoints)
        hand_root_err = np.linalg.norm(pred_handjoints[0] - hand_root[0])
        loss_errors["hand_root"].append(hand_root_err)
        seq_errors["hand_root"].append(hand_root_err)
        if frame_idx % args.display_freq == 0:
            video_path = os.path.join(vid_folder,
                                      f"rot_{seq}_{seq_frame_idx:06d}.mp4")
            with torch.no_grad():
                renderot.rot_render(pred_handverts.dot(camextr),
                                    mano_faces,
                                    pred_objverts.dot(camextr),
                                    obj_faces,
                                    video_path=video_path)
            print(f"Saved video to {video_path}")
            seq_html_data[f"rot_{seq_frame_idx:05d}_video_path"] = video_path
            seq_frame_idx += 1

        pred_handverts_pt = torch.Tensor(pred_handverts).cuda().unsqueeze(0)
        pred_objverts_pt = torch.tensor(pred_objverts).cuda().unsqueeze(0)
        inter_metrics = pointmetrics.get_inter_metrics(pred_handverts_pt,
                                                       pred_objverts_pt,
                                                       mano_faces_pt,
                                                       obj_faces_pt)
        contacts = [float(val) for val in inter_metrics["has_contact"]]

        loss_errors["has_contact"].extend(contacts)
        seq_errors["has_contact"].extend(contacts)

        loss_errors["pen_depths"].extend(inter_metrics["pen_depths"])
        seq_errors["pen_depths"].extend(inter_metrics["pen_depths"])
        img_path = interp_res["img_paths"][frame_idx]
        all_img_paths.append(img_path)
    # Render imgs in middle of sequence
    vis_extent = 30
    seq_frame_idxs = list(
        range(max(0, seq_frame_nb // 2 - vis_extent),
              min(seq_frame_nb - 1, seq_frame_nb // 2 + vis_extent)))
    video_path = os.path.join(vid_folder, f"seq_{seq}_{frame_idx:06d}.mp4")
    seq_hand_verts = [
        interp_res["hand_verts3d"][idx] for idx in seq_frame_idxs
    ]
    seq_obj_verts = [interp_res["obj_verts3d"][idx] for idx in seq_frame_idxs]
    seq_img_paths = [interp_res["img_paths"][idx] for idx in seq_frame_idxs]
    # Load and rescale images
    seq_imgs = [cv2.imread(img_path)[:, :, ::-1] for img_path in seq_img_paths]
    vis_scale_factor = 2
    seq_imgs = [
        cv2.resize(seq_img, (seq_img.shape[1] // vis_scale_factor,
                             seq_img.shape[0] // vis_scale_factor))
        for seq_img in seq_imgs
    ]
    seq_mano_faces = [mano_faces_pt[0] for _ in seq_frame_idxs]
    seq_obj_faces = [obj_faces_pt[0] for _ in seq_frame_idxs]

    camintr = seq_res[seq][0]['camintr'] / 2
    with torch.no_grad():
        renderot.seq_render(seq_hand_verts,
                            seq_mano_faces,
                            seq_obj_verts,
                            seq_obj_faces,
                            camintr=camintr,
                            imgs=np.stack(seq_imgs),
                            video_path=video_path)
    seq_html_data["clip_video_path"] = video_path
    for key, vals in seq_errors.items():
        seq_html_data[key] = np.mean(vals)
    full_html_data.append(seq_html_data)

loss_errors_avg = {key: np.mean(vals) for key, vals in loss_errors.items()}
print("Mean errors")
print(loss_errors_avg)
loss_errors_median = {
    key: np.median(vals)
    for key, vals in loss_errors.items()
}
print("Median errors")
print(loss_errors_median)
loss_errors_max = {key: np.max(vals) for key, vals in loss_errors.items()}
print("Max errors")
print(loss_errors_max)

full_html_data.insert(0, loss_errors_avg)
df_data = pd.DataFrame(full_html_data)
analyze.make_exp_html(df_data, {},
                      None,
                      os.path.join(args.root, "html"),
                      compact=False,
                      sort_loss=None,
                      drop_redundant=False)
ref_pred_path = "../handobjectconsist/pred.json"
with open(ref_pred_path, "r") as t_f:
    ref_val = json.load(t_f)[0]

fig, axes = plt.subplots(2, 4, figsize=(4 * 3, 2 * 3))
# HO3D -> ours
reorder_idxs = [
    0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20
]
# Show first
for ax_idx, show_idx in enumerate([1, -1]):
    img_path = all_img_paths[show_idx]
    camintr = samples_res[show_idx]["camintr"]
    j3d = all_joints[show_idx][reorder_idxs]
    v3d = all_verts[show_idx]
    root3d = all_hand_roots[show_idx]
    gt_ov3d = all_gt_objverts[show_idx]
    pred_ov3d = all_pred_objverts[show_idx]
    ref3d = np.array(ref_val[show_idx])[reorder_idxs]
    ax = axes[ax_idx, 3]
    img = cv2.imread(img_path)
    ax.imshow(img)
    j2d = project.proj2d(j3d, camintr, camextr=camextrfull)[0]
    # j2d = project.proj2d(j3d.dot(camextr), camintr)[0]
    r2d = project.proj2d(ref3d, camintr, camextr=camextrfull)[0]
    v2d = project.proj2d(v3d, camintr, camextr=camextrfull)[0]
    ref_ov2d = project.proj2d(gt_ov3d, camintr, camextr=camextrfull)[0]
    root2d = project.proj2d(root3d, camintr, camextr=camextrfull)[0]
    ax.scatter(ref_ov2d[:, 0], ref_ov2d[:, 1], s=1, c="pink", alpha=0.1)
    ax.scatter(root2d[:, 0], root2d[:, 1], c="k")
    ax.scatter(v2d[:, 0], v2d[:, 1], s=1, c="b", alpha=0.1)
    ax.scatter(j2d[:, 0], j2d[:, 1], s=1, c="r")
    ax.scatter(r2d[:, 0], r2d[:, 1], s=1, c="g")
    viz2d.visualize_joints_2d(ax, j2d, joint_idxs=False)
    viz2d.visualize_joints_2d(ax, r2d, joint_idxs=False)

    ax = axes[ax_idx, 0]
    ax.scatter(gt_ov3d[:, 0], gt_ov3d[:, 1], c="pink", s=1)
    ax.scatter(pred_ov3d[:, 0], pred_ov3d[:, 1], c="cyan", s=1)
    ax.scatter(v3d[:, 0], v3d[:, 1], c="purple", s=1)
    ax.scatter(root3d[:, 0], root3d[:, 1], c="k")
    viz2d.visualize_joints_2d(ax, j3d)
    viz2d.visualize_joints_2d(ax, ref3d, alpha=0.5)

    ax = axes[ax_idx, 1]
    ax.scatter(j3d[:, 1], j3d[:, 2])
    ax.scatter(gt_ov3d[:, 1], gt_ov3d[:, 2], c="pink", s=1)
    ax.scatter(pred_ov3d[:, 1], pred_ov3d[:, 2], c="cyan", s=1)
    ax.scatter(v3d[:, 1], v3d[:, 2], c="purple", s=1)
    ax.scatter(root3d[:, 1], root3d[:, 2], c="k")
    viz2d.visualize_joints_2d(ax, j3d[:, 1:])
    viz2d.visualize_joints_2d(ax, ref3d[:, 1:], alpha=0.5)

    ax = axes[ax_idx, 2]
    ax.scatter(gt_ov3d[:, 0], gt_ov3d[:, 2], c="pink", s=1)
    ax.scatter(pred_ov3d[:, 0], pred_ov3d[:, 2], c="cyan", s=1)
    ax.scatter(v3d[:, 0], v3d[:, 2], c="pink", s=1)
    ax.scatter(root3d[:, 0], root3d[:, 2], c="k")
    viz2d.visualize_joints_2d(ax, j3d[:, ::2])
    viz2d.visualize_joints_2d(ax, ref3d[:, ::2], alpha=0.5)
fig.savefig("hand.png")
print(len(all_joints), len(all_verts))
save_path = os.path.join(args.root, "pred.json")
ho3devalutils.dump(save_path, all_joints, all_verts)
print(f"Saved to {save_path}")
