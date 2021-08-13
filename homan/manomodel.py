#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0411,broad-except,too-many-statements,too-many-branches,logging-fstring-interpolation,import-error
# pylint: disable=too-many-arguments,missing-function-docstring,too-many-locals
import os

import torch
from torch import nn
from mano.model import load as mano_load

from libyana.verify import checkshape
from libyana.conversions import npt


class ManoModel(nn.Module):
    def __init__(self, mano_root, pca_comps=30, batch_size=1):
        super().__init__()
        self.pca_comps = pca_comps
        self.rh_mano_flat = mano_load(model_path=os.path.join(
            mano_root, "MANO_RIGHT.pkl"),
                                      num_pca_comps=pca_comps,
                                      use_pca=False,
                                      is_right=True,
                                      model_type="mano",
                                      batch_size=batch_size,
                                      flat_hand_mean=True)
        self.rh_mano = mano_load(model_path=os.path.join(
            mano_root, "MANO_RIGHT.pkl"),
                                 model_type="mano",
                                 num_pca_comps=pca_comps,
                                 use_pca=False,
                                 is_right=True,
                                 batch_size=batch_size,
                                 flat_hand_mean=False)
        self.lh_mano_flat = mano_load(model_path=os.path.join(
            mano_root, "MANO_LEFT.pkl"),
                                      num_pca_comps=pca_comps,
                                      model_type="mano",
                                      is_right=True,
                                      use_pca=False,
                                      batch_size=batch_size,
                                      flat_hand_mean=True)
        self.lh_mano = mano_load(model_path=os.path.join(
            mano_root, "MANO_LEFT.pkl"),
                                 num_pca_comps=pca_comps,
                                 model_type="mano",
                                 is_right=True,
                                 use_pca=False,
                                 batch_size=batch_size,
                                 flat_hand_mean=False)
        self.rh_mean = self.rh_mano.hand_mean
        self.lh_mean = self.lh_mano.hand_mean
        self.rh_mano_flat_pca = mano_load(model_path=os.path.join(
            mano_root, "MANO_RIGHT.pkl"),
                                          num_pca_comps=pca_comps,
                                          use_pca=True,
                                          model_type="mano",
                                          batch_size=batch_size,
                                          flat_hand_mean=True)
        self.rh_mano_pca = mano_load(model_path=os.path.join(
            mano_root, "MANO_RIGHT.pkl"),
                                     model_type="mano",
                                     num_pca_comps=pca_comps,
                                     use_pca=True,
                                     batch_size=batch_size,
                                     flat_hand_mean=False)
        self.lh_mano_flat_pca = mano_load(model_path=os.path.join(
            mano_root, "MANO_LEFT.pkl"),
                                          num_pca_comps=pca_comps,
                                          model_type="mano",
                                          use_pca=True,
                                          batch_size=batch_size,
                                          flat_hand_mean=True)
        self.lh_mano_pca = mano_load(model_path=os.path.join(
            mano_root, "MANO_LEFT.pkl"),
                                     num_pca_comps=pca_comps,
                                     model_type="mano",
                                     use_pca=True,
                                     batch_size=batch_size,
                                     flat_hand_mean=False)
        self.rh_mean = self.rh_mano.hand_mean
        self.lh_mean = self.lh_mano.hand_mean

    def forward_pca(self,
                    pca_pose=None,
                    rot=None,
                    betas=None,
                    side="right",
                    flat_hand_mean=False):
        if not isinstance(pca_pose, torch.Tensor):
            pca_pose = npt.tensorify(pca_pose)
        if rot is not None:
            rot = npt.tensorify(rot, device=pca_pose.device)
        if pca_pose.dim() == 1:
            pca_pose = pca_pose.unsqueeze(0)
            flatten = True
        else:
            flatten = False

        if (rot is not None) and (rot.dim() == 1):
            rot = rot.unsqueeze(0)
        if betas is not None:
            betas = npt.tensorify(betas, device=pca_pose.device)
            if betas.dim() == 1:
                betas = betas.unsqueeze(0)

        # Zero translation of correct batch size
        hand_trans = rot.new_zeros(rot.shape[0], 3)
        if side == "right":
            hand_pose = torch.einsum('bi,bij->bj', [
                pca_pose[:, :self.pca_comps],
                self.rh_mano_pca.hand_components.unsqueeze(0).repeat(
                    pca_pose.shape[0], 1, 1)
            ])
            if not flat_hand_mean:
                hand_mean = self.rh_mean.to(
                    hand_pose.device).unsqueeze(0).repeat(len(hand_pose), 1)
                hand_pose = hand_pose + hand_mean
            vertices, joints, _, _, _, hand_aa_pose = self.rh_mano_flat(
                betas=betas,
                global_orient=rot,
                hand_pose=hand_pose,
                transl=hand_trans)
        elif side == "left":
            hand_pose = torch.einsum('bi,bij->bj', [
                pca_pose[:, :self.pca_comps],
                self.lh_mano_pca.hand_components.unsqueeze(0).repeat(
                    pca_pose.shape[0], 1, 1)
            ])
            hand_pose[:, 1::3] *= -1
            hand_pose[:, 2::3] *= -1
            if not flat_hand_mean:
                hand_mean = self.lh_mean.to(
                    hand_pose.device).unsqueeze(0).repeat(len(hand_pose), 1)
                hand_pose = hand_pose + hand_mean
            vertices, joints, _, _, global_rot, hand_aa_pose = self.lh_mano_flat(
                betas=betas,
                global_orient=rot,
                hand_pose=hand_pose,
                transl=hand_trans)
        else:
            raise ValueError(f"{side} not in [left|right]")
        if flatten:
            vertices = vertices[0]
            joints = joints[0]
        mano_res = {
            "verts": vertices,
            "joints": joints,
            "hand_aa_pose": hand_aa_pose
        }
        return mano_res

    def forward(self,
                mano_pose=None,
                rot=None,
                betas=None,
                side="right",
                flat_hand_mean=False):
        """
        Args:
            mano_pose (torch.Tensor): B x N
            rot (torch.Tensor): B x 3, axisangle root rot
            betas (torch.Tensor): B x 10
        """
        if not isinstance(mano_pose, torch.Tensor):
            mano_pose = npt.tensorify(mano_pose)
        if rot is not None:
            rot = npt.tensorify(rot, device=mano_pose.device)
        if mano_pose.dim() == 1:
            mano_pose = mano_pose.unsqueeze(0)
            flatten = True
        else:
            flatten = False

        if (rot is not None) and (rot.dim() == 1):
            rot = rot.unsqueeze(0)
        if betas is not None:
            betas = npt.tensorify(betas, device=mano_pose.device)
            if betas.dim() == 1:
                betas = betas.unsqueeze(0)

        # Zero translation of correct batch size
        hand_trans = rot.new_zeros(rot.shape[0], 3)
        if side == "right":
            if not flat_hand_mean:
                hand_mean = self.rh_mean.to(
                    mano_pose.device).unsqueeze(0).repeat(len(mano_pose), 1)
                mano_pose = mano_pose + hand_mean
            vertices, joints, _, _, _, hand_aa_pose = self.rh_mano_flat(
                betas=betas,
                global_orient=rot,
                hand_pose=mano_pose,
                transl=hand_trans)
        elif side == "left":
            if not flat_hand_mean:
                hand_mean = self.lh_mean.to(
                    mano_pose.device).unsqueeze(0).repeat(len(rot), 1)
                mano_pose = mano_pose + hand_mean
            vertices, joints, _, _, _, hand_aa_pose = self.lh_mano_flat(
                betas=betas,
                global_orient=rot,
                hand_pose=mano_pose,
                transl=hand_trans)
        else:
            raise ValueError(f"{side} not in [left|right]")
        if flatten:
            vertices = vertices[0]
            joints = joints[0]
        mano_res = {
            "verts": vertices,
            "joints": joints,
            "hand_aa_pose": hand_aa_pose
        }
        return mano_res

    def get_mano_trans(self,
                       mano_pose,
                       rot,
                       ref_verts,
                       betas=None,
                       side="right"):
        checkshape.check_shape(mano_pose, (45, ), "mano_pose")
        checkshape.check_shape(rot, (3, ), "rot")
        checkshape.check_shape(ref_verts, (778, 3), "ref_verts")
        mano_pose = torch.Tensor(mano_pose)
        rot = torch.Tensor(rot)
        if betas is not None:
            betas = torch.Tensor(betas)
        mano_res = self.forward(mano_pose=mano_pose,
                                rot=rot,
                                betas=betas,
                                side=side)
        verts = mano_res["verts"]
        ref_verts = npt.tensorify(ref_verts, verts.device)
        return (ref_verts.mean(0) - verts.mean(0))[None, :]
