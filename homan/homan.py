# -*- coding: utf-8 -*-
# pylint: disable=import-error,no-member,wrong-import-order,too-many-branches,too-many-locals,too-many-statements
# pylint: disable=missing-function-docstring
import neural_renderer as nr
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from homan import lossutils
from homan.losses import Losses
from homan.lossutils import compute_ordinal_depth_loss
from homan.manomodel import ManoModel
from homan.meshutils import get_faces_and_textures
from homan.utils.camera import (
    compute_transformation_ortho,
    compute_transformation_persp,
)
from homan.utils.geometry import combine_verts, matrix_to_rot6d, rot6d_to_matrix

from libyana.conversions import npt
from libyana.lib3d import trans3d
from libyana.verify.checkshape import check_shape


class HOMan(nn.Module):
    def __init__(
        self,
        translations_object,
        rotations_object,
        verts_object_og,
        faces_object,
        translations_hand,
        rotations_hand,
        verts_hand_og,
        ref_verts2d_hand,
        hand_sides,
        mano_trans,
        mano_rot,
        mano_betas,
        mano_pca_pose,
        faces_hand,
        masks_object,
        masks_hand,
        camintr_rois_object,
        camintr_rois_hand,
        target_masks_object,
        target_masks_hand,
        class_name,
        cams_hand=None,
        int_scale_init=1.0,
        camintr=None,
        optimize_object_scale=False,
        optimize_ortho_cam=True,
        hand_proj_mode="persp",
        optimize_mano=True,
        optimize_mano_beta=True,
        inter_type="centroid",
        image_size=640,
    ):
        """
        Hands are received in batch of [h_1_t_1, h_2_t_1, ..., h_1_t_2]
        (h_{hand_index}_t_{time_step})
        """
        super().__init__()
        # Initialize object pamaters
        translation_init = translations_object.detach().clone()
        self.translations_object = nn.Parameter(translation_init,
                                                requires_grad=True)
        self.mano_model = ManoModel("extra_data/mano", pca_comps=16)
        self.hand_proj_mode = hand_proj_mode

        rotations_object = rotations_object.detach().clone()
        if rotations_object.shape[-1] == 3:
            rotations_object6d = matrix_to_rot6d(rotations_object)
        else:
            rotations_object6d = rotations_object
        self.rotations_object = nn.Parameter(
            rotations_object6d.detach().clone(), requires_grad=True)
        self.register_buffer("verts_object_og", verts_object_og)

        # Inititalize person parameters
        translation_init = translations_hand.detach().clone()
        self.translations_hand = nn.Parameter(translation_init,
                                              requires_grad=True)
        rotations_hand = rotations_hand.detach().clone()
        self.obj_rot_mult = 1  # This scaling has no effect !
        if rotations_hand.shape[-1] == 3:
            rotations_hand = matrix_to_rot6d(rotations_hand)
        self.rotations_hand = nn.Parameter(rotations_hand, requires_grad=True)
        if optimize_ortho_cam:
            self.cams_hand = nn.Parameter(cams_hand, requires_grad=True)
        else:
            self.register_buffer("cams_hand", cams_hand)
        self.hand_sides = hand_sides
        self.hand_nb = len(hand_sides)

        self.optimize_mano = optimize_mano
        if optimize_mano:
            self.mano_pca_pose = nn.Parameter(mano_pca_pose,
                                              requires_grad=True)
            self.mano_rot = nn.Parameter(mano_rot, requires_grad=True)
            self.mano_trans = nn.Parameter(mano_trans, requires_grad=True)
        else:
            self.register_buffer("mano_pca_pose", mano_pca_pose)
            self.register_buffer("mano_rot", mano_rot)
        if optimize_mano_beta:
            self.mano_betas = nn.Parameter(torch.zeros_like(mano_betas),
                                           requires_grad=True)
            self.register_buffer("int_scales_hand",
                                 torch.ones(1).float() * int_scale_init)
        else:
            self.register_buffer("mano_betas", torch.zeros_like(mano_betas))
            self.int_scales_hand = nn.Parameter(
                int_scale_init * torch.ones(1).float(),
                requires_grad=True,
            )
        self.register_buffer("verts_hand_og", verts_hand_og)
        self.register_buffer("ref_verts2d_hand", ref_verts2d_hand)

        init_scales = int_scale_init * torch.ones(1).float()
        init_scales_mean = torch.Tensor(int_scale_init).float()
        self.optimize_object_scale = optimize_object_scale
        if optimize_object_scale:
            self.int_scales_object = nn.Parameter(
                init_scales,
                requires_grad=True,
            )
        else:
            self.register_buffer("int_scales_object", init_scales)
        self.register_buffer("int_scale_object_mean", torch.ones(1).float())

        self.register_buffer("int_scale_hand_mean",
                             torch.Tensor([1.0]).float().cuda())
        self.register_buffer("ref_mask_object",
                             (target_masks_object > 0).float())
        self.register_buffer("keep_mask_object",
                             (target_masks_object >= 0).float())
        self.register_buffer("ref_mask_hand", (target_masks_hand > 0).float())
        self.register_buffer("keep_mask_hand",
                             (target_masks_hand >= 0).float())
        self.register_buffer("camintr_rois_object", camintr_rois_object)
        self.register_buffer("camintr_rois_hand", camintr_rois_hand)
        self.register_buffer("faces_object", faces_object)
        self.register_buffer(
            "textures_object",
            torch.ones(faces_object.shape[0], faces_object.shape[1], 1, 1, 1,
                       3))
        self.register_buffer(
            "textures_hand",
            torch.ones(faces_hand.shape[0], faces_hand.shape[1], 1, 1, 1, 3))
        self.register_buffer("faces_hand", faces_hand)
        self.cuda()

        # Setup renderer
        if camintr is None:
            camintr = torch.cuda.FloatTensor([[[1, 0, 0.5], [0, 1, 0.5],
                                               [0, 0, 1]]])
        else:
            camintr = npt.tensorify(camintr)
            if camintr.dim() == 2:
                camintr = camintr.unsqueeze(0)
            camintr = camintr.cuda().float()
        rot = torch.cuda.FloatTensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
        trans = torch.zeros(1, 3).cuda()
        self.register_buffer("camintr", camintr)
        self.image_size = image_size
        self.renderer = nr.renderer.Renderer(image_size=self.image_size,
                                             K=camintr.clone(),
                                             R=rot,
                                             t=trans,
                                             orig_size=1)
        self.renderer.light_direction = [1, 0.5, 1]
        self.renderer.light_intensity_direction = 0.3
        self.renderer.light_intensity_ambient = 0.5
        self.renderer.background_color = [1.0, 1.0, 1.0]
        if masks_hand is not None:
            self.register_buffer("masks_human", masks_hand)
        if masks_object.dim() == 2:
            masks_object = masks_object.unsqueeze(0)
        self.register_buffer("masks_object", masks_object)
        verts_object, _ = self.get_verts_object()
        verts_hand, _ = self.get_verts_hand()
        ref_verts_list = [verts_object[:1]] + [
            verts_hand[hand_idx:hand_idx + 1]
            for hand_idx in range(self.hand_nb)
        ]
        ref_faces_list = [faces_object[:1]] + [
            faces_hand[hand_idx:hand_idx + 1]
            for hand_idx in range(self.hand_nb)
        ]
        pred_colors = ["gold"] + [
            "grey",
        ] * self.hand_nb
        gt_colors = ["green"] + [
            "blue",
        ] * self.hand_nb

        faces, textures = get_faces_and_textures(ref_verts_list,
                                                 ref_faces_list,
                                                 color_names=pred_colors)
        # Assumes only one object
        batch_size = verts_object.shape[0]
        self.faces = faces.repeat(batch_size, 1, 1)
        self.textures = textures.repeat(batch_size, 1, 1, 1, 1, 1)
        faces_gt, textures_gt = get_faces_and_textures(ref_verts_list,
                                                       ref_faces_list,
                                                       color_names=gt_colors)
        self.textures_gt = textures_gt.repeat(batch_size, 1, 1, 1, 1, 1)
        self.faces_gt = faces_gt.repeat(batch_size, 1, 1)

        faces_with_gt, textures_with_gt = get_faces_and_textures(
            ref_verts_list + ref_verts_list,
            ref_faces_list + ref_faces_list,
            color_names=pred_colors + gt_colors)

        self.textures_with_gt = textures_with_gt.repeat(
            batch_size, 1, 1, 1, 1, 1)
        self.faces_with_gt = faces_with_gt.repeat(batch_size, 1, 1)
        self.losses = Losses(
            renderer=self.renderer,
            ref_mask_object=self.ref_mask_object,
            keep_mask_object=self.keep_mask_object,
            ref_mask_hand=self.ref_mask_hand,
            ref_verts2d_hand=self.ref_verts2d_hand,
            keep_mask_hand=self.keep_mask_hand,
            camintr_rois_object=self.camintr_rois_object,
            camintr_rois_hand=self.camintr_rois_hand,
            camintr=self.camintr,
            class_name=class_name,
            hand_nb=self.hand_nb,
            inter_type=inter_type,
        )
        verts_hand_init, _ = self.get_verts_hand()
        verts_object_init, _ = self.get_verts_object()
        self.verts_hand_init = verts_hand_init.detach().clone()
        self.verts_object_init = verts_object_init.detach().clone()

    def assign_human_masks(self, masks_human=None, min_overlap=0.5):
        """
        Uses a greedy matching algorithm to assign masks to human instances. The
        assigned human masks are used to compute the ordinal depth loss.

        If the human predictor uses the same instances as the segmentation algorithm,
        then this greedy assignment is unnecessary as the human instances will already
        have corresponding masks.

        1. Compute IOU between all human silhouettes and human masks
        2. Sort IOUs
        3. Assign people to masks in order, skipping people and masks that
            have already been assigned.

        Args:
            masks_human: Human bitmask tensor from instance segmentation algorithm.
            min_overlap (float): Minimum IOU threshold to assign the human mask to a
                human instance.

        Returns:
            N_h x
        """
        f = self.faces_hand
        verts_hand, _ = self.get_verts_hand()
        if masks_human is None:
            return torch.zeros(verts_hand.shape[0], self.image_size,
                               self.image_size).cuda()
        person_silhouettes = torch.cat([
            self.renderer(v.unsqueeze(0), f, mode="silhouettes")
            for v in verts_hand
        ]).bool()

        intersection = masks_human.unsqueeze(0) & person_silhouettes.unsqueeze(
            1)
        union = masks_human.unsqueeze(0) | person_silhouettes.unsqueeze(1)

        iou = intersection.sum(dim=(2, 3)).float() / union.sum(
            dim=(2, 3)).float()
        iou = iou.cpu().numpy()
        # https://stackoverflow.com/questions/30577375
        best_indices = np.dstack(
            np.unravel_index(np.argsort(-iou.ravel()), iou.shape))[0]
        human_indices_used = set()
        mask_indices_used = set()
        # If no match found, mask will just be empty, incurring 0 loss for depth.
        human_masks = torch.zeros(verts_hand.shape[0], self.image_size,
                                  self.image_size).bool()
        for human_index, mask_index in best_indices:
            if human_index in human_indices_used:
                continue
            if mask_index in mask_indices_used:
                continue
            if iou[human_index, mask_index] < min_overlap:
                break
            human_masks[human_index] = masks_human[mask_index]
            human_indices_used.add(human_index)
            mask_indices_used.add(mask_index)
        return human_masks.cuda()

    def get_verts_object(self):
        rotations_object = rot6d_to_matrix(self.obj_rot_mult *
                                           self.rotations_object)
        obj_verts = compute_transformation_persp(
            meshes=self.verts_object_og,
            translations=self.translations_object,
            rotations=rotations_object,
            intrinsic_scales=self.int_scales_object.abs(),
        )
        return obj_verts

    def get_joints_hand(self):
        all_hand_joints = []
        for hand_idx, side in enumerate(self.hand_sides):
            mano_pca_pose = self.mano_pca_pose[hand_idx::self.hand_nb]
            mano_rot = self.mano_rot[hand_idx::self.hand_nb]
            mano_res = self.mano_model.forward_pca(
                mano_pca_pose,
                rot=mano_rot,
                betas=self.mano_betas[hand_idx::self.hand_nb],
                side=side)
            joints = mano_res["joints"]
            verts = mano_res["verts"]
            # Add finger tips and reorder
            tips = verts[:, [745, 317, 444, 556, 673]]
            full_joints = torch.cat([joints, tips], 1)
            full_joints = full_joints[:, [
                0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7,
                8, 9, 20
            ]]
            all_hand_joints.append(full_joints)

        all_hand_joints = torch.stack(all_hand_joints).transpose(
            0, 1).contiguous().view(-1, 21, 3)
        joints_hand_og = all_hand_joints + self.mano_trans.unsqueeze(1)
        rotations_hand = rot6d_to_matrix(self.rotations_hand)
        return compute_transformation_persp(
            meshes=joints_hand_og,
            translations=self.translations_hand,
            rotations=rotations_hand,
            intrinsic_scales=self.int_scales_hand,
        )

    def get_verts_hand(self, detach_scale=False):
        if self.optimize_mano:
            all_hand_verts = []
            for hand_idx, side in enumerate(self.hand_sides):
                mano_pca_pose = self.mano_pca_pose[hand_idx::self.hand_nb]
                mano_rot = self.mano_rot[hand_idx::self.hand_nb]
                mano_res = self.mano_model.forward_pca(
                    mano_pca_pose,
                    rot=mano_rot,
                    betas=self.mano_betas[hand_idx::self.hand_nb],
                    side=side)
                vertices = mano_res["verts"]
                all_hand_verts.append(vertices)
            all_hand_verts = torch.stack(all_hand_verts).transpose(
                0, 1).contiguous().view(-1, 778, 3)
            verts_hand_og = all_hand_verts + self.mano_trans.unsqueeze(1)
        else:
            verts_hand_og = self.verts_hand_og
        if detach_scale:
            scale = self.int_scales_hand.detach()
        else:
            scale = self.int_scales_hand
        rotations_hand = rot6d_to_matrix(self.rotations_hand)
        if self.hand_proj_mode == "ortho":
            return compute_transformation_ortho(
                meshes=verts_hand_og,
                cams=self.cams_hand,
                intrinsic_scales=scale,
                K=self.renderer.K,
                img=self.masks_human,
            )
        elif self.hand_proj_mode == "persp":
            return compute_transformation_persp(
                meshes=verts_hand_og,
                translations=self.translations_hand,
                rotations=rotations_hand,
                intrinsic_scales=scale,
            )
        else:
            raise ValueError(
                f"Expected hand_proj_mode {self.hand_proj_mode} to be in [ortho|persp]"
            )

    def compute_ordinal_depth_loss(self):
        verts_object, _ = self.get_verts_object()
        verts_hand, _ = self.get_verts_hand()

        silhouettes = []
        depths = []

        _, depths_o, silhouettes_o = self.renderer.render(
            verts_object, self.faces_object, self.textures_object)
        silhouettes_o = (silhouettes_o == 1).bool()
        silhouettes.append(silhouettes_o)
        depths.append(depths_o)

        hand_masks = []
        for hand_idx, faces, textures in zip(range(self.hand_nb),
                                             self.faces_hand,
                                             self.textures_hand):
            hand_verts = verts_hand[hand_idx::self.hand_nb]
            repeat_nb = hand_verts.shape[0]
            faces_hand = faces.unsqueeze(0).repeat(repeat_nb, 1, 1)
            textures_hand = textures.unsqueeze(0).repeat(
                repeat_nb, 1, 1, 1, 1, 1)
            _, depths_p, silhouettes_p = self.renderer.render(
                hand_verts, faces_hand, textures_hand)
            silhouettes_p = (silhouettes_p == 1).bool()
            # imagify.viz_imgrow(cols, "tmp_hands.png")
            silhouettes.append(silhouettes_p)
            depths.append(depths_p)
            hand_masks.append(self.masks_human[hand_idx::self.hand_nb])

        all_masks = [self.masks_object] + [
            self.masks_human[hand_idx::self.hand_nb]
            for hand_idx in range(self.hand_nb)
        ]
        masks = torch.stack(all_masks, 1)
        return lossutils.compute_ordinal_depth_loss(masks, silhouettes, depths)

    def forward(self, loss_weights=None):
        """
        If a loss weight is zero, that loss isn't computed (to avoid unnecessary
        compute).
        """
        loss_dict = {}
        metric_dict = {}
        verts_object, _ = self.get_verts_object()
        # verts_hand_det has MANO mesh detached, which allows to backpropagate
        # coarse interaction loss simply in translation
        verts_hand, verts_hand_det = self.get_verts_hand()
        verts_hand_det_scale, _ = self.get_verts_hand(detach_scale=True)
        if loss_weights is None or loss_weights["lw_pca"] > 0:
            loss_pca = lossutils.compute_pca_loss(self.mano_pca_pose)
            loss_dict.update(loss_pca)
        if loss_weights is None or ((loss_weights["lw_smooth_hand"] > 0) or
                                    (loss_weights["lw_smooth_obj"] > 0)):
            loss_smooth = lossutils.compute_smooth_loss(
                verts_hand=verts_hand,
                verts_obj=verts_object,
            )
            loss_dict.update(loss_smooth)
        if loss_weights is None or loss_weights["lw_collision"] > 0:
            # Pushes hand out of object, gradient not flowing through object !
            loss_coll = lossutils.compute_collision_loss(
                verts_hand=verts_hand_det_scale,
                verts_object=verts_object.detach(),
                faces_object=self.faces_object,
                faces_hand=self.faces_hand)
            loss_dict.update(loss_coll)

        if loss_weights is None or loss_weights["lw_contact"] > 0:
            loss_contact, _ = lossutils.compute_contact_loss(
                verts_hand_b=verts_hand_det_scale,
                verts_object_b=verts_object,
                faces_object=self.faces_object,
                faces_hand=self.faces_hand)
            loss_dict.update(loss_contact)
        if loss_weights is None or loss_weights["lw_v2d_hand"] > 0:
            if self.optimize_object_scale:
                loss_verts2d, metric_verts2d = self.losses.compute_verts2d_loss_hand(
                    verts=verts_hand,
                    image_size=self.image_size,
                    min_hand_size=70)
            else:
                loss_verts2d, metric_verts2d = self.losses.compute_verts2d_loss_hand(
                    verts=verts_hand,
                    image_size=self.image_size,
                    min_hand_size=1000)
            loss_dict.update(loss_verts2d)
            metric_dict.update(metric_verts2d)
        if loss_weights is None or loss_weights["lw_sil_obj"] > 0:
            sil_loss_dict, sil_metric_dict = self.losses.compute_sil_loss_object(
                verts=verts_object, faces=self.faces_object)
            loss_dict.update(sil_loss_dict)
            metric_dict.update(sil_metric_dict)

            # loss_dict.update(
            #     self.losses.compute_sil_loss_hand(verts=verts_hand,
            #                                         faces=[self.faces_hand] *
            #                                         len(verts_hand)))
        if loss_weights is None or loss_weights["lw_inter"] > 0:
            # Interaction acts only on hand !
            if not self.optimize_object_scale:
                inter_verts_object = verts_object.unsqueeze(1).detach()
            else:
                inter_verts_object = verts_object.unsqueeze(1)
            inter_loss_dict, inter_metric_dict = self.losses.compute_interaction_loss(
                verts_hand_b=verts_hand_det.view(-1, self.hand_nb, 778, 3),
                verts_object_b=inter_verts_object)
            loss_dict.update(inter_loss_dict)
            metric_dict.update(inter_metric_dict)

        if loss_weights is None or loss_weights["lw_scale_obj"] > 0:
            loss_dict[
                "loss_scale_obj"] = lossutils.compute_intrinsic_scale_prior(
                    intrinsic_scales=self.int_scales_object,
                    intrinsic_mean=self.int_scale_object_mean,
                )
        if loss_weights is None or loss_weights["lw_scale_hand"] > 0:
            loss_dict[
                "loss_scale_hand"] = lossutils.compute_intrinsic_scale_prior(
                    intrinsic_scales=self.int_scales_hand,
                    intrinsic_mean=self.int_scale_hand_mean,
                )
        if loss_weights is None or loss_weights["lw_depth"] > 0:
            loss_dict.update(lossutils.compute_ordinal_depth_loss())
        return loss_dict, metric_dict

    def render_limem(self,
                     renderer,
                     verts,
                     faces,
                     textures,
                     K,
                     max_in_batch=5):
        sample_nb = verts.shape[0]
        check_shape(verts, (-1, -1, 3))
        check_shape(faces, (sample_nb, -1, 3))
        check_shape(textures, (sample_nb, -1, 1, 1, 1, 3))
        check_shape(K, (sample_nb, 3, 3))

        if max_in_batch is not None:
            chunk_nb = (sample_nb + 1) // min(max_in_batch, sample_nb)
        else:
            chunk_nb = 1
        verts_chunks = verts.chunk(chunk_nb, 0)
        faces_chunks = faces.chunk(chunk_nb, 0)
        textures_chunks = textures.chunk(chunk_nb, 0)
        K_chunks = K.chunk(chunk_nb, 0)
        all_images = []
        all_masks = []
        for vert, face, tex, camintr in zip(verts_chunks, faces_chunks,
                                            textures_chunks, K_chunks):
            chunk_images, _, chunk_masks = renderer.render(vertices=vert,
                                                           faces=face,
                                                           textures=tex,
                                                           K=camintr)
            all_images.append(
                np.clip(npt.numpify(chunk_images).transpose(0, 2, 3, 1), 0, 1))
            all_masks.append(npt.numpify(chunk_masks).astype(bool))
        all_images = np.concatenate(all_images)
        all_masks = np.concatenate(all_masks)
        return all_images, all_masks

    def render(self, renderer, rotate=False, viz_len=10, max_in_batch=None):
        verts_object = self.get_verts_object()[0]
        verts_hands = self.get_verts_hand()[0]
        verts_hands = [
            verts_hands[hand_idx::self.hand_nb]
            for hand_idx in range(self.hand_nb)
        ]
        verts_combined = combine_verts([verts_object] + verts_hands)
        if rotate:
            verts_combined = trans3d.rot_points(verts_combined)
        images, masks = self.render_limem(renderer,
                                          verts_combined[:viz_len],
                                          self.faces[:viz_len],
                                          self.textures[:viz_len],
                                          K=renderer.K[:viz_len],
                                          max_in_batch=max_in_batch)
        return images, masks

    def render_gt(self,
                  renderer,
                  verts_hand_gt=None,
                  verts_object_gt=None,
                  rotate=False,
                  viz_len=10,
                  max_in_batch=None):
        verts_list = [verts_object_gt, verts_hand_gt]
        verts_combined = combine_verts(verts_list)
        if rotate:
            verts_combined = trans3d.rot_points(verts_combined)
        images, masks = self.render_limem(renderer,
                                          verts_combined[:viz_len],
                                          self.faces[:viz_len],
                                          self.textures_gt[:viz_len],
                                          K=renderer.K[:viz_len],
                                          max_in_batch=max_in_batch)
        return images, masks

    def render_with_gt(self,
                       renderer,
                       verts_hand_gt=None,
                       verts_object_gt=None,
                       rotate=False,
                       viz_len=10,
                       init=False,
                       max_in_batch=None):
        if init:
            verts_object_pred = self.verts_object_init
            verts_hands = self.verts_hand_init
        else:
            verts_object_pred = self.get_verts_object()[0]
            verts_hands = self.get_verts_hand()[0]
        verts_hands_pred = [
            verts_hands[hand_idx::self.hand_nb]
            for hand_idx in range(self.hand_nb)
        ]
        verts_hands_gt = [verts for verts in verts_hand_gt]
        verts_list = [verts_object_pred
                      ] + verts_hands_pred + [verts_object_gt] + verts_hands_gt
        verts_combined = combine_verts(verts_list)
        if rotate:
            verts_combined = trans3d.rot_points(verts_combined)
        images, masks = self.render_limem(renderer,
                                          verts_combined[:viz_len],
                                          self.faces_with_gt[:viz_len],
                                          self.textures_with_gt[:viz_len],
                                          K=renderer.K[:viz_len],
                                          max_in_batch=max_in_batch)
        return images, masks

    def save_obj(self, fname):
        with open(fname, "w") as fp:
            verts_combined = combine_verts(
                [self.get_verts_object()[0],
                 self.get_verts_hand()[0]])
            for v in tqdm.tqdm(verts_combined[0]):
                fp.write(f"v {v[0]:f} {v[1]:f} {v[2]:f}\n")
            o = 1
            for face in tqdm.tqdm(self.faces[0]):
                fp.write(
                    f"f {face[0] + o:d} {face[1] + o:d} {face[2] + o:d}\n")
