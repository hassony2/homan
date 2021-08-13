#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=import-error,no-member,wrong-import-order,too-many-arguments,too-many-instance-attributes,comparison-with-itself
# pylint: disable=missing-function-docstring,missing-module-docstring
import numpy as np
import torch

from homan.interactions import contactloss, scenesdf

import trimesh
from libyana.visutils import imagify

# MANO_CLOSED_FACES = np.array(
#     trimesh.load("extra_data/mano/closed_fmano.obj", process=False).faces)
MANO_CLOSED_FACES = np.load("local_data/closed_fmano.npy")


def compute_smooth_loss(
    verts_hand,
    verts_obj,
):
    # Assumes a single obj
    hand_nb = verts_hand.shape[0] // verts_obj.shape[0]
    time_hands = [verts_hand[hand_idx::hand_nb] for hand_idx in range(hand_nb)]
    all_hand_verts = torch.cat(time_hands, 1)
    # from libyana.visutils import imagify
    # imagify.viz_pointsrow(all_hand_verts, "tmp.png")
    # imagify.viz_pointsrow(all_hand_verts[:, :, 1:], "tmpz.png")
    # import pudb
    # pu.db
    smooth_loss_hand = ((all_hand_verts[1:] - all_hand_verts[:-1])**2).mean()
    smooth_loss_obj = ((verts_obj[1:] - verts_obj[:-1])**2).mean()
    return {
        "loss_smooth_obj": smooth_loss_obj,
        "loss_smooth_hand": smooth_loss_hand
    }


def compute_pca_loss(mano_pca_comps):
    return {"loss_pca": (mano_pca_comps**2).mean()}


def compute_collision_loss(verts_hand,
                           verts_object,
                           faces_hand,
                           faces_object,
                           max_collisions=5000,
                           debug=True,
                           collision_mode="sdf"):
    if collision_mode == "sdf":
        hand_nb = verts_hand.shape[0] // verts_object.shape[0]
        mano_faces = faces_object[0].new(MANO_CLOSED_FACES)
        if hand_nb > 1:
            mano_faces = faces_object[0].new(MANO_CLOSED_FACES[:, ::-1].copy())
            sdfl = scenesdf.SDFSceneLoss(
                [mano_faces, mano_faces, faces_object[0]])
            hand_verts = [
                verts_hand[hand_idx::2] for hand_idx in range(hand_nb)
            ]
            sdf_loss, sdf_meta = sdfl(hand_verts + [verts_object])
        else:
            sdfl = scenesdf.SDFSceneLoss([mano_faces, faces_object[0]])
            sdf_loss, sdf_meta = sdfl([verts_hand, verts_object])
        return {"loss_collision": sdf_loss.mean()}
    else:
        import mesh_intersection.loss as collisions_loss
        from mesh_intersection.bvh_search_tree import BVH

        coll_loss = collisions_loss.DistanceFieldPenetrationLoss(
            sigma=0.5, point2plane=1, vectorized=True)

        batch_size = verts_hand.shape[0] // faces_hand.shape[0]
        hand_nb = verts_hand.shape[0] // batch_size
        hand_faces_b = [
            faces_hand[idx % hand_nb] + verts_hand.shape[1] * idx
            for idx in range(verts_hand.shape[0])
        ]
        obj_faces_b = [
            faces_object[0] + verts_object.shape[1] * idx
            for idx in range(verts_object.shape[0])
        ]
        hand_faces_b = torch.stack(hand_faces_b).long()
        obj_faces_b = torch.stack(obj_faces_b).long()
        hand_triangles = verts_hand.view(-1, 3)[hand_faces_b]
        obj_triangles = verts_object.view(-1, 3)[obj_faces_b]
        all_triangles = [
            hand_triangles[hand_idx::hand_nb] for hand_idx in range(hand_nb)
        ] + [obj_triangles]
        # Group hand and object triangles into one single mesh
        all_triangles = torch.cat(all_triangles, 1)

        search_tree = BVH(max_collisions=max_collisions)
        all_coll_idxs = []
        if debug:
            print("Colliding !")
        with torch.no_grad():
            all_coll_idxs = search_tree(all_triangles)
        # imagify.viz_pointsrow(
        #     [all_triangles[0].view(-1, 3)],
        #     overlay_list=[all_triangles[0][all_coll_idxs[0]].view(1, -1, 3)])
        colls = coll_loss(all_triangles, all_coll_idxs)
        if (colls != colls).sum():
            colls = torch.Tensor(0.0).float().cuda()
        return {"loss_collision": colls.mean()}


def compute_intrinsic_scale_prior(intrinsic_scales, intrinsic_mean):
    return torch.sum(
        (intrinsic_scales - intrinsic_mean)**2) / intrinsic_scales.shape[0]


def compute_contact_loss(verts_hand_b, verts_object_b, faces_object,
                         faces_hand):
    hand_nb = verts_hand_b.shape[0] // verts_object_b.shape[0]
    faces_hand_closed = faces_hand.new(MANO_CLOSED_FACES).unsqueeze(0)
    if hand_nb > 1:
        missed_losses = []
        contact_losses = []
        for hand_idx in range(hand_nb):
            hand_verts = verts_hand_b[hand_idx::hand_nb]
            missed_loss, contact_loss, _, _ = contactloss.compute_contact_loss(
                hand_verts, faces_hand_closed, verts_object_b, faces_object)
            missed_losses.append(missed_loss)
            contact_losses.append(contact_loss)
        missed_loss = torch.stack(missed_losses).mean()
        contact_loss = torch.stack(contact_losses).mean()
    else:
        missed_loss, contact_loss, _, _ = contactloss.compute_contact_loss(
            verts_hand_b, faces_hand_closed, verts_object_b, faces_object)
    return {"loss_contact": missed_loss + contact_loss}, None


def compute_ordinal_depth_loss(masks, silhouettes, depths):
    """
    Args:
        masks (torch.Tensor): B, obj_nb, height, width
        silhouettes (list[torch.Tensor]): [(B, height, width), ...] of len obj_nb
        depths (list[torch.Tensor]): [(B, height, width), ...] of len obj_nb
    """
    loss = torch.Tensor(0.0).float().cuda()
    num_pairs = 0
    # Create square mask to match square renders
    height = masks.shape[2]
    width = masks.shape[3]
    silhouettes = [silh[:, :height, :width] for silh in silhouettes]
    depths = [depth[:, :height, :width] for depth in depths]
    # TODO comment
    imagify.viz_imgrow([
        masks[0].sum(0), silhouettes[0][0], masks[0][0], silhouettes[1][0],
        masks[0][1], silhouettes[2][0]
    ], "tmpdebugsilor.png")
    imagify.viz_imgrow(depths[0], "tmpdebugdepths.png")
    for i in range(len(silhouettes)):
        for j in range(len(silhouettes)):
            has_pred = silhouettes[i] & silhouettes[j]
            pairs = (has_pred.sum([1, 2]) > 0).sum().item()
            if pairs == 0:
                continue
            num_pairs += pairs
            front_i_gt = masks[:, i] & (~masks[:, j])
            front_j_pred = depths[j] < depths[i]
            mask = front_i_gt & front_j_pred & has_pred
            if mask.sum() == 0:
                continue
            dists = torch.clamp(depths[i] - depths[j], min=0.0, max=2.0)
            loss += torch.sum(
                torch.log(1 + torch.exp(dists))[mask]) / mask.sum()
    loss /= num_pairs
    return {"loss_depth": loss}
