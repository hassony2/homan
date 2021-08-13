#!/usr/bin/env python
# -*- coding: utf-8 -*-
from itertools import permutations

import torch
import torch.nn as nn
import numpy as np

from sdf import SDF
from libyana.verify import checkshape


class SDFSceneLoss(nn.Module):
    def __init__(self, faces, grid_size=32, robustifier=None, debugging=False):
        """
        Args:
            faces (list): List of faces for each object in the scene

        """
        super(SDFSceneLoss, self).__init__()
        for faces_idx, face in enumerate(faces):
            if isinstance(face, torch.Tensor):
                face = face.int()
            elif isinstance(face, np.ndarray):
                face = torch.Tensor(face).int()
            else:
                raise TypeError(f"{type(face)} not in [ndarray, torch.Tensor]")
            checkshape.check_shape(face, (-1, 3), name="faces")
            self.register_buffer(f'faces{faces_idx}', face)
        self.num_objects = len(faces)

        self.sdf = SDF()
        self.grid_size = grid_size
        self.robustifier = robustifier
        self.debugging = debugging

    @torch.no_grad()
    def get_bounding_boxes(self, vertices):
        """
        Returns:
            (box_nb, 2, 3): where for each box [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        """
        num_people = vertices.shape[0]
        boxes = torch.zeros(num_people, 2, 3, device=vertices.device)
        for i in range(num_people):
            boxes[i, 0, :] = vertices[i].min(dim=0)[0]
            boxes[i, 1, :] = vertices[i].max(dim=0)[0]
        return boxes

    @torch.no_grad()
    def check_overlap(self, bbox1, bbox2):
        # check x
        if bbox1[0, 0] > bbox2[1, 0] or bbox2[0, 0] > bbox1[1, 0]:
            return False
        #check y
        if bbox1[0, 1] > bbox2[1, 1] or bbox2[0, 1] > bbox1[1, 1]:
            return False
        #check z
        if bbox1[0, 2] > bbox2[1, 2] or bbox2[0, 2] > bbox1[1, 2]:
            return False
        return True

    def filter_isolated_boxes(self, boxes):

        num_people = boxes.shape[0]
        isolated = torch.zeros(num_people,
                               device=boxes.device,
                               dtype=torch.bool)
        for i in range(num_people):
            isolated_i = False
            for j in range(num_people):
                if j != i:
                    isolated_i |= not self.check_overlap(boxes[i], boxes[j])
            isolated[i] = isolated_i
        return isolated

    def forward(self, vertices, scale_factor=0.2):
        """
        Args:
            vertices (list): list of (scene_nb, -1, 3) vertices
        """
        num_objects = len(vertices)
        num_scenes = vertices[0].shape[0]
        vertices = [verts.float() for verts in vertices]

        assert self.num_objects == len(
            vertices
        ), f"Got {len(vertices)} vertices for {self.num_objects} faces"

        for verts in vertices:
            checkshape.check_shape(verts, (-1, -1, 3), name="vertices")
        # If only one person in the scene, return 0
        loss = torch.tensor(0., device=vertices[0].device)
        if num_objects == 1:
            return loss
        scene_boxes = []
        for verts in vertices:
            boxes = self.get_bounding_boxes(verts)
            scene_boxes.append(boxes)
        # scene_nb, object_nb, 2 (mins, maxs), 3 (xyz)
        scene_boxes = torch.stack(scene_boxes, 1)

        # Filter out the isolated boxes
        scene_boxes_center = scene_boxes.mean(dim=2).unsqueeze(dim=2).permute(
            1, 0, 2, 3)
        scene_boxes_scale = ((scene_boxes[:, :, 1] - scene_boxes[:, :, 0]) *
                             ((1 + scale_factor) * 0.5)).max(
                                 dim=-1)[0].permute(1, 0)
        obj_phis = []
        for obj_idx, (boxes_center, boxes_scale, verts) in enumerate(
                zip(scene_boxes_center, scene_boxes_scale, vertices)):
            with torch.no_grad():
                verts_centered = verts - boxes_center
                verts_centered_scaled = verts_centered / boxes_scale.view(
                    verts_centered.shape[0], 1, 1)
                assert (verts_centered_scaled.min() >= -1)
                assert (verts_centered_scaled.max() <= 1)
                faces = getattr(self, f"faces{obj_idx}")
                phi = self.sdf(faces, verts_centered_scaled.contiguous())
                # Keep only inside values
                phi = phi.clamp(0)
                assert (phi.min() >= 0)
                obj_phis.append(phi)

        pair_idxs = list(permutations(list(range(self.num_objects)), 2))
        dist_values = {}
        for idx1, idx2 in pair_idxs:
            # Get 1st object Signed Distance Field
            phi1 = obj_phis[idx1]
            # Get matching normalization values
            boxes_center1 = scene_boxes_center[idx1]
            boxes_scale1 = scene_boxes_scale[idx1]

            # Apply to 2nd object
            verts2 = vertices[idx2]
            verts2_local = (verts2 - boxes_center1) / boxes_scale1.view(
                verts2.shape[0], 1, 1)

            dist_vals = nn.functional.grid_sample(
                phi1.float().unsqueeze(1),
                verts2_local.view(verts.shape[0], verts2.shape[1], 1, 1, 3))
            # Get SDF values back in original scale
            dist_values[(
                idx1,
                idx2)] = dist_vals[:, 0, :, 0, 0] * boxes_scale1.unsqueeze(1)

            loss += dist_vals.sum()
        return loss, {"sdfs": obj_phis, "dist_values": dist_values}
