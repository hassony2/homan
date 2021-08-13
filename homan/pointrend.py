# Copyright (c) Facebook, Inc. and its affiliates.
"""
Wrapper for PointRend Segmentation algorithm.
Reference: Kirillov et al. "PointRend: Image Segmentation as Rendering." (CVPR 2020).
"""
import numpy as np
import torch

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.structures import BitMasks
from detectron2.data import transforms
from detectron2.structures import Instances, Boxes

try:
    import point_rend
except Exception:
    from detectron2.projects import point_rend
from homan.constants import (
    BBOX_EXPANSION_FACTOR,
    POINTREND_CONFIG,
    POINTREND_MODEL_WEIGHTS,
    REND_SIZE,
)
from homan.utils.bbox import bbox_wh_to_xy, bbox_xy_to_wh, make_bbox_square


class MaskExtractor:
    def __init__(self,
                 pointrend_model_weights=POINTREND_MODEL_WEIGHTS,
                 pointrend_config=POINTREND_CONFIG):
        self.cfg = get_cfg()
        self.predictor = get_pointrend_predictor(
            min_confidence=0,
            pointrend_model_weights=pointrend_model_weights,
            pointrend_config=pointrend_config)
        self.aug = transforms.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST],
            self.cfg.INPUT.MAX_SIZE_TEST,
        )
        self.bbox_expansion = BBOX_EXPANSION_FACTOR

    def preprocess_img(self, original_image, input_format="BGR"):
        if input_format == "RGB":
            # whether the model expects BGR inputs or RGB
            original_image = original_image[:, :, ::-1]
        height, width = original_image.shape[:2]
        image = self.aug.get_transform(original_image).apply_image(
            original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0,
                                                                  1)).cuda()

        inputs = {"image": image, "height": height, "width": width}
        return inputs

    def masks_from_bboxes(self,
                          im,
                          boxes_wh,
                          pred_classes,
                          class_idx=-1,
                          input_format="RGB",
                          rend_size=REND_SIZE,
                          image_size=640):
        """
        Args:
            class_idx (int): coco class index, -1 for using the most likely predicted class
            boxes (np.array): (-1, 4) xyxy
        Returns:
            dict: {'square_boxes': (xywh)}
        """
        boxes_xy = [bbox_wh_to_xy(box) for box in boxes_wh]
        model = self.predictor.model

        # Initialize boxes
        if not isinstance(boxes_xy, torch.Tensor):
            boxes_xy = torch.Tensor(boxes_xy)
        if pred_classes is None:
            pred_classes = class_idx * torch.ones(len(boxes_xy)).long()
        else:
            if not isinstance(pred_classes, torch.Tensor):
                pred_classes = torch.Tensor(pred_classes)
            pred_classes = pred_classes.long()
        # Clamp boxes to valid image region !
        boxes_xy[:, :2].clamp_(0, max(im.shape))
        boxes_xy[:, 3].clamp_(0, im.shape[0] + 1)
        boxes_xy[:, 2].clamp_(0, im.shape[1] + 1)
        trans_boxes = Boxes(self.aug.get_transform(im).apply_box(boxes_xy))
        inp_im = self.preprocess_img(im, input_format=input_format)
        _, height, width = inp_im["image"].shape
        try:
            instances = Instances(
                image_size=(height, width),
                pred_boxes=trans_boxes,
                pred_classes=pred_classes,
            )
        except Exception:
            import pdb
            pdb.set_trace()

        # Preprocess image
        inf_out = model.inference([inp_im], [instances])

        # Extract masks
        instance = inf_out[0]["instances"]
        masks = instance.pred_masks
        inst_boxes = instance.pred_boxes.tensor
        try:
            scores = instance.scores
        except AttributeError:
            scores = masks.new_ones(masks.shape[0])
        pred_classes = instance.pred_classes
        bit_masks = BitMasks(masks.cpu())
        keep_annotations = []
        full_boxes = torch.tensor([[0, 0, image_size, image_size]] *
                                  len(inst_boxes)).float()
        full_sized_masks = bit_masks.crop_and_resize(full_boxes, image_size)

        for bbox_idx, box in enumerate(inst_boxes):
            bbox = bbox_xy_to_wh(box.cpu())  # xy_wh
            square_bbox = make_bbox_square(bbox, self.bbox_expansion)
            square_boxes = torch.FloatTensor(
                np.tile(bbox_wh_to_xy(square_bbox),
                        (len(instances), 1)))  # xy_xy
            crop_masks = bit_masks.crop_and_resize(square_boxes,
                                                   rend_size).clone().detach()
            keep_annotations.append({
                "bbox":
                bbox,
                "class_id":
                pred_classes[bbox_idx],
                "full_mask":
                full_sized_masks[bbox_idx, :im.shape[0], :im.shape[1]].cpu(),
                "score":
                scores[bbox_idx],
                "square_bbox":
                square_bbox,  # xy_wh
                "crop_mask":
                crop_masks[0].cpu().numpy(),
            })
        return keep_annotations


def get_pointrend_predictor(min_confidence=0.9,
                            image_format="RGB",
                            pointrend_config=POINTREND_CONFIG,
                            pointrend_model_weights=POINTREND_MODEL_WEIGHTS):
    cfg = get_cfg()
    point_rend.add_pointrend_config(cfg)
    cfg.merge_from_file(pointrend_config)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = min_confidence
    cfg.MODEL.WEIGHTS = pointrend_model_weights
    cfg.INPUT.FORMAT = image_format
    return DefaultPredictor(cfg)


def get_class_masks_from_instances(
    instances,
    class_id=1,
    add_ignore=True,
    rend_size=REND_SIZE,
    bbox_expansion=BBOX_EXPANSION_FACTOR,
    min_confidence=0.0,
    image_size=640,
):
    """
    Gets occlusion-aware masks for a specific class index and additional metadata from
    PointRend instances.

    Args:
        instances: Detectron2 Instances with segmentation predictions.
        class_id (int): Object class id (using COCO dense ordering).
        add_ignore (bool): If True, adds occlusion-aware masking.
        rend_size (int): Mask size.
        bbox_expansion (float): Amount to pad the masks. This is important to prevent
            ignoring background pixels right outside the bounding box.
        min_confidence (float): Minimum confidence threshold for masks.

    Returns:
        keep_masks (N x rend_size x rend_size).
        keep_annotations (dict):
            "bbox":
            "class_id":
            "segmentation":
            "square_bbox":
    """
    if len(instances) == 0:
        return [], []
    instances = instances.to(torch.device("cpu:0"))
    boxes = instances.pred_boxes.tensor.numpy()
    class_ids = instances.pred_classes.numpy()
    scores = instances.scores.numpy()
    keep_ids = np.logical_and(class_ids == class_id, scores > min_confidence)
    bit_masks = BitMasks(instances.pred_masks)

    keep_annotations = []
    keep_masks = []
    full_boxes = torch.tensor([[0, 0, image_size, image_size]] *
                              len(boxes)).float()
    full_sized_masks = bit_masks.crop_and_resize(full_boxes, image_size)
    for k in np.where(keep_ids)[0]:
        bbox = bbox_xy_to_wh(boxes[k])
        square_bbox = make_bbox_square(bbox, bbox_expansion)
        square_boxes = torch.FloatTensor(
            np.tile(bbox_wh_to_xy(square_bbox), (len(instances), 1)))
        masks = bit_masks.crop_and_resize(square_boxes,
                                          rend_size).clone().detach()
        if add_ignore:
            ignore_mask = masks[0]
            for i in range(1, len(masks)):
                ignore_mask = ignore_mask | masks[i]
            ignore_mask = -ignore_mask.float().numpy()
        else:
            ignore_mask = np.zeros(rend_size, rend_size)
        m = ignore_mask.copy()
        mask = masks[k]
        m[mask] = mask[mask]
        keep_masks.append(m)
        keep_annotations.append({
            "bbox": bbox,
            "class_id": class_ids[k],
            "mask": full_sized_masks[k],
            "score": scores[k],
            "square_bbox": square_bbox,
        })
    return keep_masks, keep_annotations
