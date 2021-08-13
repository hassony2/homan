#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Epic-Kitchens dataset utilities
from https://github.com/epic-kitchens/epic-kitchens-100-object-masks
"""

from enum import Enum, unique
from itertools import chain
from typing import Dict, Iterator, List, Tuple, cast

import numpy as np
from dataclasses import dataclass

import homan.datasets.types_pb2 as pb

__all__ = [
    "HandSide",
    "HandState",
    "FloatVector",
    "BBox",
    "HandDetection",
    "ObjectDetection",
    "FrameDetections",
]


@unique
class HandSide(Enum):
    LEFT = 0
    RIGHT = 1


@unique
class HandState(Enum):
    """An enum describing the different states a hand can be in:
    - No contact: The hand isn't touching anything
    - Self contact: The hand is touching itself
    - Another person: The hand is touching another person
    - Portable object: The hand is in contact with a portable object
    - Stationary object: The hand is in contact with an immovable/stationary object"""

    NO_CONTACT = 0
    SELF_CONTACT = 1
    ANOTHER_PERSON = 2
    PORTABLE_OBJECT = 3
    STATIONARY_OBJECT = 4


@dataclass
class FloatVector:
    """A floating-point 2D vector representation"""

    x: np.float32
    y: np.float32

    def to_protobuf(self) -> pb.FloatVector:
        vector = pb.FloatVector()
        vector.x = self.x
        vector.y = self.y
        assert vector.IsInitialized()
        return vector

    @staticmethod
    def from_protobuf(vector: pb.FloatVector) -> "FloatVector":
        return FloatVector(x=vector.x, y=vector.y)

    def __add__(self, other: "FloatVector") -> "FloatVector":
        return FloatVector(x=self.x + other.x, y=self.y + other.y)

    def __mul__(self, scaler: float) -> "FloatVector":
        return FloatVector(x=self.x * scaler, y=self.y * scaler)

    def __iter__(self) -> Iterator[float]:
        yield from (self.x, self.y)

    @property
    def coord(self) -> Tuple[float, float]:
        """Return coordinates as a tuple"""
        return (self.x, self.y)

    def scale(self, width_factor: float = 1, height_factor: float = 1) -> None:
        """Scale x component by ``width_factor`` and y component by ``height_factor``"""
        self.x *= width_factor
        self.y *= height_factor


@dataclass
class BBox:
    left: float
    top: float
    right: float
    bottom: float

    def to_protobuf(self) -> pb.BBox:
        bbox = pb.BBox()
        bbox.left = self.left
        bbox.top = self.top
        bbox.right = self.right
        bbox.bottom = self.bottom
        assert bbox.IsInitialized()
        return bbox

    @staticmethod
    def from_protobuf(bbox: pb.BBox) -> "BBox":
        return BBox(
            left=bbox.left,
            top=bbox.top,
            right=bbox.right,
            bottom=bbox.bottom,
        )

    @property
    def center(self) -> Tuple[float, float]:
        x = (self.left + self.right) / 2
        y = (self.top + self.bottom) / 2
        return x, y

    @property
    def center_int(self) -> Tuple[int, int]:
        """Get center position as a tuple of integers (rounded)"""
        x, y = self.center
        return (round(x), round(y))

    def scale(self, width_factor: float = 1, height_factor: float = 1) -> None:
        self.left *= width_factor
        self.right *= width_factor
        self.top *= height_factor
        self.bottom *= height_factor

    def center_scale(self,
                     width_factor: float = 1,
                     height_factor: float = 1) -> None:
        x, y = self.center
        new_width = self.width * width_factor
        new_height = self.height * height_factor
        self.left = x - new_width / 2
        self.right = x + new_width / 2
        self.top = y - new_height / 2
        self.bottom = y + new_height / 2

    @property
    def coords(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return (
            self.top_left,
            self.bottom_right,
        )

    @property
    def coords_int(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return (
            self.top_left_int,
            self.bottom_right_int,
        )

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def top_left(self) -> Tuple[float, float]:
        return (self.left, self.top)

    @property
    def bottom_right(self) -> Tuple[float, float]:
        return (self.right, self.bottom)

    @property
    def top_left_int(self) -> Tuple[int, int]:
        return (round(self.left), round(self.top))

    @property
    def bottom_right_int(self) -> Tuple[int, int]:
        return (round(self.right), round(self.bottom))


@dataclass
class HandDetection:
    """Dataclass representing a hand detection, consisting of a bounding box,
    a score (representing the model's confidence this is a hand), the predicted state
    of the hand, whether this is a left/right hand, and a predicted offset to the
    interacted object if the hand is interacting."""

    bbox: BBox
    score: np.float32
    state: HandState
    side: HandSide
    object_offset: FloatVector

    def to_protobuf(self) -> pb.HandDetection:
        detection = pb.HandDetection()
        detection.bbox.MergeFrom(self.bbox.to_protobuf())
        detection.score = self.score
        detection.state = self.state.value
        detection.object_offset.MergeFrom(self.object_offset.to_protobuf())
        detection.side = self.side.value
        assert detection.IsInitialized()
        return detection

    @staticmethod
    def from_protobuf(detection: pb.HandDetection) -> "HandDetection":
        return HandDetection(
            bbox=BBox.from_protobuf(detection.bbox),
            score=detection.score,
            state=HandState(detection.state),
            object_offset=FloatVector.from_protobuf(detection.object_offset),
            side=HandSide(detection.side),
        )

    def scale(self, width_factor: float = 1, height_factor: float = 1) -> None:
        self.bbox.scale(width_factor=width_factor, height_factor=height_factor)
        self.object_offset.scale(width_factor=width_factor,
                                 height_factor=height_factor)

    def center_scale(self,
                     width_factor: float = 1,
                     height_factor: float = 1) -> None:
        self.bbox.center_scale(width_factor=width_factor,
                               height_factor=height_factor)


@dataclass
class ObjectDetection:
    """Dataclass representing an object detection, consisting of a bounding box and a
    score (the model's confidence this is an object)"""

    bbox: BBox
    score: np.float32

    def to_protobuf(self) -> pb.ObjectDetection:
        detection = pb.ObjectDetection()
        detection.bbox.MergeFrom(self.bbox.to_protobuf())
        detection.score = self.score
        assert detection.IsInitialized()
        return detection

    @staticmethod
    def from_protobuf(detection: pb.ObjectDetection) -> "ObjectDetection":
        return ObjectDetection(bbox=BBox.from_protobuf(detection.bbox),
                               score=detection.score)

    def scale(self, width_factor: float = 1, height_factor: float = 1) -> None:
        self.bbox.scale(width_factor=width_factor, height_factor=height_factor)

    def center_scale(self,
                     width_factor: float = 1,
                     height_factor: float = 1) -> None:
        self.bbox.center_scale(width_factor=width_factor,
                               height_factor=height_factor)


@dataclass
class FrameDetections:
    """Dataclass representing hand-object detections for a frame of a video"""

    video_id: str
    frame_number: int
    objects: List[ObjectDetection]
    hands: List[HandDetection]

    def to_protobuf(self) -> pb.Detections:
        detections = pb.Detections()
        detections.video_id = self.video_id
        detections.frame_number = self.frame_number
        detections.hands.extend([hand.to_protobuf() for hand in self.hands])
        detections.objects.extend(
            [object.to_protobuf() for object in self.objects])
        assert detections.IsInitialized()
        return detections

    @staticmethod
    def from_protobuf(detections: pb.Detections) -> "FrameDetections":
        return FrameDetections(
            video_id=detections.video_id,
            frame_number=detections.frame_number,
            hands=[HandDetection.from_protobuf(pb) for pb in detections.hands],
            objects=[
                ObjectDetection.from_protobuf(pb) for pb in detections.objects
            ],
        )

    @staticmethod
    def from_protobuf_str(pb_str: bytes) -> "FrameDetections":
        pb_detection = pb.Detections()
        pb_detection.MergeFromString(pb_str)
        return FrameDetections.from_protobuf(pb_detection)

    def get_hand_object_interactions(
            self,
            object_threshold: float = 0,
            hand_threshold: float = 0) -> Dict[int, int]:
        """Match the hands to objects based on the hand offset vector that the model
        uses to predict the location of the interacted object.
        Args:
            object_threshold: Object score threshold above which to consider objects
                for matching
            hand_threshold: Hand score threshold above which to consider hands for
                matching.
        Returns:
            A dictionary mapping hand detections to objects by indices
        """
        interactions = dict()
        object_idxs = [
            i for i, obj in enumerate(self.objects)
            if obj.score >= object_threshold
        ]
        object_centers = np.array(
            [self.objects[object_id].bbox.center for object_id in object_idxs])
        for hand_idx, hand_detection in enumerate(self.hands):
            if (hand_detection.state.value == HandState.NO_CONTACT.value
                    or hand_detection.score <= hand_threshold):
                continue
            estimated_object_position = np.array(
                hand_detection.bbox.center) + np.array(
                    hand_detection.object_offset.coord)
            distances = ((object_centers -
                          estimated_object_position)**2).sum(axis=-1)
            interactions[hand_idx] = object_idxs[cast(int,
                                                      np.argmin(distances))]
        return interactions

    def scale(self, width_factor: float = 1, height_factor: float = 1) -> None:
        """
        Scale the coordinates of all the hands/objects. x components are multiplied
        by the ``width_factor`` and y components by the ``height_factor``
        """
        for det in chain(self.hands, self.objects):
            det.scale(width_factor=width_factor, height_factor=height_factor)

    def center_scale(self,
                     width_factor: float = 1,
                     height_factor: float = 1) -> None:
        """
        Scale all the hands/objects about their center points.
        """
        for det in chain(self.hands, self.objects):
            det.center_scale(width_factor=width_factor,
                             height_factor=height_factor)
