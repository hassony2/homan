#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Epic-Kitchens dataset utilities
from https://github.com/epic-kitchens/epic-kitchens-100-object-masks
"""

from pathlib import Path
from typing import List, Union

from .types import FrameDetections


def load_detections(path: Union[str, Path]) -> List[FrameDetections]:
    """
    Load detections from file.
    Args:
        path: Path to detections pickle. This should contain a pickled list of
            serialized protobuf descriptions of detections
    Returns:
        Deserialized detections contained in pickle.
    """
    import pickle

    with open(path, "rb") as f:
        return [FrameDetections.from_protobuf_str(s) for s in pickle.load(f)]


def save_detections(detections: List[FrameDetections],
                    path: Union[str, Path]) -> None:
    """
    Save detections to file.
    Args:
        detections: A list of detections. These should be ordered by frame.
        path: Path to write serialized detections to. Non-existent folders in the
            path are created.
    """
    import pickle

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump([d.to_protobuf().SerializeToString() for d in detections],
                    f)
