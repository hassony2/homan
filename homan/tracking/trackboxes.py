# pylint: disable=broad-except,too-many-statements,too-many-branches,logging-fstring-interpolation,import-error
import numpy as np

from motpy.tracker import get_single_object_tracker
from motpy.model import ModelPreset, Model
from homan.utils.bbox import bbox_wh_to_xy


def track_boxes(boxes_to_track, dt=0.5, out_xyxy=True):
    """
    Arguments:
        boxes_to_track (list): [(x_min, y_min, x_max, y_max), ..., None, ...] 
    Returns:
        smoothed_boxes (np.ndarray): [frame_nb, 4] where each row encoder box
            in (x_min, y_min, x_max, y_max) format
    """
    track_model = Model(dt=dt,
                        order_pos=0,
                        dim_pos=2,
                        order_size=0,
                        dim_size=2)

    # Converts x_min, y_min, x_max, y_max to c_x, c_x', c_y, c_y', w, h
    processed_boxes = [
        (track_model.box_to_z(box).tolist() if box is not None else None)
        for box in boxes_to_track
    ]
    tracker = get_single_object_tracker(track_model)
    # Fix beginning artefact
    tracker.batch_filter(processed_boxes)
    mu, cov, _, _ = tracker.batch_filter(processed_boxes)
    smoothed_state, _, _, _ = tracker.rts_smoother(mu, cov)

    smoothed_boxes = np.stack(
        [track_model.x_to_box(smooth[:, 0]) for smooth in smoothed_state])
    if out_xyxy:
        smoothed_boxes = bbox_wh_to_xy(smoothed_boxes)
    return smoothed_boxes
