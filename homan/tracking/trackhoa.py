#!/usr/bin/env python
# -*- coding: utf-8 -*-

from motpy import Detection, MultiObjectTracker
import pandas as pd
import numpy as np
from tqdm import tqdm

from homan.datasets import epichoa as gethoa
from homan.tracking import trackconv


def print_track_info(tracks, track_type="object"):
    unique_tracks = tracks.track_id.unique()
    print(f"Got {len(unique_tracks)} {track_type} tracks with ids and lengths"
          f"{tracks.groupby('track_id').frame.nunique()}")


def filter_longest_track(tracks, frame_field="frame"):
    longest_track_idx = (tracks.groupby("track_id").frame.nunique().idxmax())
    # Filter object which has longest track
    filtered_track = tracks[tracks.track_id == longest_track_idx]
    return filtered_track


def track_hoa_df(
    hoa_dets,
    dt=0.02,
    start_frame=0,
    end_frame=100,
    video_id=None,
    verbose=True,
):
    # Initialize track lists and tracker
    obj_tracker = MultiObjectTracker(dt=dt)
    tracked_obj = []

    lh_tracker = MultiObjectTracker(dt=dt)
    rh_tracker = MultiObjectTracker(dt=dt)

    # Intialize tracked dicts
    tracked_lh = []
    tracked_rh = []

    # Last non-empty df
    for frame_idx in tqdm(range(start_frame, end_frame)):
        hoa_df = hoa_dets[hoa_dets.frame == frame_idx]
        obj_df = hoa_df[hoa_df.det_type == "object"]
        obj_dets = [
            Detection(gethoa.row2box(row)) for _, row in obj_df.iterrows()
        ]
        obj_tracker.step(detections=obj_dets)
        tracked_obj.extend(
            trackconv.track2dicts(
                obj_tracker.active_tracks(),
                frame_idx,
                video_id=video_id,
                det_type="object",
            ))
        lh_df = hoa_df[(hoa_df.det_type == "hand") & (hoa_df.side == "left")]
        rh_df = hoa_df[(hoa_df.det_type == "hand") & (hoa_df.side == "right")]
        lh_dets = [
            Detection(gethoa.row2box(row)) for _, row in lh_df.iterrows()
        ]
        rh_dets = [
            Detection(gethoa.row2box(row)) for _, row in rh_df.iterrows()
        ]
        lh_tracker.step(detections=lh_dets)
        rh_tracker.step(detections=rh_dets)
        tracked_lh.extend(
            trackconv.track2dicts(
                lh_tracker.active_tracks(),
                frame_idx,
                video_id=video_id,
                det_type="hand",
                side="left",
            ))
        tracked_rh.extend(
            trackconv.track2dicts(
                rh_tracker.active_tracks(),
                frame_idx,
                video_id=video_id,
                det_type="hand",
                side="right",
            ))
    if verbose:
        obj_tracks = pd.DataFrame(tracked_obj)
        tracked_obj = filter_longest_track(obj_tracks)
        print_track_info(tracked_obj)
        lh_tracks = pd.DataFrame(tracked_lh)
        rh_tracks = pd.DataFrame(tracked_rh)
        if len(lh_tracks):
            print_track_info(lh_tracks, track_type="left hand")
        else:
            print(
                "No lh tracks for video_id {video_id} between {start_frame} and {end_frame}"
            )
        if len(rh_tracks):
            print_track_info(rh_tracks, track_type="right hand")
        else:
            print(
                "No rh tracks for video_id {video_id} between {start_frame} and {end_frame}"
            )
        start_obj_frame = tracked_obj.frame.min()
        end_obj_frame = tracked_obj.frame.max()
        # Keep only region that focuses on longest track
        if len(tracked_rh):
            tracked_rh = pd.DataFrame(tracked_rh)
            tracked_rh = tracked_rh[(tracked_rh.frame >= start_obj_frame)
                                    & (tracked_rh.frame <= end_obj_frame)]
            tracked_rh = filter_longest_track(tracked_rh)
            start_rh_frame = tracked_rh.frame.min()
            end_rh_frame = tracked_rh.frame.max()
            # Reduce hand and object tracks
            tracked_rh = tracked_rh[(tracked_rh.frame >= start_rh_frame)
                                    & (tracked_rh.frame <= end_rh_frame)]
            tracked_obj = tracked_obj[(tracked_obj.frame >= start_rh_frame)
                                      & (tracked_obj.frame <= end_rh_frame)]
        if len(tracked_lh):
            tracked_lh = pd.DataFrame(tracked_lh)
            tracked_lh = tracked_lh[(tracked_lh.frame > start_obj_frame)
                                    & (tracked_lh.frame <= end_obj_frame)]
            tracked_lh = filter_longest_track(tracked_lh)
            start_lh_frame = tracked_lh.frame.min()
            end_lh_frame = tracked_lh.frame.max()
            # Reduce hand and object tracks
            tracked_lh = tracked_lh[(tracked_lh.frame > start_lh_frame)
                                    & (tracked_lh.frame <= end_lh_frame)]
            tracked_obj = tracked_obj[(tracked_obj.frame > start_lh_frame)
                                      & (tracked_obj.frame <= end_lh_frame)]
            if len(tracked_rh):
                tracked_rh = tracked_rh[(tracked_rh.frame > start_lh_frame)
                                        & (tracked_rh.frame <= end_lh_frame)]

        # For now, discard frames  of epic to only keep ones whith both hand and object tracks,
        # It would be cleaner to interpolate !
        keep_frames = set(tracked_obj.frame)
        if len(tracked_rh):
            keep_frames = keep_frames & set(tracked_rh.frame)
        if len(tracked_lh):
            keep_frames = keep_frames & set(tracked_lh.frame)
        # Finally get final tracks
        res = {}
        tracked_obj = tracked_obj[tracked_obj.frame.isin(keep_frames)]
        # Interpolate missing boxes
        new_index = pd.Index(range(min(keep_frames), max(keep_frames) + 1))
        tracked_obj = tracked_obj[tracked_obj.frame.isin(keep_frames)]
        tracked_obj = tracked_obj.set_index("frame").reindex(
            new_index).reset_index().rename(columns={
                tracked_obj.index.name: 'frame'
            }).interpolate(method="linear")

        obj_boxes = np.stack([
            tracked_obj.left, tracked_obj.top, tracked_obj.right,
            tracked_obj.bottom
        ], 1)
        res = {"objects": obj_boxes}
        if len(tracked_rh):
            tracked_rh = tracked_rh[tracked_rh.frame.isin(keep_frames)]
            tracked_rh = tracked_rh.set_index("frame").reindex(
                new_index).reset_index().rename(columns={
                    tracked_rh.index.name: 'frame'
                }).interpolate(method="linear")
            # xyxy boxes
            rh_boxes = np.stack([
                tracked_rh.left, tracked_rh.top, tracked_rh.right,
                tracked_rh.bottom
            ], 1)
            res["right_hand"] = rh_boxes
        if len(tracked_lh):
            tracked_lh = tracked_lh[tracked_lh.frame.isin(keep_frames)]
            tracked_lh = tracked_lh.set_index("frame").reindex(
                new_index).reset_index().rename(columns={
                    tracked_lh.index.name: 'frame'
                }).interpolate(method="linear")
            lh_boxes = np.stack([
                tracked_lh.left, tracked_lh.top, tracked_lh.right,
                tracked_lh.bottom
            ], 1)
            res["left_hand"] = lh_boxes
    frame_idxs = np.array(tracked_obj["index"]).tolist()
    return frame_idxs, res
