#!/usr/bin/env python
# -*- coding: utf-8 -*-

from filterpy.common import kinematic_kf
from libyana.conversions import npt


def rtsmooth_th(measurements, dt=0.02, order=2):
    smoothed = rtsmooth(measurements, dt=dt, order=order)
    return measurements.new(smoothed)


def rtsmooth(measurements, dt=0.02, order=2):
    """
    Args:
        measurements (np.array): (time, measurements_dim)
    Returns:
        data (np.array): (time, measurements_dim)
    """
    measure_dim = measurements.shape[1]
    kf = kinematic_kf(dim=measure_dim, order=order, dt=dt)
    # print(kf.F[:3, :3])  # State transition
    # kf.P is ordered with [2, 2] or [3, 3] blocks for each dimension
    # (2 if 1st order - constant velocity, 3 if 2nd order - constant acceleration)
    kf.P[::order + 1, ::order + 1] *= 1
    kf.P *= 10
    kf.Q[::order + 1, ::order + 1] *= 1
    mu, cov, _, _ = kf.batch_filter(npt.numpify(measurements))
    smoothed, _, _, _ = kf.rts_smoother(mu, cov)
    print(smoothed.shape)
    return smoothed[:, ::order + 1, 0]
