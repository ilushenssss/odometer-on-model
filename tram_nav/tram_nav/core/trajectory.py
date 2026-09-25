"""Trajectory reconstruction metrics.

The navigator outputs the along-track distance ``s``; the 2-D trajectory is
obtained through the track map ``s -> (x, y, yaw)``.  These helpers compare
an estimated 2-D trajectory with the true one:

* 2-D (Euclidean) position error, its along-track / cross-track components;
* heading error;
* distance of every estimated point from the rail line (map consistency);
* continuity: the largest jump between consecutive published points while the
  tram is moving.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import numpy as np

from .track import TrackMap


def trajectory_from_s(track: TrackMap, s: Sequence[float]) -> np.ndarray:
    """``(N, 3)`` array of ``x, y, yaw`` for a sequence of along-track distances."""
    return np.array([track.pose(float(v)) for v in s])


def distance_to_track(track: TrackMap, x: float, y: float, s_hint: Optional[float] = None,
                      window: float = 200.0) -> float:
    """Shortest distance from point ``(x, y)`` to the rail polyline.

    With ``s_hint`` only the part of the polyline within ``window`` metres is
    searched (fast); without it the whole line is searched.
    """
    xs, ys, ss = track.x, track.y, track.s
    if s_hint is not None:
        i0 = max(0, int(np.searchsorted(ss, s_hint - window)) - 1)
        i1 = min(len(ss), int(np.searchsorted(ss, s_hint + window)) + 1)
    else:
        i0, i1 = 0, len(ss)
    ax, ay = xs[i0:i1 - 1], ys[i0:i1 - 1]
    bx, by = xs[i0 + 1:i1], ys[i0 + 1:i1]
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    L2[L2 == 0.0] = 1e-12
    tt = ((x - ax) * dx + (y - ay) * dy) / L2
    lo = np.zeros_like(tt)
    hi = np.ones_like(tt)
    # beyond the ends the map continues along the terminal tangent (TrackMap.pose)
    if i0 == 0:
        lo[0] = -np.inf
    if i1 == len(ss):
        hi[-1] = np.inf
    tt = np.clip(tt, lo, hi)
    px, py = ax + tt * dx, ay + tt * dy
    return float(np.min(np.hypot(x - px, y - py)))


def wrap_angle(a: np.ndarray) -> np.ndarray:
    return (np.asarray(a) + math.pi) % (2.0 * math.pi) - math.pi


def trajectory_metrics(track: TrackMap, t: np.ndarray, s_true: np.ndarray, s_est: np.ndarray,
                       xy_true: np.ndarray, xy_est: np.ndarray, yaw_true: Optional[np.ndarray] = None,
                       yaw_est: Optional[np.ndarray] = None, v_true: Optional[np.ndarray] = None,
                       sigma_s: Optional[np.ndarray] = None, check_on_track: bool = True,
                       track_step: int = 10) -> Dict[str, float]:
    """Metrics of an estimated 2-D trajectory against the ground truth."""
    e2d = np.hypot(xy_est[:, 0] - xy_true[:, 0], xy_est[:, 1] - xy_true[:, 1])
    e_along = np.abs(np.asarray(s_est) - np.asarray(s_true))
    out = {
        "rms_2d": float(np.sqrt(np.mean(e2d ** 2))),
        "mean_2d": float(np.mean(e2d)),
        "p95_2d": float(np.percentile(e2d, 95)),
        "max_2d": float(np.max(e2d)),
        "final_2d": float(e2d[-1]),
        "max_along": float(np.max(e_along)),
        # a chord is never longer than the arc: 2-D error <= along-track error
        "chord_le_arc": float(np.mean(e2d <= e_along + 1e-6)),
        "within_10m": float(np.mean(e2d <= 10.0)),
        "within_25m": float(np.mean(e2d <= 25.0)),
    }
    if yaw_true is not None and yaw_est is not None:
        de = np.degrees(np.abs(wrap_angle(np.asarray(yaw_est) - np.asarray(yaw_true))))
        out["heading_p50_deg"] = float(np.median(de))
        out["heading_p95_deg"] = float(np.percentile(de, 95))
        out["heading_max_deg"] = float(np.max(de))
    if sigma_s is not None:
        out["within_3sigma"] = float(np.mean(e2d <= 3.0 * np.asarray(sigma_s) + 0.5))
    if check_on_track:
        idx = range(0, len(s_est), max(1, track_step))
        d = [distance_to_track(track, xy_est[i, 0], xy_est[i, 1], s_est[i]) for i in idx]
        out["max_off_track"] = float(np.max(d))
    if v_true is not None and len(t) > 1:
        step = np.hypot(np.diff(xy_est[:, 0]), np.diff(xy_est[:, 1]))
        dt = np.diff(t)
        moving = np.asarray(v_true)[1:] > 1.0
        if moving.any():
            # jump beyond what the speed allows (margin: 5 m/s + 1 m)
            excess = step[moving] - (np.asarray(v_true)[1:][moving] + 5.0) * dt[moving] - 1.0
            out["max_jump_moving"] = float(max(0.0, np.max(excess)))
    return out
