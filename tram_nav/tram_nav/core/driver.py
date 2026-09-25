"""Simple human-like driver: follows speed limits, stops at every station.

Produces the controller handle position ``u`` in [-1, 1] with a limited
handle rate, a coasting band and occasional emergency brakes.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .params import TramParams
from .track import TrackMap


class DriverModel:
    def __init__(self, track: TrackMap, p: TramParams, rng: Optional[np.random.Generator] = None,
                 v_line: float = 16.5, a_lat: float = 0.7, b_plan: float = 0.9,
                 dwell: float = 20.0, handle_rate: float = 1.2):
        self.track = track
        self.p = p
        self.rng = rng or np.random.default_rng(0)
        self.b = b_plan
        self.dwell = dwell
        self.handle_rate = handle_rate
        self.u = 0.0
        self.state = "dwell"
        self.timer = 5.0
        self.emerg_until = -1.0
        self.target_station: Optional[float] = None
        # static speed profile: line speed, curve limits and braking curves
        step = 5.0
        grid = np.arange(0.0, track.length + step, step)
        kap = np.array([abs(track.curvature_at(s)) for s in grid])
        vlim = np.minimum(v_line, np.sqrt(a_lat / np.maximum(kap, 1e-6)))
        vlim = np.maximum(vlim, 4.0)
        for i in range(len(grid) - 2, -1, -1):  # backward pass - braking curves
            vlim[i] = min(vlim[i], math.sqrt(vlim[i + 1] ** 2 + 2 * b_plan * step))
        for i in range(1, len(grid)):  # forward pass - acceleration curves (cosmetic)
            vlim[i] = min(vlim[i], math.sqrt(vlim[i - 1] ** 2 + 2 * 1.2 * step)) if vlim[i - 1] > 0 else vlim[i]
        self._grid, self._vlim = grid, vlim

    def force_emergency(self, t: float, duration: float) -> None:
        self.emerg_until = t + duration

    def speed_target(self, s: float) -> float:
        v = float(np.interp(s, self._grid, self._vlim))
        if self.target_station is not None:
            d = self.target_station - s
            v = min(v, math.sqrt(max(0.0, 2.0 * self.b * (d - 0.5))))
        return v

    def update(self, t: float, s: float, v: float, dt: float) -> float:
        if t < self.emerg_until:
            self.u = -1.0
            return self.u
        if self.state == "dwell":
            self.timer -= dt
            u_des = -0.5
            if self.timer <= 0.0:
                self.target_station = self.track.next_station(s, margin=3.0)
                self.state = "run" if self.target_station is not None else "end"
        elif self.state == "end":
            u_des = -0.5
        else:
            vt = self.speed_target(s)
            e = vt - v
            if e > 0.8:
                u_des = min(1.0, 0.35 + 0.4 * e)
            elif e > 0.15:
                u_des = 0.25 if self.u > 0.05 else 0.0  # hold or coast
            elif e > -0.4:
                u_des = 0.0
            else:
                u_des = max(-0.9, 0.55 * e)
            near = self.target_station is not None and self.target_station - s < 4.0
            if near or vt < 0.3:
                u_des = -0.7 if v > 0.05 else -0.5
            if near and v < 0.05:
                self.state = "dwell"
                self.timer = self.dwell + float(self.rng.uniform(-5, 5))
                u_des = -0.5
        du = u_des - self.u
        lim = self.handle_rate * dt
        self.u += max(-lim, min(lim, du))
        return self.u
