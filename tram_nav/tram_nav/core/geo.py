"""Geodetic helpers: WGS-84 latitude/longitude <-> local ENU plane.

The track map and the navigator work in a local metric frame (x = east,
y = north).  GNSS references (sensor_msgs/NavSatFix) are converted with a
local tangent-plane approximation around ``origin`` - the error is below
1 cm per km for city-scale lines (<= 30 km).
"""
from __future__ import annotations

import math
from typing import Tuple

A = 6378137.0                 # WGS-84 semi-major axis
F = 1.0 / 298.257223563
E2 = F * (2.0 - F)


def _radii(lat_rad: float) -> Tuple[float, float]:
    s = math.sin(lat_rad)
    w = math.sqrt(1.0 - E2 * s * s)
    rn = A / w                          # prime vertical
    rm = A * (1.0 - E2) / (w ** 3)      # meridian
    return rm, rn


class LocalFrame:
    def __init__(self, lat0: float, lon0: float, alt0: float = 0.0):
        self.lat0, self.lon0, self.alt0 = lat0, lon0, alt0
        rm, rn = _radii(math.radians(lat0))
        self.ky = rm                                    # m per rad latitude
        self.kx = (rn + alt0) * math.cos(math.radians(lat0))  # m per rad longitude

    def to_xy(self, lat: float, lon: float) -> Tuple[float, float]:
        return (math.radians(lon - self.lon0) * self.kx, math.radians(lat - self.lat0) * self.ky)

    def to_latlon(self, x: float, y: float) -> Tuple[float, float]:
        return (self.lat0 + math.degrees(y / self.ky), self.lon0 + math.degrees(x / self.kx))
