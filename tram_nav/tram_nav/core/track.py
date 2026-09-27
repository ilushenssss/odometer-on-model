"""Track (rail line) map: 1-D along-track coordinate <-> 2-D/3-D position.

A tram is constrained to the rails, so its full pose is a function of a single
coordinate - the travelled distance ``s``.  The map is used to

* convert the estimated distance into ``x, y, yaw`` for publication;
* provide the grade ``theta(s)`` and curvature ``kappa(s)`` to the motion
  model (known disturbances);
* provide station positions to the simulated driver.

CSV format (header required): ``s,x,y,z[,station]`` - ``s`` may be omitted,
it is then computed as the cumulative 2-D length.
"""
from __future__ import annotations

import bisect
import csv
import math
from typing import List, Optional, Sequence

import numpy as np


class TrackMap:
    def __init__(self, x: Sequence[float], y: Sequence[float], z: Optional[Sequence[float]] = None,
                 s: Optional[Sequence[float]] = None, stations: Optional[Sequence[float]] = None,
                 smooth_window: float = 10.0):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        z = np.zeros_like(x) if z is None else np.asarray(z, float)
        if s is None:
            ds = np.hypot(np.diff(x), np.diff(y))
            s = np.concatenate([[0.0], np.cumsum(ds)])
        s = np.asarray(s, float)
        if len(s) < 2 or np.any(np.diff(s) <= 0):
            raise ValueError("track 's' must be strictly increasing with >= 2 points")
        self.s, self.x, self.y, self.z = s, x, y, z
        self.length = float(s[-1])
        self.stations: List[float] = sorted(float(v) for v in (stations or []))
        self.geo = None           # optional LocalFrame (WGS-84 origin)

        heading = np.unwrap(np.arctan2(np.gradient(y, s), np.gradient(x, s)))
        self.heading = heading
        kappa = np.gradient(heading, s)
        grade = np.arctan(np.gradient(z, s))
        # moving-average smoothing to remove polyline artefacts
        step = float(np.median(np.diff(s)))
        n = max(1, int(round(smooth_window / max(step, 1e-6))))
        if n > 1:
            ker = np.ones(n) / n
            kappa = np.convolve(kappa, ker, mode="same")
            grade = np.convolve(grade, ker, mode="same")
        self.curvature = kappa
        self.grade = grade
        # python lists for fast scalar lookups
        self._s = s.tolist()
        self._x, self._y = x.tolist(), y.tolist()
        self._h = heading.tolist()
        self._k, self._g = kappa.tolist(), grade.tolist()

    # ------------------------------------------------------------------
    def _locate(self, s: float):
        s = min(max(s, self._s[0]), self._s[-1])
        i = bisect.bisect_right(self._s, s) - 1
        i = min(max(i, 0), len(self._s) - 2)
        s0, s1 = self._s[i], self._s[i + 1]
        return i, (s - s0) / (s1 - s0)

    @staticmethod
    def _lerp(arr, i, w):
        return arr[i] + w * (arr[i + 1] - arr[i])

    def pose(self, s: float):
        """Return ``(x, y, yaw)`` at distance ``s``.  Beyond the ends the
        position is extrapolated along the terminal tangent."""
        i, w = self._locate(s)
        x, y, h = self._lerp(self._x, i, w), self._lerp(self._y, i, w), self._lerp(self._h, i, w)
        extra = 0.0
        if s > self._s[-1]:
            extra = s - self._s[-1]
        elif s < self._s[0]:
            extra = s - self._s[0]
        if extra:
            x += extra * math.cos(h)
            y += extra * math.sin(h)
        return x, y, h

    def grade_at(self, s: float) -> float:
        i, w = self._locate(s)
        return self._lerp(self._g, i, w)

    def curvature_at(self, s: float) -> float:
        i, w = self._locate(s)
        return self._lerp(self._k, i, w)

    def project(self, x: float, y: float) -> float:
        """Along-track distance of the point of the line closest to ``(x, y)``."""
        ax, ay = self.x[:-1], self.y[:-1]
        dx, dy = np.diff(self.x), np.diff(self.y)
        L2 = np.maximum(dx * dx + dy * dy, 1e-12)
        t = np.clip(((x - ax) * dx + (y - ay) * dy) / L2, 0.0, 1.0)
        d2 = (ax + t * dx - x) ** 2 + (ay + t * dy - y) ** 2
        i = int(np.argmin(d2))
        return float(self.s[i] + t[i] * (self.s[i + 1] - self.s[i]))

    def next_station(self, s: float, margin: float = 2.0) -> Optional[float]:
        i = bisect.bisect_right(self.stations, s + margin)
        return self.stations[i] if i < len(self.stations) else None

    # ------------------------------------------------------------------
    @classmethod
    def from_csv(cls, path: str, smooth_window: float = 10.0) -> "TrackMap":
        """Load a line map.  Columns: ``s,x,y,z,station`` (local metres) or
        ``lat,lon[,alt][,station]`` (WGS-84; the first point is the origin of the
        local frame, kept in ``TrackMap.geo``)."""
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError(f"empty track file {path}")
        names = rows[0].keys()
        geo = None
        if "lat" in names and "lon" in names:
            from .geo import LocalFrame
            geo = LocalFrame(float(rows[0]["lat"]), float(rows[0]["lon"]))
            xy = [geo.to_xy(float(r["lat"]), float(r["lon"])) for r in rows]
            xs, ys = [p[0] for p in xy], [p[1] for p in xy]
            zs = [float(r.get("alt", r.get("z", 0.0)) or 0.0) for r in rows]
        else:
            xs = [float(r["x"]) for r in rows]
            ys = [float(r["y"]) for r in rows]
            zs = [float(r.get("z", 0.0) or 0.0) for r in rows]
        has_s = "s" in names
        if has_s:
            s_arr = [float(r["s"]) for r in rows]
        else:
            ds = np.hypot(np.diff(xs), np.diff(ys))
            s_arr = list(np.concatenate([[0.0], np.cumsum(ds)]))
        st = [s_arr[i] for i, r in enumerate(rows) if int(float(r.get("station", 0) or 0))]
        tm = cls(xs, ys, zs, s_arr, st, smooth_window)
        tm.geo = geo
        return tm

    def to_csv(self, path: str, step: float = 5.0):
        grid = np.arange(0.0, self.length + 1e-9, step)
        st_idx = {int(round(v / step)) for v in self.stations}
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["s", "x", "y", "z", "station"])
            for k, s in enumerate(grid):
                x, y, _ = self.pose(s)
                i, a = self._locate(s)
                z = self.z[i] + a * (self.z[i + 1] - self.z[i])
                w.writerow([f"{s:.2f}", f"{x:.3f}", f"{y:.3f}", f"{z:.3f}", int(k in st_idx)])


def generate_demo_track(seed: int = 7, step: float = 1.0) -> TrackMap:
    """Synthetic ~6.5 km urban tram line: straights, tight curves (R 25..300 m),
    grades up to +-50 per mille and 10 stations."""
    rng = np.random.default_rng(seed)
    # (length [m], radius [m] or 0 for straight, turn direction)
    segs = [
        (320, 0, 0), (60, 60, 1), (540, 0, 0), (45, 30, -1), (380, 0, 0), (220, 300, 1),
        (410, 0, 0), (70, 45, -1), (600, 0, 0), (150, 150, -1), (350, 0, 0), (40, 25, 1),
        (520, 0, 0), (260, 250, 1), (480, 0, 0), (80, 80, -1), (700, 0, 0), (120, 120, 1),
        (560, 0, 0),
    ]
    xs, ys, hs = [0.0], [0.0], 0.0
    for length, radius, direction in segs:
        n = max(1, int(length / step))
        ds = length / n
        for _ in range(n):
            if radius:
                hs += direction * ds / radius
            xs.append(xs[-1] + ds * math.cos(hs))
            ys.append(ys[-1] + ds * math.sin(hs))
    xs, ys = np.array(xs), np.array(ys)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
    # smooth random vertical profile, grade limited to 50 per mille
    knots = np.linspace(0, s[-1], 18)
    gk = rng.uniform(-0.045, 0.045, size=len(knots))
    gk[0] = gk[-1] = 0.0
    grade = np.interp(s, knots, gk)
    grade = np.convolve(grade, np.ones(51) / 51, mode="same")
    z = np.concatenate([[0.0], np.cumsum(grade[1:] * np.diff(s))])
    stations = [0.0] + list(np.linspace(0, s[-1], 11)[1:-1] + rng.uniform(-60, 60, 9)) + [s[-1] - 5.0]
    stations = [5.0 * round(float(v) / 5.0) for v in stations]  # on the 5 m CSV grid
    return TrackMap(xs, ys, z, s, stations)
