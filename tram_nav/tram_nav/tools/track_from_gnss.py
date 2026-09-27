#!/usr/bin/env python3
"""Build the line map (track CSV) from a reference GNSS run stored in a rosbag.

    ros2 run tram_nav track_from_gnss --bag <bag> --topic /tram/gnss --out track.csv

The tram runs on rails, so one reference pass with GNSS gives the map used by
the navigator afterwards (in operation the navigator needs no GNSS):

1. NavSatFix -> local ENU (origin = first fix);
2. samples while standing (GNSS speed < --stop-speed) are dropped, long stops
   (>= --min-dwell s) become platform candidates;
3. the path is smoothed (moving average over --smooth m) and resampled every
   --step m by arc length;
4. written as ``lat,lon,alt,station``.
"""
from __future__ import annotations

import argparse
import math

import numpy as np


def read_fixes(bag: str, topic: str):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import NavSatFix
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id=""), rosbag2_py.ConverterOptions("cdr", "cdr"))
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    out = []
    while reader.has_next():
        _, data, t_ns = reader.read_next()
        m = deserialize_message(data, NavSatFix)
        if m.status.status >= 0 and math.isfinite(m.latitude) and math.isfinite(m.longitude):
            t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9 or t_ns * 1e-9
            out.append((t, m.latitude, m.longitude, m.altitude if math.isfinite(m.altitude) else 0.0))
    if not out:
        raise SystemExit(f"no valid NavSatFix messages on {topic}")
    return np.array(sorted(out))


def _time_smooth(t, a, win):
    """Moving average over +-win seconds (cumulative sums, O(n))."""
    c = np.concatenate([[0.0], np.cumsum(a)])
    lo = np.searchsorted(t, t - win)
    hi = np.searchsorted(t, t + win, side="right")
    return (c[hi] - c[lo]) / np.maximum(hi - lo, 1)


def build_track(t, x, y, z, step=2.0, smooth=15.0, stop_speed=0.5, min_dwell=10.0, smooth_z=250.0,
                smooth_t=1.0, speed_win=5.0):
    # 1) GNSS noise: moving average in time before anything else
    x, y = _time_smooth(t, x, smooth_t), _time_smooth(t, y, smooth_t)
    # 2) speed over +-speed_win s (robust to the remaining noise)
    a = np.searchsorted(t, t - speed_win)
    b = np.minimum(np.searchsorted(t, t + speed_win), len(t) - 1)
    dt = np.maximum(t[b] - t[a], 1e-3)
    v = np.hypot(x[b] - x[a], y[b] - y[a]) / dt
    moving = v >= stop_speed
    # stops -> platform candidates (median position of the stop)
    stops = []
    i = 0
    while i < len(t):
        if not moving[i]:
            j = i
            while j < len(t) and not moving[j]:
                j += 1
            if t[j - 1] - t[i] >= min_dwell:
                stops.append((float(np.median(x[i:j])), float(np.median(y[i:j]))))
            i = j
        else:
            i += 1
    xm, ym, zm = x[moving], y[moving], z[moving]
    # rough arc length, then moving-average smoothing in arc length
    d = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xm), np.diff(ym)))])
    keep = np.concatenate([[True], np.diff(d) > 1e-3])
    xm, ym, zm, d = xm[keep], ym[keep], zm[keep], d[keep]
    grid = np.arange(0.0, d[-1], step)
    gx, gy, gz = np.interp(grid, d, xm), np.interp(grid, d, ym), np.interp(grid, d, zm)
    n = max(1, int(round(smooth / step)))
    if n > 1:
        k = np.ones(n) / n
        pad = lambda a: np.concatenate([np.full(n, a[0]), a, np.full(n, a[-1])])  # noqa: E731
        gx = np.convolve(pad(gx), k, mode="same")[n:-n]
        gy = np.convolve(pad(gy), k, mode="same")[n:-n]
    nz = max(1, int(round(smooth_z / step)))          # altitude: much noisier -> longer window
    if nz > 1:
        kz = np.ones(nz) / nz
        gz = np.convolve(np.concatenate([np.full(nz, gz[0]), gz, np.full(nz, gz[-1])]), kz, mode="same")[nz:-nz]
    # final arc-length resampling
    d2 = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(gx), np.diff(gy)))])
    grid = np.arange(0.0, d2[-1], step)
    fx, fy, fz = np.interp(grid, d2, gx), np.interp(grid, d2, gy), np.interp(grid, d2, gz)
    station = np.zeros(len(grid), dtype=int)
    for sx, sy in stops:
        station[int(np.argmin(np.hypot(fx - sx, fy - sy)))] = 1
    return fx, fy, fz, station


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", required=True)
    ap.add_argument("--topic", default="/tram/gnss")
    ap.add_argument("--out", default="track.csv")
    ap.add_argument("--step", type=float, default=2.0)
    ap.add_argument("--smooth", type=float, default=15.0)
    ap.add_argument("--smooth-alt", type=float, default=250.0)
    ap.add_argument("--stop-speed", type=float, default=0.5)
    ap.add_argument("--min-dwell", type=float, default=10.0)
    args = ap.parse_args(argv)
    from tram_nav.core.geo import LocalFrame
    fx = read_fixes(args.bag, args.topic)
    geo = LocalFrame(fx[0, 1], fx[0, 2])
    xy = np.array([geo.to_xy(la, lo) for la, lo in fx[:, 1:3]])
    x, y, z, st = build_track(fx[:, 0], xy[:, 0], xy[:, 1], fx[:, 3] - fx[0, 3], args.step, args.smooth,
                              args.stop_speed, args.min_dwell, args.smooth_alt)
    with open(args.out, "w") as f:
        f.write("lat,lon,alt,station\n")
        for xi, yi, zi, si in zip(x, y, z, st):
            la, lo = geo.to_latlon(xi, yi)
            f.write(f"{la:.9f},{lo:.9f},{zi:.3f},{si}\n")
    print(f"track '{args.out}': {len(x)} points, {len(x) * args.step / 1000:.2f} km, {int(st.sum())} platforms "
          f"(from {len(fx)} GNSS fixes)")


if __name__ == "__main__":
    main()
