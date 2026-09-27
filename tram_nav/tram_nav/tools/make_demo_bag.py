#!/usr/bin/env python3
"""Write a demo rosbag2 with the inputs the navigator needs and a GNSS reference.

    ros2 run tram_nav make_demo_bag --scenario combined --out demo_bag --duration 900

Topics (stamps = simulated time, starting at --t0 epoch seconds):
    /tram/controller_handle   std_msgs/Float64          handle u in [-1, 1]              50 Hz
    /tram/wheel_speeds        sensor_msgs/JointState     axle_0, axle_2, axle_3 [rad/s]   50 Hz
    /tram/gnss                sensor_msgs/NavSatFix      GNSS reference (noise, outages)  10 Hz
    /tram/ground_truth        nav_msgs/Odometry          exact simulator state            10 Hz

A line map with WGS-84 coordinates is written next to the bag
(``<out>_track.csv``) so the navigator can publish /result/position_geo.
"""
from __future__ import annotations

import argparse
import math
import os
import shutil

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="combined")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--duration", type=float, default=0.0, help="override scenario duration [s]")
    ap.add_argument("--out", default="tram_demo_bag")
    ap.add_argument("--storage", default="sqlite3")
    ap.add_argument("--t0", type=float, default=1_760_000_000.0, help="start time (epoch seconds)")
    ap.add_argument("--origin", type=float, nargs=2, default=[55.7558, 37.6173], help="lat lon of the map origin")
    ap.add_argument("--gnss-noise", type=float, default=1.5, help="GNSS horizontal noise std [m]")
    ap.add_argument("--gnss-rate", type=float, default=10.0)
    args = ap.parse_args(argv)

    import rosbag2_py
    from builtin_interfaces.msg import Time
    from nav_msgs.msg import Odometry
    from rclpy.serialization import serialize_message
    from sensor_msgs.msg import JointState, NavSatFix, NavSatStatus
    from std_msgs.msg import Float64

    from tram_nav.core.geo import LocalFrame
    from tram_nav.core.scenarios import get_scenario
    from tram_nav.core.simulator import TramSimulator
    from tram_nav.core.track import generate_demo_track

    sc = get_scenario(args.scenario, args.seed)
    if args.duration > 0:
        sc.duration = args.duration
    track = generate_demo_track()
    geo = LocalFrame(*args.origin)
    sim = TramSimulator(track, sc)
    rng = np.random.default_rng(args.seed + 1000)

    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=args.out, storage_id=args.storage),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    topics = {"/tram/controller_handle": "std_msgs/msg/Float64",
              "/tram/wheel_speeds": "sensor_msgs/msg/JointState",
              "/tram/gnss": "sensor_msgs/msg/NavSatFix",
              "/tram/ground_truth": "nav_msgs/msg/Odometry"}
    for name, typ in topics.items():
        writer.create_topic(rosbag2_py.TopicMetadata(name=name, type=typ, serialization_format="cdr"))

    def stamp(t):
        ns = int(round((args.t0 + t) * 1e9))
        return Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000), ns

    sub = int(round(1.0 / (50.0 * sim.dt)))
    gnss_every = int(round(50.0 / args.gnss_rate))
    k = 0
    gnss_err = np.zeros(2)
    n = 0
    while sim.t < sc.duration:
        for _ in range(sub):
            sim.step()
        k += 1
        st, ns = stamp(sim.t)
        meas = sim.sample_sensors()
        writer.write("/tram/controller_handle", serialize_message(Float64(data=float(sim.u))), ns)
        js = JointState()
        js.header.stamp = st
        js.name = ["axle_0", "axle_2", "axle_3"]
        js.velocity = [float("nan") if m is None else float(m) for m in meas]
        writer.write("/tram/wheel_speeds", serialize_message(js), ns)
        n += 2
        if k % gnss_every == 0:
            tr = sim.truth()
            # GNSS error: slowly varying (multipath) + white noise
            gnss_err = 0.98 * gnss_err + rng.normal(0.0, args.gnss_noise * 0.2, 2)
            x = tr["x"] + gnss_err[0] + rng.normal(0.0, args.gnss_noise * 0.5)
            y = tr["y"] + gnss_err[1] + rng.normal(0.0, args.gnss_noise * 0.5)
            lat, lon = geo.to_latlon(x, y)
            fix = NavSatFix()
            fix.header.stamp = st
            fix.header.frame_id = "gnss"
            fix.status.status = NavSatStatus.STATUS_FIX
            fix.status.service = NavSatStatus.SERVICE_GPS
            alt = 150.0 + float(np.interp(tr["s"], track.s, track.z)) + rng.normal(0.0, 2.0 * args.gnss_noise)
            fix.latitude, fix.longitude, fix.altitude = lat, lon, alt
            fix.position_covariance = [args.gnss_noise ** 2, 0.0, 0.0, 0.0, args.gnss_noise ** 2, 0.0,
                                       0.0, 0.0, 4.0 * args.gnss_noise ** 2]
            fix.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
            writer.write("/tram/gnss", serialize_message(fix), ns)
            od = Odometry()
            od.header.stamp = st
            od.header.frame_id = "map"
            od.child_frame_id = "base_link_truth"
            od.pose.pose.position.x, od.pose.pose.position.y = tr["x"], tr["y"]
            od.pose.pose.position.z = tr["s"]
            od.pose.pose.orientation.z = math.sin(0.5 * tr["yaw"])
            od.pose.pose.orientation.w = math.cos(0.5 * tr["yaw"])
            od.twist.twist.linear.x = tr["v"]
            writer.write("/tram/ground_truth", serialize_message(od), ns)
            n += 2
    del writer
    # map with WGS-84 coordinates (station flags on the 5 m grid)
    track_csv = args.out.rstrip("/") + "_track.csv"
    with open(track_csv, "w") as f:
        f.write("lat,lon,alt,station\n")
        stations = {int(round(v / 5.0)) for v in track.stations}
        for i, s in enumerate(np.arange(0.0, track.length + 1e-9, 5.0)):
            x, y, _ = track.pose(float(s))
            lat, lon = geo.to_latlon(x, y)
            z = float(np.interp(s, track.s, track.z))
            f.write(f"{lat:.9f},{lon:.9f},{z:.3f},{int(i in stations)}\n")
    print(f"bag '{args.out}': {n} messages, {sim.t:.0f} s, scenario '{args.scenario}', {sim.s:.0f} m; "
          f"map '{track_csv}'")


if __name__ == "__main__":
    main()
