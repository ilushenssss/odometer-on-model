"""Integration test of the ROS 2 navigator node (skipped without ROS 2)."""
import math
import time

import pytest

rclpy = pytest.importorskip("rclpy")

from nav_msgs.msg import Odometry  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402
from std_msgs.msg import Float64  # noqa: E402


def test_navigator_node_publishes_odometry():
    from tram_nav.nodes.navigator_node import NavigatorNode
    rclpy.init()
    try:
        nav = NavigatorNode()
        io = rclpy.create_node("tram_nav_test_io")
        got = []
        io.create_subscription(Odometry, "odometry", lambda m: got.append(m), 10)
        pub_u = io.create_publisher(Float64, "controller_handle", 10)
        pub_w = io.create_publisher(JointState, "wheel_speeds", 10)
        ex = rclpy.executors.SingleThreadedExecutor()
        ex.add_node(nav)
        ex.add_node(io)
        r = 0.33
        t_end = time.time() + 4.0
        v = 0.0
        while time.time() < t_end:
            v = min(v + 0.02, 5.0)
            pub_u.publish(Float64(data=0.3))
            js = JointState()
            js.header.stamp = nav.get_clock().now().to_msg()
            js.name = ["axle_0", "axle_2", "axle_3"]
            js.velocity = [v / r, v / r, float("nan")]      # third sensor silent
            pub_w.publish(js)
            ex.spin_once(timeout_sec=0.02)
        assert len(got) > 50
        last = got[-1]
        assert last.header.frame_id == "map"
        assert last.twist.twist.linear.x > 1.0
        assert all(math.isfinite(c) for c in last.pose.covariance)
        assert nav.nav.state.sensor_status[2] == "DROPOUT"
    finally:
        rclpy.shutdown()


def test_ros_trajectory_end_to_end():
    """End-to-end trajectory test through ROS 2 topics in simulated time.

    The simulator drives /clock, controller_handle and wheel_speeds; the
    navigator node publishes nav_msgs/Odometry; the published 2-D trajectory
    is compared with the ground truth at the message stamps.
    """
    import numpy as np
    from rclpy.parameter import Parameter
    from rosgraph_msgs.msg import Clock
    from builtin_interfaces.msg import Time

    from tram_nav.core.params import ScenarioConfig
    from tram_nav.core.simulator import TramSimulator
    from tram_nav.core.track import generate_demo_track
    from tram_nav.core.trajectory import distance_to_track
    from tram_nav.nodes.navigator_node import NavigatorNode

    track = generate_demo_track()
    sim = TramSimulator(track, ScenarioConfig(duration=150))
    rclpy.init()
    try:
        nav = NavigatorNode(parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True),
                                                 Parameter("publish_tf", Parameter.Type.BOOL, False)])
        io = rclpy.create_node("tram_nav_traj_io")
        got = []
        io.create_subscription(Odometry, "odometry", lambda m: got.append(m), 50)
        pub_clock = io.create_publisher(Clock, "/clock", 10)
        pub_u = io.create_publisher(Float64, "controller_handle", 10)
        pub_w = io.create_publisher(JointState, "wheel_speeds", 10)
        ex = rclpy.executors.SingleThreadedExecutor()
        ex.add_node(nav)
        ex.add_node(io)
        truth = {}
        t_end = 120.0
        while sim.t < t_end:
            for _ in range(4):
                sim.step()
            ns = int(round(sim.t * 1e9))
            stamp = Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)
            truth[round(sim.t, 3)] = (sim.truth()["x"], sim.truth()["y"], sim.s)
            meas = sim.sample_sensors()
            pub_clock.publish(Clock(clock=stamp))
            pub_u.publish(Float64(data=float(sim.u)))
            js = JointState()
            js.header.stamp = stamp
            js.name = ["axle_0", "axle_2", "axle_3"]
            js.velocity = [float("nan") if m is None else float(m) for m in meas]
            pub_w.publish(js)
            # let DDS deliver /clock and the measurements before the next tick
            deadline = time.time() + 0.01
            while time.time() < deadline:
                ex.spin_once(timeout_sec=0.001)
        for _ in range(50):
            ex.spin_once(timeout_sec=0.01)

        # ~50 Hz in simulated time (a few timer ticks may merge when /clock
        # messages arrive in a burst - the test runs faster than real time)
        assert len(got) > 0.8 * t_end * 50
        errs, off = [], []
        for m in got:
            t = round(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, 3)
            if t not in truth:
                continue
            xt, yt, st = truth[t]
            x, y = m.pose.pose.position.x, m.pose.pose.position.y
            errs.append(math.hypot(x - xt, y - yt))
            off.append(distance_to_track(track, x, y, st))
            assert m.pose.covariance[0] > 0.0 and m.pose.covariance[7] > 0.0
        errs = np.asarray(errs)
        assert len(errs) > 0.5 * len(got)
        assert sim.s > 600.0                          # the tram really travelled (~900 m)
        assert np.sqrt(np.mean(errs ** 2)) < 5.0
        assert errs.max() < 12.0
        assert max(off) < 0.05                        # published points lie on the rails
    finally:
        rclpy.shutdown()


def test_navigator_custom_inputs_and_result_topics():
    """Int16 handle 0..1000 (neutral 500) and one Float64 topic per sensor in rpm;
    /result/velocity and /result/position must be published."""
    from geometry_msgs.msg import PoseStamped, TwistStamped
    from rclpy.parameter import Parameter
    from std_msgs.msg import Int16

    from tram_nav.nodes.navigator_node import NavigatorNode

    rclpy.init()
    try:
        P = Parameter
        nav = NavigatorNode(parameter_overrides=[
            P("mode", P.Type.STRING, "event"), P("publish_tf", P.Type.BOOL, False),
            P("handle_topic", P.Type.STRING, "/in/handle"), P("handle_type", P.Type.STRING, "int16"),
            P("handle_min", P.Type.DOUBLE, 0.0), P("handle_neutral", P.Type.DOUBLE, 500.0),
            P("handle_max", P.Type.DOUBLE, 1000.0),
            P("wheel_type", P.Type.STRING, "float64"),
            P("wheel_topics", P.Type.STRING_ARRAY, ["/in/w0", "/in/w1", "/in/w2"]),
            P("wheel_units", P.Type.STRING, "rpm")])
        io = rclpy.create_node("tram_nav_io2")
        vel, pos = [], []
        io.create_subscription(TwistStamped, "/result/velocity", vel.append, 50)
        io.create_subscription(PoseStamped, "/result/position", pos.append, 50)
        pu = io.create_publisher(Int16, "/in/handle", 10)
        pw = [io.create_publisher(Float64, f"/in/w{i}", 10) for i in range(3)]
        ex = rclpy.executors.SingleThreadedExecutor()
        ex.add_node(nav)
        ex.add_node(io)
        t0 = time.time()
        v = 0.0
        while time.time() < t0 + 7.0:
            tt = time.time() - t0
            v = min(1.0 * tt, 5.0)                         # 1 m/s^2, then constant speed
            pu.publish(Int16(data=800 if v < 5.0 else 550))  # u = +0.6, then +0.1
            for k, p in enumerate(pw):
                # realistic sensors are never bit-identical over time (else: STUCK)
                vi = v + 0.02 * math.sin(37.0 * time.time() + k)
                p.publish(Float64(data=vi / 0.33 * 60.0 / (2.0 * math.pi)))
            ex.spin_once(timeout_sec=0.02)
        for _ in range(20):
            ex.spin_once(timeout_sec=0.01)
        assert nav.nav.u_raw == pytest.approx(0.1)
        assert nav.nav.state.mode != "BLIND"
        assert len(vel) > 20 and len(pos) > 20
        assert vel[-1].twist.linear.x == pytest.approx(v, abs=0.6)
        assert pos[-1].header.frame_id == "map"
        assert nav.perf()["latency_ms"] > 0.0
    finally:
        rclpy.shutdown()


def test_demo_bag_and_track_from_gnss(tmp_path):
    pytest.importorskip("rosbag2_py")
    from tram_nav.core.track import TrackMap
    from tram_nav.tools import make_demo_bag, track_from_gnss

    bag = str(tmp_path / "bag")
    make_demo_bag.main(["--scenario", "nominal", "--duration", "120", "--out", bag])
    out = str(tmp_path / "track.csv")
    track_from_gnss.main(["--bag", bag, "--out", out])
    true_tr = TrackMap.from_csv(bag + "_track.csv")
    g = TrackMap.from_csv(out)
    assert g.geo is not None and g.length > 500.0
    # the GNSS-built line follows the true one within a few metres
    from tram_nav.core.trajectory import distance_to_track
    for s in range(50, int(g.length) - 50, 50):
        la, lo = g.geo.to_latlon(*g.pose(float(s))[:2])
        assert distance_to_track(true_tr, *true_tr.geo.to_xy(la, lo)) < 5.0
