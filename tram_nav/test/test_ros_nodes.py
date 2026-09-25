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
