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
