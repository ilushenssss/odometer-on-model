"""ROS 2 node: high-fidelity tram simulator (stand-in for the real vehicle).

Publishes exactly what the real tram offers the navigator - the driver's
controller handle and the odometry - plus the ground truth for evaluation.

    ``~/controller_handle``  std_msgs/Float64
    ``~/wheel_speeds``       sensor_msgs/JointState  (velocity [rad/s], NaN when a sensor is silent)
    ``~/ground_truth``       nav_msgs/Odometry
    ``~/ground_truth/speed`` std_msgs/Float64
    ``~/weather``            std_msgs/String
    ``/clock``               rosgraph_msgs/Clock     (when ``publish_clock`` - for faster than real time)
"""
from __future__ import annotations

import math
import os

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, String

from ..core.params import SensorParams
from ..core.scenarios import SCENARIOS, get_scenario
from ..core.simulator import TramSimulator
from ..core.track import TrackMap, generate_demo_track


class SimulatorNode(Node):
    def __init__(self):
        super().__init__("tram_simulator")
        self.declare_parameter("scenario", "nominal")
        self.declare_parameter("seed", 1)
        self.declare_parameter("track_csv", "")
        self.declare_parameter("sim_dt", 0.005)
        self.declare_parameter("sensor_rate", 50.0)
        self.declare_parameter("real_time_factor", 1.0)
        self.declare_parameter("publish_clock", False)
        self.declare_parameter("wheel_joint_names", ["axle_0", "axle_2", "axle_3"])
        self.declare_parameter("external_handle", False)
        self.declare_parameter("shutdown_at_end", True)

        name = self.get_parameter("scenario").value
        sc = get_scenario(name, int(self.get_parameter("seed").value))
        track_csv = self.get_parameter("track_csv").value
        track = TrackMap.from_csv(track_csv) if track_csv and os.path.isfile(track_csv) else generate_demo_track()
        rate = float(self.get_parameter("sensor_rate").value)
        self.sim = TramSimulator(track, sc, sensors=SensorParams(rate=rate), dt=float(self.get_parameter("sim_dt").value))
        self.names = list(self.get_parameter("wheel_joint_names").value)
        self.sub = max(1, int(round(1.0 / (rate * self.sim.dt))))
        self.rtf = float(self.get_parameter("real_time_factor").value)
        self.publish_clock = bool(self.get_parameter("publish_clock").value)
        self.shutdown_at_end = bool(self.get_parameter("shutdown_at_end").value)
        self.u_ext = None
        if self.get_parameter("external_handle").value:
            self.create_subscription(Float64, "external_handle", lambda m: setattr(self, "u_ext", m.data), 10)

        self.pub_handle = self.create_publisher(Float64, "controller_handle", 20)
        self.pub_wheels = self.create_publisher(JointState, "wheel_speeds", 20)
        self.pub_truth = self.create_publisher(Odometry, "ground_truth", 10)
        self.pub_truth_v = self.create_publisher(Float64, "ground_truth/speed", 10)
        self.pub_weather = self.create_publisher(String, "weather", 10)
        self.pub_clock = self.create_publisher(Clock, "/clock", 10) if self.publish_clock else None
        self.t0_ns = 0 if self.publish_clock else self.get_clock().now().nanoseconds
        self.done = False
        self.create_timer(1.0 / (rate * self.rtf), self._on_timer)
        self.get_logger().info(
            f"scenario '{name}' ({sc.duration:.0f} s, track {track.length:.0f} m) at x{self.rtf:g} real time; "
            f"available: {', '.join(SCENARIOS)}")

    def _stamp(self):
        ns = self.t0_ns + int(self.sim.t * 1e9)
        from builtin_interfaces.msg import Time
        return Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)

    def _on_timer(self):
        if self.done:
            return
        for _ in range(self.sub):
            self.sim.step(self.u_ext)
        stamp = self._stamp()
        if self.pub_clock is not None:
            self.pub_clock.publish(Clock(clock=stamp))
        meas = self.sim.sample_sensors()
        self.pub_handle.publish(Float64(data=float(self.sim.u)))
        js = JointState()
        js.header.stamp = stamp
        js.name = self.names[:len(meas)]
        js.velocity = [float("nan") if m is None else float(m) for m in meas]
        self.pub_wheels.publish(js)

        tr = self.sim.truth()
        od = Odometry()
        od.header.stamp = stamp
        od.header.frame_id = "map"
        od.child_frame_id = "base_link_truth"
        od.pose.pose.position.x, od.pose.pose.position.y = tr["x"], tr["y"]
        od.pose.pose.orientation.z = math.sin(0.5 * tr["yaw"])
        od.pose.pose.orientation.w = math.cos(0.5 * tr["yaw"])
        od.pose.pose.position.z = tr["s"]    # along-track distance (convenience)
        od.twist.twist.linear.x = tr["v"]
        od.twist.twist.linear.z = tr["a"]
        self.pub_truth.publish(od)
        self.pub_truth_v.publish(Float64(data=tr["v"]))
        if int(self.sim.t * 50) % 50 == 0:
            self.pub_weather.publish(String(data=tr["weather"]))
        if self.sim.t >= self.sim.sc.duration:
            self.done = True
            self.get_logger().info(f"scenario finished at t = {self.sim.t:.0f} s, s = {self.sim.s:.0f} m")
            if self.shutdown_at_end:
                raise SystemExit


def main(args=None):
    rclpy.init(args=args)
    node = SimulatorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
