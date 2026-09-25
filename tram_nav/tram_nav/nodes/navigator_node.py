"""ROS 2 node: GNSS-free tram navigator.

Subscribes
    ``~/controller_handle``  std_msgs/Float64          handle position u in [-1, 1]
    ``~/wheel_speeds``       sensor_msgs/JointState    angular wheel speeds [rad/s]
                                                       (``velocity[]``; NaN = no data)
    ``~/set_position``       std_msgs/Float64          re-initialise the along-track position [m]

Publishes (at ``rate`` Hz)
    ``~/odometry``   nav_msgs/Odometry                   pose on the track map + speed, with covariances
    ``~/pose``       geometry_msgs/PoseWithCovarianceStamped
    ``~/speed``      std_msgs/Float64                    estimated speed [m/s]
    ``~/distance``   std_msgs/Float64                    along-track distance [m]
    ``~/state``      std_msgs/Float64MultiArray          full internal state (see STATE_FIELDS)
    ``/diagnostics`` diagnostic_msgs/DiagnosticArray     mode, sensor health, timing (1 Hz)
    TF ``map -> base_link`` (optional)

Topic names are relative to the node namespace; the launch files remap them
to ``/tram/...``.
"""
from __future__ import annotations

import math
import os
import time

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray, MultiArrayDimension

from ..core.estimator import MODE_CODES, TramNavigator
from ..core.params import NavigatorParams, TramParams, params_to_dict, update_from_dict
from ..core.track import TrackMap, generate_demo_track

STATE_FIELDS = ["s", "v", "a", "sigma_s", "sigma_v", "eta", "mass_est", "disturbance", "mu_hat",
                "rbf_residual", "odo_scale", "mode", "blind_time", "n_used", "n_fixes", "x", "y", "yaw"]


def yaw_to_quat(yaw: float):
    return 0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)


def declare_dataclass(node: Node, prefix: str, obj):
    """Declare every field of a parameter dataclass as a ROS parameter and
    return a copy of ``obj`` with the values set by the user."""
    vals = {}
    for k, v in params_to_dict(obj).items():
        default = list(v) if isinstance(v, tuple) else v
        if isinstance(default, list) and default and isinstance(default[0], int):
            default = [int(a) for a in default]
        node.declare_parameter(f"{prefix}.{k}", default)
        vals[k] = node.get_parameter(f"{prefix}.{k}").value
    return update_from_dict(obj, vals)


class NavigatorNode(Node):
    def __init__(self):
        super().__init__("tram_navigator")
        self.declare_parameter("rate", 50.0)
        self.declare_parameter("track_csv", "")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("child_frame_id", "base_link")
        self.declare_parameter("publish_tf", True)
        self.declare_parameter("initial_position", 0.0)
        self.declare_parameter("wheel_joint_names", ["axle_0", "axle_2", "axle_3"])
        self.declare_parameter("sensor_powered", [True, True, False])
        self.declare_parameter("time_budget_ms", 5.0)

        tram = declare_dataclass(self, "tram", TramParams())
        nav = declare_dataclass(self, "nav", NavigatorParams())
        nav = update_from_dict(nav, {"rate": self.get_parameter("rate").value})

        track_csv = self.get_parameter("track_csv").value
        if track_csv and os.path.isfile(track_csv):
            track = TrackMap.from_csv(track_csv)
            self.get_logger().info(f"track map '{track_csv}': {track.length:.0f} m, "
                                   f"{len(track.stations)} stations")
        else:
            track = generate_demo_track()
            self.get_logger().warn("no track_csv given - using the built-in demo track")
        self.names = list(self.get_parameter("wheel_joint_names").value)
        powered = list(self.get_parameter("sensor_powered").value)
        if len(powered) != len(self.names):
            powered = [True] * len(self.names)
        self.nav = TramNavigator(tram, nav, track, len(self.names), powered,
                                 s0=float(self.get_parameter("initial_position").value))
        self.frame_id = self.get_parameter("frame_id").value
        self.child_frame_id = self.get_parameter("child_frame_id").value
        self.budget = float(self.get_parameter("time_budget_ms").value) * 1e-3

        qos = QoSProfile(depth=20, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(Float64, "controller_handle", self._on_handle, qos)
        self.create_subscription(JointState, "wheel_speeds", self._on_wheels, qos)
        self.create_subscription(Float64, "set_position", self._on_set_position, 10)
        self.pub_odom = self.create_publisher(Odometry, "odometry", 10)
        self.pub_pose = self.create_publisher(PoseWithCovarianceStamped, "pose", 10)
        self.pub_speed = self.create_publisher(Float64, "speed", 10)
        self.pub_dist = self.create_publisher(Float64, "distance", 10)
        self.pub_state = self.create_publisher(Float64MultiArray, "state", 10)
        self.pub_diag = self.create_publisher(DiagnosticArray, "/diagnostics", 10)
        self.tf_broadcaster = None
        if self.get_parameter("publish_tf").value:
            from tf2_ros import TransformBroadcaster
            self.tf_broadcaster = TransformBroadcaster(self)

        self._step_times = []
        self._overruns = 0
        self._last_diag = -1e9
        self.create_timer(1.0 / nav.rate, self._on_timer)
        self.get_logger().info(f"tram navigator running at {nav.rate:.0f} Hz with {len(self.names)} "
                               f"odometry channels {self.names}")

    # ------------------------------------------------------------------ callbacks
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_handle(self, msg: Float64):
        self.nav.set_handle(msg.data)

    def _on_wheels(self, msg: JointState):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if stamp <= 0.0:
            stamp = self._now()
        names = list(msg.name) if msg.name else self.names[:len(msg.velocity)]
        for name, omega in zip(names, msg.velocity):
            if name in self.names and omega == omega:  # skip NaN = no data
                self.nav.set_wheel_speed(self.names.index(name), stamp, omega)

    def _on_set_position(self, msg: Float64):
        self.nav.reset(s0=msg.data, v0=self.nav.x[1])
        self.get_logger().info(f"position re-initialised to s = {msg.data:.1f} m")

    def _on_timer(self):
        t = self._now()
        t0 = time.perf_counter()
        st = self.nav.step(t)
        dt = time.perf_counter() - t0
        self._step_times.append(dt)
        if dt > self.budget:
            self._overruns += 1
        self._publish(st)
        if t - self._last_diag >= 1.0:
            self._last_diag = t
            self._publish_diag(st)

    # ------------------------------------------------------------------ output
    def _publish(self, st):
        stamp = self.get_clock().now().to_msg()
        qx, qy, qz, qw = yaw_to_quat(st.yaw)
        c, s_ = math.cos(st.yaw), math.sin(st.yaw)
        var_s = st.sigma_s ** 2
        var_lat = 0.05 ** 2     # rail-bound: lateral error only from the map
        cov = [0.0] * 36
        cov[0] = var_s * c * c + var_lat * s_ * s_
        cov[1] = cov[6] = (var_s - var_lat) * c * s_
        cov[7] = var_s * s_ * s_ + var_lat * c * c
        cov[14] = 0.1 ** 2
        cov[21] = cov[28] = 0.01 ** 2
        cov[35] = 0.02 ** 2

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.frame_id
        odom.child_frame_id = self.child_frame_id
        odom.pose.pose.position.x = st.x
        odom.pose.pose.position.y = st.y
        odom.pose.pose.orientation.x, odom.pose.pose.orientation.y = qx, qy
        odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = qz, qw
        odom.pose.covariance = cov
        odom.twist.twist.linear.x = st.v
        tcov = [0.0] * 36
        tcov[0] = st.sigma_v ** 2
        tcov[7] = tcov[14] = 1e-4
        tcov[21] = tcov[28] = tcov[35] = 1e-4
        odom.twist.covariance = tcov
        self.pub_odom.publish(odom)

        pose = PoseWithCovarianceStamped()
        pose.header = odom.header
        pose.pose = odom.pose
        self.pub_pose.publish(pose)
        self.pub_speed.publish(Float64(data=st.v))
        self.pub_dist.publish(Float64(data=st.s))

        arr = Float64MultiArray()
        arr.layout.dim = [MultiArrayDimension(label=",".join(STATE_FIELDS), size=len(STATE_FIELDS), stride=1)]
        arr.data = [float(v) for v in (st.s, st.v, st.a, st.sigma_s, st.sigma_v, st.eta, st.mass_est, st.dist,
                                       st.mu_hat, st.rbf, st.scale, MODE_CODES[st.mode], st.blind_time,
                                       st.n_used, st.n_fixes, st.x, st.y, st.yaw)]
        self.pub_state.publish(arr)

        if self.tf_broadcaster is not None:
            tf = TransformStamped()
            tf.header = odom.header
            tf.child_frame_id = self.child_frame_id
            tf.transform.translation.x, tf.transform.translation.y = st.x, st.y
            tf.transform.rotation.x, tf.transform.rotation.y = qx, qy
            tf.transform.rotation.z, tf.transform.rotation.w = qz, qw
            self.tf_broadcaster.sendTransform(tf)

    def _publish_diag(self, st):
        arr = DiagnosticArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        level = {"NOMINAL": DiagnosticStatus.OK, "STANDSTILL": DiagnosticStatus.OK,
                 "DEGRADED": DiagnosticStatus.WARN, "BLIND": DiagnosticStatus.ERROR}[st.mode]
        times = self._step_times[-500:] or [0.0]
        mean_us = 1e6 * sum(times) / len(times)
        max_us = 1e6 * max(times)
        main = DiagnosticStatus(name="tram_nav: navigator", hardware_id="tram", level=level,
                                message=f"{st.mode}: v={st.v:.2f} m/s, s={st.s:.1f} m (+-{st.sigma_s:.1f} m)")
        main.values = [KeyValue(key=k, value=v) for k, v in [
            ("mode", st.mode), ("speed_mps", f"{st.v:.3f}"), ("distance_m", f"{st.s:.2f}"),
            ("sigma_s_m", f"{st.sigma_s:.2f}"), ("sigma_v_mps", f"{st.sigma_v:.3f}"),
            ("blind_time_s", f"{st.blind_time:.1f}"), ("mass_est_kg", f"{st.mass_est:.0f}"),
            ("mu_hat", f"{st.mu_hat:.3f}"), ("odo_scale", f"{st.scale:.4f}"),
            ("station_fixes", str(st.n_fixes)), ("slip", str(st.slip)),
            ("step_mean_us", f"{mean_us:.0f}"), ("step_max_us", f"{max_us:.0f}"),
            ("budget_overruns", str(self._overruns))]]
        arr.status.append(main)
        for name, status, calib in zip(self.names, st.sensor_status, st.sensor_calib):
            lvl = DiagnosticStatus.OK if status == "OK" else (
                DiagnosticStatus.WARN if status in ("SLIP", "SUSPECT") else DiagnosticStatus.ERROR)
            d = DiagnosticStatus(name=f"tram_nav: odometry {name}", hardware_id=name, level=lvl, message=status)
            d.values = [KeyValue(key="calibration", value=f"{calib:.4f}")]
            arr.status.append(d)
        self.pub_diag.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = NavigatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
