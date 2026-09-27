"""ROS 2 node: GNSS-free tram navigator.

Inputs (topic names, message types and units are parameters - see
``config/navigator.yaml`` and ``docs/JURY_GUIDE.md``):

    handle_topic   controller handle position    std_msgs/Float64 | Float32 | Int32 | Int16 | Int8 | UInt8
                   (raw value mapped to u in [-1, 1] with handle_min / handle_neutral / handle_max)
    wheel_topic    wheel speeds                  sensor_msgs/JointState (velocity[]) |
                                                 std_msgs/Float64MultiArray | Float32MultiArray (data[])
    wheel_topics   ... or one topic per sensor   std_msgs/Float64 | Float32
                   (units: rad/s | rpm | m/s | km/h; NaN = no data)
    set_position   std_msgs/Float64              re-initialise the along-track position [m]

Outputs:

    /result/velocity      geometry_msgs/TwistStamped   twist.linear.x = speed [m/s], angular.z = yaw rate
                          (or std_msgs/Float64 / geometry_msgs/Vector3Stamped - result_velocity_type)
    /result/position      geometry_msgs/PoseStamped    x, y [m] in the map frame (ENU), orientation = heading
                          (or PointStamped / PoseWithCovarianceStamped - result_position_type)
    /result/position_geo  sensor_msgs/NavSatFix        WGS-84 position (when the map has a geo origin)
    /result/perf          std_msgs/Float64MultiArray   [step_ms, latency_ms, rate_hz, cpu_pct, rss_mb]
    ~/odometry            nav_msgs/Odometry            pose + speed with covariances
    ~/pose, ~/speed, ~/distance, ~/state                (see README)
    /diagnostics          diagnostic_msgs/DiagnosticArray  mode, sensor health, timing, latency, CPU, memory
    TF map -> base_link   (optional)

Modes: ``timer`` - estimate at a fixed ``rate``; ``event`` - estimate on every
wheel-speed message (lowest latency, output stamped with the input stamp).
"""
from __future__ import annotations

import collections
import math
import os
import time

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import (PointStamped, PoseStamped, PoseWithCovarianceStamped, TransformStamped,
                               TwistStamped, Vector3Stamped)
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState, NavSatFix, NavSatStatus
from std_msgs import msg as std
from std_msgs.msg import Float64, Float64MultiArray, MultiArrayDimension

from ..core.estimator import MODE_CODES, TramNavigator
from ..core.geo import LocalFrame
from ..core.params import NavigatorParams, TramParams, params_to_dict, update_from_dict
from ..core.track import TrackMap, generate_demo_track

STATE_FIELDS = ["s", "v", "a", "sigma_s", "sigma_v", "eta", "mass_est", "disturbance", "mu_hat",
                "rbf_residual", "odo_scale", "mode", "blind_time", "n_used", "n_fixes", "x", "y", "yaw"]
PERF_FIELDS = ["step_ms", "latency_ms", "rate_hz", "cpu_pct", "rss_mb"]
SCALAR_TYPES = {"float64": std.Float64, "float32": std.Float32, "int32": std.Int32, "int16": std.Int16,
                "int8": std.Int8, "uint8": std.UInt8, "uint16": std.UInt16, "int64": std.Int64}
UNITS = {"rad_s": None, "rpm": 2.0 * math.pi / 60.0, "m_s": "linear", "km_h": "linear_kmh"}


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


class _ProcStats:
    """CPU (% of one core) and resident memory of this process."""

    def __init__(self):
        self.t0 = time.monotonic()
        self.c0 = sum(os.times()[:2])

    def sample(self):
        t, c = time.monotonic(), sum(os.times()[:2])
        cpu = 100.0 * (c - self.c0) / max(t - self.t0, 1e-6)
        self.t0, self.c0 = t, c
        rss = 0.0
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss = float(line.split()[1]) / 1024.0
                        break
        except OSError:
            pass
        return cpu, rss


class NavigatorNode(Node):
    def __init__(self, **kwargs):
        super().__init__("tram_navigator", **kwargs)
        P = self.declare_parameter
        P("mode", "timer")                      # timer | event
        P("rate", 50.0)
        P("track_csv", "")
        P("geo_origin", [0.0, 0.0])             # [lat, lon] of the map origin; [0, 0] = from the map / none
        P("frame_id", "map")
        P("child_frame_id", "base_link")
        P("publish_tf", True)
        P("initial_position", 0.0)              # along-track distance s0 [m]
        P("initial_xy", [0.0, 0.0, 0.0])        # [x, y, enable] - projected onto the line
        P("initial_latlon", [0.0, 0.0])         # [lat, lon]; [0, 0] = not used
        # --- inputs
        P("handle_topic", "controller_handle")
        P("handle_type", "float64")
        P("handle_min", -1.0)                   # raw value of full brake
        P("handle_neutral", 0.0)                # raw value of neutral (coasting)
        P("handle_max", 1.0)                    # raw value of full traction
        P("wheel_topic", "wheel_speeds")
        P("wheel_type", "joint_state")          # joint_state | float64_array | float32_array | float64 | float32
        P("wheel_topics", ["wheel_0", "wheel_1", "wheel_2"])   # for scalar per-sensor topics
        P("wheel_units", "rad_s")               # rad_s | rpm | m_s | km_h
        P("wheel_joint_names", ["axle_0", "axle_2", "axle_3"])
        P("sensor_powered", [True, True, False])
        P("use_input_stamps", True)             # take measurement time from header.stamp when present
        # --- outputs
        P("result_velocity_topic", "/result/velocity")
        P("result_position_topic", "/result/position")
        P("result_velocity_type", "twist")      # twist | float64 | vector3
        P("result_position_type", "pose")       # pose | point | pose_cov
        P("time_budget_ms", 5.0)

        tram = declare_dataclass(self, "tram", TramParams())
        nav = declare_dataclass(self, "nav", NavigatorParams())
        self.mode = self.get_parameter("mode").value
        nav = update_from_dict(nav, {"rate": self.get_parameter("rate").value})
        self.tram = tram

        track_csv = self.get_parameter("track_csv").value
        if track_csv and os.path.isfile(track_csv):
            track = TrackMap.from_csv(track_csv)
            self.get_logger().info(f"track map '{track_csv}': {track.length:.0f} m, {len(track.stations)} stations")
        else:
            track = generate_demo_track()
            self.get_logger().warn("no track_csv given - using the built-in demo track")
        go = list(self.get_parameter("geo_origin").value)
        if abs(go[0]) > 1e-9 or abs(go[1]) > 1e-9:
            track.geo = LocalFrame(go[0], go[1])
        self.geo = track.geo

        self.names = list(self.get_parameter("wheel_joint_names").value)
        self.wheel_type = self.get_parameter("wheel_type").value
        if self.wheel_type in ("float64", "float32"):
            self.names = list(self.get_parameter("wheel_topics").value)
        powered = list(self.get_parameter("sensor_powered").value)
        if len(powered) != len(self.names):
            self.get_logger().warn("sensor_powered does not match the number of wheel channels - "
                                   "assuming all axles motored")
            powered = [True] * len(self.names)
        s0 = self._initial_position(track)
        self.nav = TramNavigator(tram, nav, track, len(self.names), powered, s0=s0)
        self.frame_id = self.get_parameter("frame_id").value
        self.child_frame_id = self.get_parameter("child_frame_id").value
        self.budget = float(self.get_parameter("time_budget_ms").value) * 1e-3
        self.use_stamps = bool(self.get_parameter("use_input_stamps").value)
        hmin, hneu, hmax = (float(self.get_parameter(k).value) for k in ("handle_min", "handle_neutral",
                                                                        "handle_max"))
        self.handle_map = (hmin, hneu, hmax)
        units = self.get_parameter("wheel_units").value
        if units not in UNITS:
            raise ValueError(f"wheel_units must be one of {list(UNITS)}")
        self.units = units

        qos = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        htype = self.get_parameter("handle_type").value
        self.create_subscription(SCALAR_TYPES[htype], self.get_parameter("handle_topic").value,
                                 self._on_handle, qos)
        wt = self.wheel_type
        if wt == "joint_state":
            self.create_subscription(JointState, self.get_parameter("wheel_topic").value, self._on_joint_state, qos)
        elif wt in ("float64_array", "float32_array"):
            mt = std.Float64MultiArray if wt == "float64_array" else std.Float32MultiArray
            self.create_subscription(mt, self.get_parameter("wheel_topic").value, self._on_array, qos)
        elif wt in ("float64", "float32"):
            for i, tp in enumerate(self.names):
                self.create_subscription(SCALAR_TYPES[wt], tp, lambda m, i=i: self._on_scalar(i, m), qos)
        else:
            raise ValueError("wheel_type must be joint_state | float64_array | float32_array | float64 | float32")
        self.create_subscription(Float64, "set_position", self._on_set_position, 10)

        self.pub_odom = self.create_publisher(Odometry, "odometry", 10)
        self.pub_pose = self.create_publisher(PoseWithCovarianceStamped, "pose", 10)
        self.pub_speed = self.create_publisher(Float64, "speed", 10)
        self.pub_dist = self.create_publisher(Float64, "distance", 10)
        self.pub_state = self.create_publisher(Float64MultiArray, "state", 10)
        self.pub_diag = self.create_publisher(DiagnosticArray, "/diagnostics", 10)
        self.pub_perf = self.create_publisher(Float64MultiArray, "/result/perf", 10)
        vt = self.get_parameter("result_velocity_type").value
        pt = self.get_parameter("result_position_type").value
        self.res_vel_type, self.res_pos_type = vt, pt
        self.pub_res_vel = self.create_publisher(
            {"twist": TwistStamped, "float64": Float64, "vector3": Vector3Stamped}[vt],
            self.get_parameter("result_velocity_topic").value, 10)
        self.pub_res_pos = self.create_publisher(
            {"pose": PoseStamped, "point": PointStamped, "pose_cov": PoseWithCovarianceStamped}[pt],
            self.get_parameter("result_position_topic").value, 10)
        self.pub_res_geo = self.create_publisher(NavSatFix, "/result/position_geo", 10) if self.geo else None
        self.tf_broadcaster = None
        if self.get_parameter("publish_tf").value:
            from tf2_ros import TransformBroadcaster
            self.tf_broadcaster = TransformBroadcaster(self)

        # performance bookkeeping
        self._step_times = collections.deque(maxlen=2000)
        self._latencies = collections.deque(maxlen=2000)
        self._pub_times = collections.deque(maxlen=500)
        self._pending = []                       # monotonic arrival times of unconsumed measurements
        self._overruns = 0
        self._n_out = 0
        self._last_diag = -1e9
        self._last_perf = 0.0
        self._proc = _ProcStats()
        self._cpu, self._rss = 0.0, 0.0
        self._in_stamp = None
        if self.mode == "timer":
            self.create_timer(1.0 / nav.rate, self._on_timer)
        elif self.mode != "event":
            raise ValueError("mode must be 'timer' or 'event'")
        self.get_logger().info(
            f"tram navigator: mode={self.mode}, rate={nav.rate:.0f} Hz, wheels={wt} {self.names} [{units}], "
            f"handle={htype} on '{self.get_parameter('handle_topic').value}', s0={s0:.1f} m, "
            f"outputs {self.get_parameter('result_velocity_topic').value} ({vt}), "
            f"{self.get_parameter('result_position_topic').value} ({pt})")

    # ------------------------------------------------------------------ setup helpers
    def _initial_position(self, track: TrackMap) -> float:
        ll = list(self.get_parameter("initial_latlon").value)
        xy = list(self.get_parameter("initial_xy").value)
        if abs(ll[0]) > 1e-9 or abs(ll[1]) > 1e-9:
            if track.geo is None:
                raise ValueError("initial_latlon needs a map with lat/lon or the geo_origin parameter")
            x, y = track.geo.to_xy(ll[0], ll[1])
            return track.project(x, y)
        if len(xy) >= 3 and xy[2] > 0.5:
            return track.project(xy[0], xy[1])
        return float(self.get_parameter("initial_position").value)

    # ------------------------------------------------------------------ input callbacks
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _stamp_of(self, msg) -> float:
        if self.use_stamps and hasattr(msg, "header"):
            st = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if st > 0.0:
                return st
        return self._now()

    def _map_handle(self, raw: float) -> float:
        lo, neu, hi = self.handle_map
        if raw >= neu:
            return (raw - neu) / (hi - neu) if hi != neu else 0.0
        return -(neu - raw) / (neu - lo) if neu != lo else 0.0

    def _omega(self, val: float) -> float:
        u = UNITS[self.units]
        if u is None:
            return val
        if u == "linear":
            return val / self.tram.wheel_radius
        if u == "linear_kmh":
            return val / 3.6 / self.tram.wheel_radius
        return val * u

    def _on_handle(self, msg):
        self.nav.set_handle(self._map_handle(float(msg.data)))

    def _measurement(self, stamp: float, values):
        arrival = time.monotonic()
        for i, val in values:
            if 0 <= i < len(self.names) and val == val:            # skip NaN = no data
                self.nav.set_wheel_speed(i, stamp, self._omega(float(val)))
        self._pending.append(arrival)
        self._in_stamp = stamp
        if self.mode == "event":
            self._run_step(stamp)

    def _on_joint_state(self, msg: JointState):
        names = list(msg.name) if msg.name else self.names[:len(msg.velocity)]
        vals = [(self.names.index(n), v) for n, v in zip(names, msg.velocity) if n in self.names]
        self._measurement(self._stamp_of(msg), vals)

    def _on_array(self, msg):
        self._measurement(self._now(), list(enumerate(msg.data)))

    def _on_scalar(self, i: int, msg):
        self._measurement(self._now(), [(i, msg.data)])

    def _on_set_position(self, msg: Float64):
        self.nav.reset(s0=msg.data, v0=self.nav.x[1])
        self.get_logger().info(f"position re-initialised to s = {msg.data:.1f} m")

    # ------------------------------------------------------------------ estimation
    def _on_timer(self):
        self._run_step(self._now())

    def _run_step(self, t: float):
        t0 = time.perf_counter()
        st = self.nav.step(t)
        dt = time.perf_counter() - t0
        self._step_times.append(dt)
        if dt > self.budget:
            self._overruns += 1
        self._publish(st, t)
        now_m = time.monotonic()
        for a in self._pending:
            self._latencies.append(now_m - a)
        self._pending.clear()
        self._pub_times.append(now_m)
        self._n_out += 1
        if now_m - self._last_perf >= 1.0:
            self._last_perf = now_m
            self._cpu, self._rss = self._proc.sample()
            self._publish_perf()
        if t - self._last_diag >= 1.0 or t < self._last_diag:
            self._last_diag = t
            self._publish_diag(st)

    def perf(self):
        st = list(self._step_times) or [0.0]
        lat = list(self._latencies) or [0.0]
        pt = list(self._pub_times)
        rate = (len(pt) - 1) / (pt[-1] - pt[0]) if len(pt) > 2 and pt[-1] > pt[0] else 0.0
        return {"step_ms": 1e3 * sum(st) / len(st), "step_max_ms": 1e3 * max(st),
                "latency_ms": 1e3 * sum(lat) / len(lat), "latency_max_ms": 1e3 * max(lat),
                "rate_hz": rate, "cpu_pct": self._cpu, "rss_mb": self._rss}

    # ------------------------------------------------------------------ output
    def _stamp_msg(self, t: float):
        from builtin_interfaces.msg import Time
        ns = int(round(t * 1e9))
        return Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)

    def _publish(self, st, t: float):
        stamp = self._stamp_msg(t) if self.mode == "event" else self.get_clock().now().to_msg()
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
        yaw_rate = st.v * self.nav.track.curvature_at(st.s) if self.nav.track is not None else 0.0

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
        odom.twist.twist.angular.z = yaw_rate
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

        # ---- /result/velocity
        if self.res_vel_type == "twist":
            m = TwistStamped()
            m.header.stamp = stamp                         # own header (no aliasing of odom.header)
            m.header.frame_id = self.child_frame_id
            m.twist.linear.x = st.v
            m.twist.angular.z = yaw_rate
        elif self.res_vel_type == "vector3":
            m = Vector3Stamped()
            m.header = odom.header
            m.vector.x, m.vector.y = st.v * c, st.v * s_          # ENU velocity in the map frame
        else:
            m = Float64(data=st.v)
        self.pub_res_vel.publish(m)
        # ---- /result/position
        if self.res_pos_type == "pose":
            m = PoseStamped()
            m.header = odom.header
            m.pose = odom.pose.pose
        elif self.res_pos_type == "point":
            m = PointStamped()
            m.header = odom.header
            m.point.x, m.point.y = st.x, st.y
        else:
            m = pose
        self.pub_res_pos.publish(m)
        if self.pub_res_geo is not None:
            lat, lon = self.geo.to_latlon(st.x, st.y)
            fix = NavSatFix()
            fix.header = odom.header
            fix.status.status = NavSatStatus.STATUS_NO_FIX     # not a satellite fix: dead reckoning
            fix.latitude, fix.longitude = lat, lon
            fix.position_covariance = [cov[7], cov[6], 0.0, cov[1], cov[0], 0.0, 0.0, 0.0, 0.01]
            fix.position_covariance_type = NavSatFix.COVARIANCE_TYPE_KNOWN
            self.pub_res_geo.publish(fix)

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

    def _publish_perf(self):
        p = self.perf()
        arr = Float64MultiArray()
        arr.layout.dim = [MultiArrayDimension(label=",".join(PERF_FIELDS), size=len(PERF_FIELDS), stride=1)]
        arr.data = [p["step_ms"], p["latency_ms"], p["rate_hz"], p["cpu_pct"], p["rss_mb"]]
        self.pub_perf.publish(arr)

    def _publish_diag(self, st):
        arr = DiagnosticArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        level = {"NOMINAL": DiagnosticStatus.OK, "STANDSTILL": DiagnosticStatus.OK,
                 "DEGRADED": DiagnosticStatus.WARN, "BLIND": DiagnosticStatus.ERROR}[st.mode]
        p = self.perf()
        main = DiagnosticStatus(name="tram_nav: navigator", hardware_id="tram", level=level,
                                message=f"{st.mode}: v={st.v:.2f} m/s, s={st.s:.1f} m (+-{st.sigma_s:.1f} m)")
        main.values = [KeyValue(key=k, value=v) for k, v in [
            ("mode", st.mode), ("speed_mps", f"{st.v:.3f}"), ("distance_m", f"{st.s:.2f}"),
            ("x_m", f"{st.x:.2f}"), ("y_m", f"{st.y:.2f}"),
            ("sigma_s_m", f"{st.sigma_s:.2f}"), ("sigma_v_mps", f"{st.sigma_v:.3f}"),
            ("blind_time_s", f"{st.blind_time:.1f}"), ("mass_est_kg", f"{st.mass_est:.0f}"),
            ("mu_hat", f"{st.mu_hat:.3f}"), ("odo_scale", f"{st.scale:.4f}"),
            ("station_fixes", str(st.n_fixes)), ("slip", str(st.slip)),
            ("step_mean_ms", f"{p['step_ms']:.3f}"), ("step_max_ms", f"{p['step_max_ms']:.3f}"),
            ("latency_mean_ms", f"{p['latency_ms']:.2f}"), ("latency_max_ms", f"{p['latency_max_ms']:.2f}"),
            ("output_rate_hz", f"{p['rate_hz']:.1f}"), ("cpu_pct", f"{p['cpu_pct']:.1f}"),
            ("rss_mb", f"{p['rss_mb']:.1f}"), ("budget_overruns", str(self._overruns)),
            ("outputs", str(self._n_out))]]
        arr.status.append(main)
        for name, status, calib in zip(self.names, st.sensor_status, st.sensor_calib):
            lvl = DiagnosticStatus.OK if status == "OK" else (
                DiagnosticStatus.WARN if status in ("SLIP", "SUSPECT") else DiagnosticStatus.ERROR)
            d = DiagnosticStatus(name=f"tram_nav: odometry {name}", hardware_id=name, level=lvl, message=status)
            d.values = [KeyValue(key="calibration", value=f"{calib:.4f}")]
            arr.status.append(d)
        self.pub_diag.publish(arr)

    def log_summary(self):
        p = self.perf()
        self.get_logger().info(
            f"outputs {self._n_out}, step {p['step_ms']:.3f} ms (max {p['step_max_ms']:.2f}), latency "
            f"{p['latency_ms']:.2f} ms (max {p['latency_max_ms']:.2f}), rate {p['rate_hz']:.1f} Hz, "
            f"CPU {p['cpu_pct']:.1f} %, RSS {p['rss_mb']:.0f} MB, overruns {self._overruns}")


def main(args=None):
    rclpy.init(args=args)
    node = NavigatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.log_summary()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
