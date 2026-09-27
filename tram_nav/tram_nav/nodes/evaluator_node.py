"""ROS 2 node: accuracy / performance evaluation of /result/* against a reference.

Reference (``reference`` parameter):
    ``gnss``          sensor_msgs/NavSatFix on ``gnss_topic`` (WGS-84), converted into the
                      local map frame with the map origin (``track_csv`` with lat/lon or
                      ``geo_origin``). Reference speed = finite difference of GNSS positions
                      over +-``gnss_speed_window`` s.
    ``ground_truth``  nav_msgs/Odometry on ``truth_topic`` (simulator: pose, twist.linear.x,
                      along-track distance in pose.position.z).

Estimates are taken from ``/result/position`` (PoseStamped | PointStamped |
PoseWithCovarianceStamped) and ``/result/velocity`` (TwistStamped | Vector3Stamped),
matched to the reference by header stamps (linear interpolation of the reference).
Performance (step time, latency, output rate, CPU, memory) comes from /result/perf.

Outputs: ``/eval/errors`` std_msgs/Float64MultiArray
[e_2d, e_along, e_v, rms_2d, max_2d, rms_v, n], a periodic log line, a CSV with every
matched sample (``csv_path``) and a JSON summary (``summary_path``, rewritten every
``report_period`` s and at shutdown).
"""
from __future__ import annotations

import bisect
import collections
import csv
import json
import math
import os

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped, PoseWithCovarianceStamped, TwistStamped, Vector3Stamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float64MultiArray

from ..core.geo import LocalFrame
from ..core.track import TrackMap, generate_demo_track

POS_TYPES = {"pose": PoseStamped, "point": PointStamped, "pose_cov": PoseWithCovarianceStamped}
VEL_TYPES = {"twist": TwistStamped, "vector3": Vector3Stamped}


def _t(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


class EvaluatorNode(Node):
    def __init__(self, **kwargs):
        super().__init__("tram_nav_evaluator", **kwargs)
        P = self.declare_parameter
        P("reference", "gnss")                       # gnss | ground_truth
        P("gnss_topic", "/tram/gnss")
        P("truth_topic", "/tram/ground_truth")
        P("track_csv", "")
        P("geo_origin", [0.0, 0.0])
        P("position_topic", "/result/position")
        P("velocity_topic", "/result/velocity")
        P("position_type", "pose")
        P("velocity_type", "twist")
        P("gnss_speed_window", 2.0)                  # +-s for the reference speed from GNSS positions
        P("max_ref_gap", 1.0)                        # no interpolation across larger reference gaps [s]
        P("csv_path", "")
        P("summary_path", "")
        P("report_period", 10.0)
        g = self.get_parameter
        self.reference = g("reference").value
        track_csv = g("track_csv").value
        self.track = TrackMap.from_csv(track_csv) if track_csv and os.path.isfile(track_csv) else generate_demo_track()
        go = list(g("geo_origin").value)
        if abs(go[0]) > 1e-9 or abs(go[1]) > 1e-9:
            self.track.geo = LocalFrame(go[0], go[1])
        if self.reference == "gnss" and self.track.geo is None:
            raise ValueError("reference=gnss needs a map with lat/lon (track_csv) or the geo_origin parameter")
        self.win = float(g("gnss_speed_window").value)
        self.max_gap = float(g("max_ref_gap").value)
        self.csv_path, self.summary_path = g("csv_path").value, g("summary_path").value

        qos = QoSProfile(depth=200, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.ref_t, self.ref_x, self.ref_y, self.ref_v, self.ref_s = [], [], [], [], []
        if self.reference == "gnss":
            self.create_subscription(NavSatFix, g("gnss_topic").value, self._on_gnss, qos)
        else:
            self.create_subscription(Odometry, g("truth_topic").value, self._on_truth, qos)
        self.pos_type = g("position_type").value
        self.create_subscription(POS_TYPES[self.pos_type], g("position_topic").value, self._on_pos, qos)
        self.create_subscription(VEL_TYPES[g("velocity_type").value], g("velocity_topic").value, self._on_vel, qos)
        self.create_subscription(Float64MultiArray, "/result/perf", self._on_perf, 10)
        self.pub = self.create_publisher(Float64MultiArray, "/eval/errors", 10)
        self.pending = collections.deque()      # (t, x, y) estimates waiting for the reference
        self.vel = {}                            # stamp -> speed
        self.rows = []
        self.perf = []
        self.create_timer(0.2, self._evaluate)
        self.create_timer(float(g("report_period").value), self._report)
        self.get_logger().info(f"evaluating {g('position_topic').value}, {g('velocity_topic').value} against "
                               f"{self.reference} ({g('gnss_topic').value if self.reference == 'gnss' else g('truth_topic').value})")

    # ------------------------------------------------------------------ inputs
    def _add_ref(self, t, x, y, v=None, s=None):
        if self.ref_t and t <= self.ref_t[-1]:
            return
        self.ref_t.append(t)
        self.ref_x.append(x)
        self.ref_y.append(y)
        self.ref_v.append(v)
        self.ref_s.append(s)

    def _on_gnss(self, m: NavSatFix):
        if m.status.status < 0 or not (math.isfinite(m.latitude) and math.isfinite(m.longitude)):
            return                                   # no fix
        x, y = self.track.geo.to_xy(m.latitude, m.longitude)
        self._add_ref(_t(m.header.stamp), x, y)

    def _on_truth(self, m: Odometry):
        p = m.pose.pose.position
        self._add_ref(_t(m.header.stamp), p.x, p.y, m.twist.twist.linear.x, p.z)

    def _on_pos(self, m):
        p = m.point if self.pos_type == "point" else (m.pose.pose.position if self.pos_type == "pose_cov"
                                                      else m.pose.position)
        self.pending.append((_t(m.header.stamp), p.x, p.y))

    def _on_vel(self, m):
        v = m.twist.linear.x if isinstance(m, TwistStamped) else math.hypot(m.vector.x, m.vector.y)
        self.vel[round(_t(m.header.stamp), 4)] = v

    def _on_perf(self, m: Float64MultiArray):
        if len(m.data) >= 5:
            self.perf.append(list(m.data[:5]))

    # ------------------------------------------------------------------ matching
    def _interp(self, arr, t):
        i = bisect.bisect_left(self.ref_t, t)
        if i == 0 or i >= len(self.ref_t):
            return None
        t0, t1 = self.ref_t[i - 1], self.ref_t[i]
        if t1 - t0 > self.max_gap:
            return None
        w = (t - t0) / (t1 - t0)
        a0, a1 = arr[i - 1], arr[i]
        if a0 is None or a1 is None:
            return None
        return a0 + w * (a1 - a0)

    def _ref_speed(self, t):
        if self.reference != "gnss":
            return self._interp(self.ref_v, t)
        xa, ya = self._interp(self.ref_x, t - self.win), self._interp(self.ref_y, t - self.win)
        xb, yb = self._interp(self.ref_x, t + self.win), self._interp(self.ref_y, t + self.win)
        if None in (xa, ya, xb, yb):
            return None
        return math.hypot(xb - xa, yb - ya) / (2.0 * self.win)

    def _evaluate(self):
        if not self.ref_t:
            return
        horizon = self.ref_t[-1] - (self.win + 0.05 if self.reference == "gnss" else 0.0)
        while self.pending and self.pending[0][0] <= horizon:
            t, x, y = self.pending.popleft()
            xr, yr = self._interp(self.ref_x, t), self._interp(self.ref_y, t)
            if xr is None or yr is None:
                continue
            e2d = math.hypot(x - xr, y - yr)
            s_ref = self._interp(self.ref_s, t) if self.reference != "gnss" else None
            if s_ref is None:
                s_ref = self.track.project(xr, yr)
            e_along = self.track.project(x, y) - s_ref
            v = self.vel.pop(round(t, 4), None)
            vr = self._ref_speed(t)
            ev = (v - vr) if (v is not None and vr is not None) else float("nan")
            self.rows.append((t, xr, yr, x, y, e2d, e_along, vr if vr is not None else float("nan"),
                              v if v is not None else float("nan"), ev))
        if len(self.vel) > 20000:
            self.vel.clear()
        if self.rows and rclpy.ok():
            r = self.rows[-1]
            m = self.metrics()
            self.pub.publish(Float64MultiArray(data=[r[5], r[6], r[9], m["rms_2d_m"], m["max_2d_m"],
                                                     m["rms_v_mps"], float(m["n"])]))

    # ------------------------------------------------------------------ metrics
    def metrics(self):
        if not self.rows:
            return {"n": 0, "rms_2d_m": 0.0, "max_2d_m": 0.0, "rms_v_mps": 0.0}
        a = np.array(self.rows, dtype=float)
        e2d, ea, ev = a[:, 5], np.abs(a[:, 6]), a[:, 9]
        ev = ev[np.isfinite(ev)]
        out = {"reference": self.reference, "n": int(len(a)), "duration_s": float(a[-1, 0] - a[0, 0]),
               "rms_2d_m": float(np.sqrt(np.mean(e2d ** 2))), "mean_2d_m": float(np.mean(e2d)),
               "p95_2d_m": float(np.percentile(e2d, 95)), "max_2d_m": float(np.max(e2d)),
               "final_2d_m": float(e2d[-1]), "rms_along_m": float(np.sqrt(np.mean(ea ** 2))),
               "max_along_m": float(np.max(ea)),
               "rms_v_mps": float(np.sqrt(np.mean(ev ** 2))) if len(ev) else float("nan"),
               "p95_abs_v_mps": float(np.percentile(np.abs(ev), 95)) if len(ev) else float("nan"),
               "distance_m": float(abs(self.track.project(a[-1, 1], a[-1, 2]) - self.track.project(a[0, 1], a[0, 2])))}
        if self.perf:
            p = np.array(self.perf)
            out.update({"step_ms_mean": float(np.mean(p[:, 0])), "latency_ms_mean": float(np.mean(p[:, 1])),
                        "latency_ms_max_of_means": float(np.max(p[:, 1])),
                        "output_rate_hz": float(np.median(p[:, 2])),
                        "cpu_pct_mean": float(np.mean(p[:, 3])), "cpu_pct_max": float(np.max(p[:, 3])),
                        "rss_mb_max": float(np.max(p[:, 4]))})
        return out

    def _report(self):
        m = self.metrics()
        if m["n"]:
            extra = (f", latency {m['latency_ms_mean']:.2f} ms, rate {m['output_rate_hz']:.1f} Hz, "
                     f"CPU {m['cpu_pct_mean']:.1f} %, RSS {m['rss_mb_max']:.0f} MB") if "latency_ms_mean" in m else ""
            self.get_logger().info(f"[{self.reference}] n={m['n']}: 2D RMS {m['rms_2d_m']:.2f} m, "
                                   f"max {m['max_2d_m']:.1f} m, speed RMS {m['rms_v_mps']:.3f} m/s{extra}")
        self.save()

    def save(self):
        if self.summary_path:
            with open(self.summary_path, "w") as f:
                json.dump(self.metrics(), f, indent=1)
        if self.csv_path and self.rows:
            with open(self.csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t", "x_ref", "y_ref", "x_est", "y_est", "e_2d", "e_along", "v_ref", "v_est", "e_v"])
                w.writerows(self.rows)


def main(args=None):
    rclpy.init(args=args)
    node = EvaluatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._evaluate()
        node._report()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
