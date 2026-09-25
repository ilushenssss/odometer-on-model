"""ROS 2 node: on-line accuracy evaluation against the simulator ground truth.

Subscribes ``~/ground_truth`` (nav_msgs/Odometry from the simulator, the
along-track distance is carried in ``pose.position.z``) and ``~/nav_state``
(std_msgs/Float64MultiArray of the navigator).  Publishes
``~/errors`` = [e_s, e_v, rmse_v, max|e_s|, sigma_s] and optionally writes a
CSV log for offline analysis.
"""
from __future__ import annotations

import csv
import math

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from .navigator_node import STATE_FIELDS


class EvaluatorNode(Node):
    def __init__(self):
        super().__init__("tram_nav_evaluator")
        self.declare_parameter("csv_path", "")
        self.declare_parameter("report_period", 10.0)
        self.truth = None
        self.n = 0
        self.se_v = 0.0
        self.max_es = 0.0
        self.rows = []
        self.create_subscription(Odometry, "ground_truth", self._on_truth, 20)
        self.create_subscription(Float64MultiArray, "nav_state", self._on_nav, 20)
        self.pub = self.create_publisher(Float64MultiArray, "errors", 10)
        self.csv_path = self.get_parameter("csv_path").value
        self.create_timer(float(self.get_parameter("report_period").value), self._report)

    def _on_truth(self, msg: Odometry):
        self.truth = (msg.pose.pose.position.z, msg.twist.twist.linear.x)

    def _on_nav(self, msg: Float64MultiArray):
        if self.truth is None or len(msg.data) < len(STATE_FIELDS):
            return
        est = dict(zip(STATE_FIELDS, msg.data))
        es = est["s"] - self.truth[0]
        ev = est["v"] - self.truth[1]
        self.n += 1
        self.se_v += ev * ev
        self.max_es = max(self.max_es, abs(es))
        rmse = math.sqrt(self.se_v / self.n)
        self.pub.publish(Float64MultiArray(data=[es, ev, rmse, self.max_es, est["sigma_s"]]))
        if self.csv_path:
            t = self.get_clock().now().nanoseconds * 1e-9
            self.rows.append([t, self.truth[0], self.truth[1], est["s"], est["v"], est["sigma_s"],
                              est["sigma_v"], est["mode"]])

    def _report(self):
        if self.n:
            self.get_logger().info(f"speed RMSE {math.sqrt(self.se_v / self.n):.3f} m/s, "
                                   f"max position error {self.max_es:.1f} m ({self.n} samples)")

    def save(self):
        if self.csv_path and self.rows:
            with open(self.csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["t", "s_true", "v_true", "s_est", "v_est", "sigma_s", "sigma_v", "mode"])
                w.writerows(self.rows)
            self.get_logger().info(f"log written to {self.csv_path}")


def main(args=None):
    rclpy.init(args=args)
    node = EvaluatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._report()
        node.save()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
