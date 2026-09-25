"""Navigator only - for the real vehicle (or a rosbag).

    ros2 launch tram_nav navigator.launch.py track_csv:=/path/to/line.csv
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("tram_nav")
    return LaunchDescription([
        DeclareLaunchArgument("track_csv", default_value=os.path.join(share, "config", "demo_track.csv")),
        DeclareLaunchArgument("params", default_value=os.path.join(share, "config", "navigator.yaml")),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        Node(
            package="tram_nav", executable="navigator", name="tram_navigator", namespace="tram",
            output="screen",
            parameters=[LaunchConfiguration("params"),
                        {"track_csv": LaunchConfiguration("track_csv"),
                         "use_sim_time": LaunchConfiguration("use_sim_time")}],
            remappings=[("odometry", "nav/odometry"), ("pose", "nav/pose"), ("speed", "nav/speed"),
                        ("distance", "nav/distance"), ("state", "nav/state"),
                        ("set_position", "nav/set_position")],
        ),
    ])
