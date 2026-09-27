"""Full closed-loop demo: simulator + navigator + evaluator.

    ros2 launch tram_nav sim_demo.launch.py scenario:=autumn real_time_factor:=5.0
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("tram_nav")
    track = LaunchConfiguration("track_csv")
    rtf = LaunchConfiguration("real_time_factor")
    # faster than real time -> everybody follows the simulator's /clock
    sim_time = PythonExpression(["'", rtf, "' != '1.0'"])
    sim = Node(
        package="tram_nav", executable="simulator", name="tram_simulator", namespace="tram", output="screen",
        parameters=[os.path.join(share, "config", "simulator.yaml"),
                    {"scenario": LaunchConfiguration("scenario"), "track_csv": track,
                     "real_time_factor": rtf, "publish_clock": sim_time}],
    )
    nav = Node(
        package="tram_nav", executable="navigator", name="tram_navigator", namespace="tram", output="screen",
        parameters=[os.path.join(share, "config", "navigator.yaml"),
                    {"track_csv": track, "use_sim_time": sim_time}],
        remappings=[("odometry", "nav/odometry"), ("pose", "nav/pose"), ("speed", "nav/speed"),
                    ("distance", "nav/distance"), ("state", "nav/state"), ("set_position", "nav/set_position")],
    )
    ev = Node(
        package="tram_nav", executable="evaluator", name="tram_nav_evaluator", namespace="tram", output="screen",
        parameters=[{"csv_path": LaunchConfiguration("log_csv"), "use_sim_time": sim_time,
                     "reference": "ground_truth", "truth_topic": "/tram/ground_truth", "track_csv": track}],
    )
    return LaunchDescription([
        DeclareLaunchArgument("scenario", default_value="nominal"),
        DeclareLaunchArgument("real_time_factor", default_value="1.0"),
        DeclareLaunchArgument("track_csv", default_value=os.path.join(share, "config", "demo_track.csv")),
        DeclareLaunchArgument("log_csv", default_value=""),
        sim, nav, ev,
        RegisterEventHandler(OnProcessExit(target_action=sim, on_exit=[EmitEvent(event=Shutdown())])),
    ])
