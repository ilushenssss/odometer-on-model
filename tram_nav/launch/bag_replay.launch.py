"""Replay a rosbag through the navigator and evaluate it against the GNSS reference.

    ros2 launch tram_nav bag_replay.launch.py bag:=/path/to/bag track_csv:=/path/to/track.csv

Arguments
    bag           rosbag2 directory; empty -> play it yourself: ros2 bag play <bag> --clock
    rate          playback rate (default 1.0)
    params        navigator parameter file (input topic names/types/units, tram data) -
                  default <share>/config/jury.yaml
    track_csv     line map (lat,lon[,alt][,station] or s,x,y,z,station)
    reference     gnss | ground_truth        gnss_topic   reference NavSatFix topic
    out_dir       where eval.csv / summary.json are written (default ./tram_nav_eval)
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, ExecuteProcess, OpaqueFunction, RegisterEventHandler,
                            TimerAction)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _setup(context):
    share = get_package_share_directory("tram_nav")
    lc = lambda k: LaunchConfiguration(k).perform(context)  # noqa: E731
    out_dir = os.path.abspath(lc("out_dir"))
    os.makedirs(out_dir, exist_ok=True)
    track = lc("track_csv")
    params = lc("params") or os.path.join(share, "config", "jury.yaml")
    nav = Node(package="tram_nav", executable="navigator", name="tram_navigator", output="screen",
               parameters=[params, {"use_sim_time": True, "track_csv": track}])
    ev = Node(package="tram_nav", executable="evaluator", name="tram_nav_evaluator", output="screen",
              parameters=[{"use_sim_time": True, "track_csv": track, "reference": lc("reference"),
                           "gnss_topic": lc("gnss_topic"), "truth_topic": lc("truth_topic"),
                           "csv_path": os.path.join(out_dir, "eval.csv"),
                           "summary_path": os.path.join(out_dir, "summary.json")}])
    actions = [nav, ev]
    bag = lc("bag")
    if bag:
        play = ExecuteProcess(cmd=["ros2", "bag", "play", bag, "--clock", "100", "-r", lc("rate")], output="screen")
        actions.append(TimerAction(period=3.0, actions=[play]))
        actions.append(RegisterEventHandler(OnProcessExit(
            target_action=play, on_exit=[TimerAction(period=3.0, actions=[EmitEvent(event=Shutdown())])])))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("bag", default_value=""),
        DeclareLaunchArgument("rate", default_value="1.0"),
        DeclareLaunchArgument("params", default_value=""),
        DeclareLaunchArgument("track_csv", default_value=""),
        DeclareLaunchArgument("reference", default_value="gnss"),
        DeclareLaunchArgument("gnss_topic", default_value="/tram/gnss"),
        DeclareLaunchArgument("truth_topic", default_value="/tram/ground_truth"),
        DeclareLaunchArgument("out_dir", default_value="tram_nav_eval"),
        OpaqueFunction(function=_setup),
    ])
