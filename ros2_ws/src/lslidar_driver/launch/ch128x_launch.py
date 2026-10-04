"""启动 CH128X1 驱动，并按需打开 RViz2。"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_dir = Path(get_package_share_directory("lslidar_driver"))
    config_path = package_dir / "config" / "lslidar_ch.yaml"
    rviz_path = package_dir / "rviz" / "lslidar_ch.rviz"
    show_rviz = LaunchConfiguration("rviz")

    return LaunchDescription(
        [
            DeclareLaunchArgument("rviz", default_value="false", description="是否同时打开 RViz2"),
            Node(
                package="lslidar_driver",
                executable="lslidar_driver_node",
                name="lslidar_driver_node",
                namespace="ch",
                parameters=[str(config_path)],
                output="screen",
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                arguments=["-d", str(rviz_path)],
                condition=IfCondition(show_rviz),
                output="screen",
            ),
        ]
    )
