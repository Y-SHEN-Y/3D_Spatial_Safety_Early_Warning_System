#!/usr/bin/env bash
set -euo pipefail

# 雷达驱动作为独立 ROS 2 进程运行，不混用 torch Conda 环境的 Python。
export PATH="/opt/ros/humble/bin:/usr/local/bin:/usr/bin:/bin"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROS_LOG_DIR="$workspace_dir/log/ros"
mkdir -p "$ROS_LOG_DIR"
set +u
source /opt/ros/humble/setup.bash
source "$workspace_dir/install/setup.bash"
set -u
exec ros2 launch lslidar_driver ch128x_launch.py "$@"
