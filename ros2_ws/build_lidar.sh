#!/usr/bin/env bash
set -euo pipefail

# 用系统 Python 构建 ROS 2 包，避免当前 Conda 环境影响 ament 和 colcon。
export PATH="/opt/ros/humble/bin:/usr/local/bin:/usr/bin:/bin"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV
export CMAKE_BUILD_PARALLEL_LEVEL=2
export MAKEFLAGS="-j2"

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set +u
source /opt/ros/humble/setup.bash
set -u

# Humble 的消息生成器不能可靠处理当前项目路径中的中文目录名。
# 源码仍保存在项目内，编译时复制到临时英文路径，安装结果写回项目。
build_dir="$(mktemp -d /tmp/lslidar-build.XXXXXX)"
mkdir -p "$build_dir/src"
cp -a "$workspace_dir/src/." "$build_dir/src/"
cd "$build_dir"
colcon build --install-base "$workspace_dir/install" --packages-select lslidar_msgs lslidar_driver
