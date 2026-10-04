#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/humble/setup.bash
exec /home/shenby/miniconda3/envs/torch/bin/python "$project_dir/main.py" "$@"
