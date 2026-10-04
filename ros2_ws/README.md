# CH128X 雷达的 ROS 2 工作区

本机系统是 Ubuntu 22.04，使用 ROS 2 Humble。`src` 中是[镭神官方 ROS 2 驱动](https://github.com/Lslidar/Lslidar_ROS2_driver/tree/LS-S1_V1.0)的 `LS-S1_V1.0` 分支，包含 `lslidar_driver` 和 `lslidar_msgs`。项目配置按 `CH128X1` 型号准备；如果设备铭牌是其他子型号，需要先核对 `config/lslidar_ch.yaml` 的 `lidar_model`。

驱动配置在 `src/lslidar_driver/config/lslidar_ch.yaml`。当前雷达 IP 为厂商示例 `192.168.1.200`，数据端口为 `2368`，设备端口为 `2369`，点云发布到 `/cloudpoints`，坐标系名为 `laser_link`。**启动前应核对实际雷达 IP 和电脑有线网口地址；两者应在同一网段，且网线应有链路。**

系统安装完成后，构建和启动：

```bash
bash ros2_ws/build_lidar.sh
bash ros2_ws/run_lidar.sh
```

构建脚本会把源码复制到 `/tmp` 下的英文路径编译，因为 ROS 2 Humble 的消息生成器在当前项目的中文路径下会报错；安装结果仍写入本工作区的 `install` 目录。

需要同时打开 RViz2 时：

```bash
bash ros2_ws/run_lidar.sh rviz:=true
```

另开终端检查点云：

```bash
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
ros2 topic type /cloudpoints
ros2 topic hz /cloudpoints
```

正常情况下类型应为 `sensor_msgs/msg/PointCloud2`，且频率不是零。项目的 `Point_Cloud.py` 已使用 ROS 2 `rclpy` 订阅 `/cloudpoints`，缓存最近三帧消息；融合模块只处理最新且未过期的一帧。
