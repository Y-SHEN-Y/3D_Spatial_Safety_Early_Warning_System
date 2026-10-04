"""订阅 ROS 2 点云话题，并缓存最近三帧 PointCloud2 消息。"""

from collections import deque
from threading import Lock, Thread
from time import monotonic

import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2


class PointCloudSubscriber:
    def __init__(self, topic: str = "/cloudpoints", queue_size: int = 3):
        if queue_size < 1:
            raise ValueError("点云缓存帧数必须大于 0")
        self.topic = topic
        self.queue_size = queue_size
        self._queue: deque[tuple[PointCloud2, float]] = deque(maxlen=queue_size)
        self._lock = Lock()
        self._context = Context()
        rclpy.init(context=self._context)
        self._node = rclpy.create_node("point_cloud_subscriber", context=self._context)
        self._subscription = self._node.create_subscription(
            PointCloud2, topic, self._cloud_callback, qos_profile_sensor_data
        )
        self._executor = SingleThreadedExecutor(context=self._context)
        self._executor.add_node(self._node)
        self._thread: Thread | None = None
        self._closed = False

    def _cloud_callback(self, message: PointCloud2) -> None:
        # 收包时间使用单调时钟，避免系统时间调整影响过期判断。
        with self._lock:
            self._queue.append((message, monotonic()))

    def start(self) -> None:
        """在后台处理 ROS 回调，使相机读取与检测循环不被阻塞。"""
        if self._closed:
            raise RuntimeError("点云订阅器已关闭")
        if self._thread is None:
            self._thread = Thread(target=self._executor.spin, daemon=True)
            self._thread.start()

    def get_latest(self, max_age_seconds: float | None = None) -> PointCloud2 | None:
        """获取最新一帧；指定时限后，超时的旧帧视为无数据。"""
        if max_age_seconds is not None and max_age_seconds <= 0:
            raise ValueError("点云有效期必须大于 0")
        with self._lock:
            if not self._queue:
                return None
            message, received_at = self._queue[-1]
        if max_age_seconds is not None and monotonic() - received_at > max_age_seconds:
            return None
        return message

    def get_queue(self) -> list[PointCloud2]:
        """返回最近帧的快照，按接收时间从旧到新排列。"""
        with self._lock:
            return [message for message, _ in self._queue]

    def get_queue_len(self) -> int:
        with self._lock:
            return len(self._queue)

    def is_ready(self) -> bool:
        """三帧缓存是否已装满；首次投影不必等待装满。"""
        return self.get_queue_len() >= self.queue_size

    def close(self) -> None:
        """停止回调线程并释放 ROS 2 资源。"""
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._executor.remove_node(self._node)
        self._node.destroy_node()
        self._context.shutdown()


if __name__ == "__main__":
    subscriber = PointCloudSubscriber()
    subscriber.start()
    print("正在接收 /cloudpoints；按 Ctrl+C 退出")
    try:
        from time import sleep

        while True:
            sleep(1)
            latest = subscriber.get_latest(max_age_seconds=1.0)
            if latest is None:
                print("等待点云...")
            else:
                print(f"已缓存 {subscriber.get_queue_len()} 帧，最新一帧有 {latest.width * latest.height} 个点")
    except KeyboardInterrupt:
        pass
    finally:
        subscriber.close()
