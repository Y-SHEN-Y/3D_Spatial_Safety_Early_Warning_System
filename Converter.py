"""将雷达点云投影到相机图像，同时保留像素与三维点的对应关系。"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

from GroundFilter import GroundFilter
from Point_Cloud import PointCloudSubscriber


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "bin_cam_config.yaml"


@dataclass(frozen=True)
class ProjectedCloud:
    """同一帧中可见的雷达点；各数组的第 i 行属于同一个点。"""

    pixels_xy: np.ndarray
    points_lidar: np.ndarray
    points_camera: np.ndarray
    depth_m: np.ndarray
    depth_map: np.ndarray


class Converter:
    def __init__(
        self,
        point_cloud: PointCloudSubscriber,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        max_cloud_age_seconds: float = 0.5,
        lidar_frame: str = "laser_link",
        ground_filter: GroundFilter | None = None,
    ):
        if max_cloud_age_seconds <= 0:
            raise ValueError("点云有效期必须大于 0")
        self.point_cloud = point_cloud
        self.max_cloud_age_seconds = max_cloud_age_seconds
        self.lidar_frame = lidar_frame
        self.ground_filter = ground_filter
        self.last_ground_count = 0
        self.last_ground_status = "none"
        self._last_message: PointCloud2 | None = None
        self._last_projection: ProjectedCloud | None = None
        self._last_image_shape: tuple[int, int] | None = None

        with Path(config_path).open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        calib = config["calib"]
        self.width = int(config["param"]["Width"])
        self.height = int(config["param"]["Height"])
        self.max_depth = float(config["param"]["max_depth"])
        self.intrinsic = np.asarray(calib["intrinsic_r"], dtype=np.float64)
        self.distortion = np.asarray(calib["distortion_r"], dtype=np.float64)
        camera_to_lidar_rotation = np.asarray(
            calib["extrinsic_to_lidar_R_r"], dtype=np.float64
        )
        self.camera_to_lidar_translation = np.asarray(
            calib["extrinsic_to_lidar_T_r"], dtype=np.float64
        )
        if self.intrinsic.shape != (3, 3) or camera_to_lidar_rotation.shape != (3, 3):
            raise ValueError("相机内参和旋转矩阵必须是 3×3")
        if self.camera_to_lidar_translation.shape != (3,):
            raise ValueError("相机到雷达的平移向量必须有 3 个元素")
        if self.width <= 0 or self.height <= 0 or self.max_depth <= 0:
            raise ValueError("图像宽高和最大深度必须大于 0")
        if not np.all(np.isfinite(camera_to_lidar_rotation)) or not np.all(np.isfinite(self.intrinsic)):
            raise ValueError("标定矩阵包含无效数值")
        if not np.isclose(np.linalg.det(camera_to_lidar_rotation), 1.0, atol=1e-2):
            raise ValueError("相机到雷达的旋转矩阵无效")

        # 配置给的是相机→雷达；投影前必须取逆，得到雷达→相机。
        self.lidar_to_camera_rotation = np.linalg.inv(camera_to_lidar_rotation)

    @staticmethod
    def points_from_message(message: PointCloud2) -> np.ndarray:
        """从 PointCloud2 取出 Nx3 的 xyz，过滤 NaN 和无穷大。"""
        field_names = {field.name for field in message.fields}
        if not {"x", "y", "z"}.issubset(field_names):
            raise ValueError("点云缺少 x、y 或 z 字段")
        points = point_cloud2.read_points(
            message, field_names=["x", "y", "z"], skip_nans=True
        )
        xyz = np.column_stack([points[name] for name in ("x", "y", "z")])
        return np.asarray(xyz[np.isfinite(xyz).all(axis=1)], dtype=np.float64)

    def project_points_with_xyz(
        self, points_lidar: np.ndarray, image_shape: tuple[int, int]
    ) -> ProjectedCloud:
        """投影单帧点云，并保存每个可见像素对应的原始三维点。"""
        if tuple(image_shape) != (self.height, self.width):
            raise ValueError(
                f"投影必须使用原始 {self.width}×{self.height} 图像；收到 {image_shape[1]}×{image_shape[0]}"
            )
        points_lidar = np.asarray(points_lidar, dtype=np.float64)
        if points_lidar.ndim != 2 or points_lidar.shape[1] != 3:
            raise ValueError("雷达点云必须是 Nx3 数组")
        depth_map = np.zeros((self.height, self.width), dtype=np.float32)
        empty_pixels = np.empty((0, 2), dtype=np.int32)
        empty_points = np.empty((0, 3), dtype=np.float32)
        empty_depth = np.empty(0, dtype=np.float32)

        def empty_projection() -> ProjectedCloud:
            return ProjectedCloud(
                empty_pixels, empty_points, empty_points.copy(), empty_depth, depth_map
            )

        if len(points_lidar) == 0:
            return empty_projection()

        finite = np.isfinite(points_lidar).all(axis=1)
        points_lidar = points_lidar[finite]
        if not len(points_lidar):
            return empty_projection()
        # 行向量形式：p_cam = R_lidar_to_cam × (p_lidar - T_cam_to_lidar)。
        points_camera = (points_lidar - self.camera_to_lidar_translation) @ self.lidar_to_camera_rotation.T
        depth = points_camera[:, 2]
        in_front = (depth > 0) & (depth <= self.max_depth)
        points_lidar = points_lidar[in_front]
        points_camera = points_camera[in_front]
        depth = depth[in_front]
        if not len(points_camera):
            return empty_projection()

        # 使用 OpenCV 畸变模型投影到未去畸变的相机原图。
        image_points, _ = cv2.projectPoints(
            points_camera,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            self.intrinsic,
            self.distortion,
        )
        image_points = image_points.reshape(-1, 2)
        inside = (
            np.isfinite(image_points).all(axis=1)
            & (image_points[:, 0] >= -0.5)
            & (image_points[:, 0] < self.width - 0.5)
            & (image_points[:, 1] >= -0.5)
            & (image_points[:, 1] < self.height - 0.5)
        )
        pixels = np.rint(image_points[inside]).astype(np.int32)
        if not len(pixels):
            return empty_projection()
        points_lidar = points_lidar[inside]
        points_camera = points_camera[inside]
        depth = depth[inside]

        # 同像素只保留最近的雷达回波，并让像素和两个坐标系的三维点保持同一索引。
        pixel_id = pixels[:, 1] * self.width + pixels[:, 0]
        ordered = np.lexsort((depth, pixel_id))
        ordered_pixel_id = pixel_id[ordered]
        first = np.r_[True, ordered_pixel_id[1:] != ordered_pixel_id[:-1]]
        selected = ordered[first]
        pixels = pixels[selected]
        points_lidar = points_lidar[selected].astype(np.float32)
        points_camera = points_camera[selected].astype(np.float32)
        depth = depth[selected].astype(np.float32)
        depth_map[pixels[:, 1], pixels[:, 0]] = depth
        return ProjectedCloud(pixels, points_lidar, points_camera, depth, depth_map)

    def project_points(self, points_lidar: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
        """兼容原接口：只返回以米为单位的深度图，空像素为 0。"""
        return self.project_points_with_xyz(points_lidar, image_shape).depth_map

    def project_latest_cloud(self, image_shape: tuple[int, int]) -> ProjectedCloud | None:
        """只处理最新且未过期的一帧，不合并三帧点云。"""
        message = self.point_cloud.get_latest(self.max_cloud_age_seconds)
        if message is None:
            self.last_ground_count = 0
            self.last_ground_status = "none"
            self._last_message = None
            self._last_projection = None
            return None
        if message is self._last_message and tuple(image_shape) == self._last_image_shape:
            return self._last_projection
        if message.header.frame_id != self.lidar_frame:
            raise ValueError(
                f"点云坐标系为 {message.header.frame_id!r}，预期为 {self.lidar_frame!r}；需要先做 TF 变换"
            )
        points = self.points_from_message(message)
        if self.ground_filter is not None:
            filtered = self.ground_filter.filter_points(points)
            self.last_ground_count = int(np.count_nonzero(filtered.ground_mask))
            self.last_ground_status = filtered.status
            points = filtered.non_ground_points
        else:
            self.last_ground_count = 0
            self.last_ground_status = "none"
        projection = self.project_points_with_xyz(points, image_shape)
        self._last_message = message
        self._last_image_shape = tuple(image_shape)
        self._last_projection = projection
        return projection

    def project_latest(self, image_shape: tuple[int, int]) -> np.ndarray | None:
        """兼容原接口：只返回最新帧的深度图。"""
        projection = self.project_latest_cloud(image_shape)
        return None if projection is None else projection.depth_map

    def draw_depth(self, image: np.ndarray, depth_map: np.ndarray | None) -> np.ndarray:
        """仅为显示给投影点上色；深度图本身仍保留原始米数。"""
        result = image.copy()
        if depth_map is None:
            return result
        if depth_map.shape != image.shape[:2]:
            raise ValueError("深度图尺寸与图像尺寸不一致")
        valid = depth_map > 0
        if not np.any(valid):
            return result
        scaled = np.clip(depth_map / self.max_depth * 255, 0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(255 - scaled, cv2.COLORMAP_TURBO)
        # 膨胀仅用于让稀疏点在窗口中更容易看见，不影响深度数据。
        colored = cv2.dilate(colored, np.ones((3, 3), dtype=np.uint8))
        display_mask = cv2.dilate(valid.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)).astype(bool)
        blended = cv2.addWeighted(result, 0.35, colored, 0.65, 0)
        result[display_mask] = blended[display_mask]
        return result
