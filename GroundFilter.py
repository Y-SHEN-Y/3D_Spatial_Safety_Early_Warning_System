"""用 Open3D RANSAC 找地面平面，并从当前帧点云中标出地面点。"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "ground_filter_config.yaml"


@dataclass(frozen=True)
class GroundFilterResult:
    non_ground_points: np.ndarray
    ground_mask: np.ndarray
    plane: np.ndarray | None
    status: str = "none"


class GroundFilter:
    def __init__(self, config_path: str | Path = DEFAULT_CONFIG_PATH):
        with Path(config_path).open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        if not isinstance(config, dict):
            raise ValueError("地面过滤配置必须是 YAML 映射")

        self.enabled = bool(config.get("enabled", True))
        self.voxel_size = float(config.get("voxel_size", 0.15))
        self.distance_threshold = float(config.get("distance_threshold", 0.08))
        self.ransac_n = int(config.get("ransac_n", 3))
        self.num_iterations = int(config.get("num_iterations", 150))
        self.max_planes = int(config.get("max_planes", 3))
        self.candidate_max_z = float(config.get("candidate_max_z", 0.0))
        self.ground_z_min = float(config.get("ground_z_min", -4.0))
        self.ground_z_max = float(config.get("ground_z_max", -0.05))
        self.max_tilt_degrees = float(config.get("max_tilt_degrees", 15.0))
        self.min_candidate_points = int(config.get("min_candidate_points", 60))
        self.min_plane_inliers = int(config.get("min_plane_inliers", 50))
        self.min_plane_ratio = float(config.get("min_plane_ratio", 0.05))
        self.max_plane_hold_frames = int(config.get("max_plane_hold_frames", 2))
        self.min_hold_ground_points = int(config.get("min_hold_ground_points", 10))
        if not (
            self.voxel_size > 0
            and self.distance_threshold > 0
            and self.ransac_n >= 3
            and self.num_iterations >= 1
            and self.max_planes >= 1
            and self.ground_z_min < self.ground_z_max <= self.candidate_max_z
            and 0 <= self.max_tilt_degrees < 90
            and self.min_candidate_points >= self.ransac_n
            and self.min_plane_inliers >= self.ransac_n
            and 0 < self.min_plane_ratio <= 1
            and self.max_plane_hold_frames >= 0
            and self.min_hold_ground_points >= 1
        ):
            raise ValueError("地面过滤参数无效，请检查 ground_filter_config.yaml")
        self._min_vertical_cosine = float(np.cos(np.deg2rad(self.max_tilt_degrees)))
        self._last_plane: np.ndarray | None = None
        self._missed_frames = 0

    def _apply_plane(
        self, points: np.ndarray, finite: np.ndarray, plane: np.ndarray, status: str
    ) -> GroundFilterResult:
        # 旧平面只提供几何位置；每次判定仍使用当前帧的点，不叠加历史点云。
        ground_mask = np.zeros(len(points), dtype=bool)
        ground_mask[finite] = (
            np.abs(points[finite] @ plane[:3] + plane[3]) <= self.distance_threshold
        )
        return GroundFilterResult(points[~ground_mask], ground_mask, plane, status)

    def _fallback(self, points: np.ndarray, finite: np.ndarray) -> GroundFilterResult:
        if (
            self._last_plane is not None
            and self._missed_frames < self.max_plane_hold_frames
        ):
            result = self._apply_plane(points, finite, self._last_plane, "held")
            self._missed_frames += 1
            if np.count_nonzero(result.ground_mask) >= self.min_hold_ground_points:
                return result
        else:
            self._missed_frames += 1
        if self._missed_frames >= self.max_plane_hold_frames:
            self._last_plane = None
        return GroundFilterResult(
            points, np.zeros(len(points), dtype=bool), None
        )

    def filter_points(self, points_lidar: np.ndarray) -> GroundFilterResult:
        """拟合当前帧的地面；拟合不可信时保留所有点。"""
        points = np.asarray(points_lidar, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("雷达点云必须是 Nx3 数组")
        finite = np.isfinite(points).all(axis=1)
        if not self.enabled:
            self._last_plane = None
            self._missed_frames = 0
            return GroundFilterResult(points, np.zeros(len(points), dtype=bool), None)
        if len(points) < self.min_candidate_points:
            return self._fallback(points, finite)

        candidates = points[finite & (points[:, 2] <= self.candidate_max_z)]
        if len(candidates) < self.min_candidate_points:
            return self._fallback(points, finite)

        # 降采样仅加快平面估计，判定地面时仍遍历当前帧原始点。
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(candidates)
        cloud = cloud.voxel_down_sample(self.voxel_size)
        original_candidate_count = len(cloud.points)
        if original_candidate_count < self.min_candidate_points:
            return self._fallback(points, finite)

        for _ in range(self.max_planes):
            if len(cloud.points) < self.min_candidate_points:
                break
            model, inlier_indices = cloud.segment_plane(
                distance_threshold=self.distance_threshold,
                ransac_n=self.ransac_n,
                num_iterations=self.num_iterations,
            )
            plane = np.asarray(model, dtype=np.float64)
            normal_length = np.linalg.norm(plane[:3])
            if normal_length == 0:
                break
            plane /= normal_length
            if plane[2] < 0:
                plane = -plane
            plane_height = -plane[3] / plane[2] if plane[2] != 0 else np.inf
            supported = (
                len(inlier_indices) >= self.min_plane_inliers
                and len(inlier_indices) / original_candidate_count >= self.min_plane_ratio
            )
            plausible_ground = (
                plane[2] >= self._min_vertical_cosine
                and self.ground_z_min <= plane_height <= self.ground_z_max
            )
            if supported and plausible_ground:
                self._last_plane = plane.copy()
                self._missed_frames = 0
                return self._apply_plane(points, finite, plane, "fitted")

            # 最大平面若为墙面或其他结构，移除它后继续寻找地面。
            if not inlier_indices:
                break
            cloud = cloud.select_by_index(inlier_indices, invert=True)

        return self._fallback(points, finite)
