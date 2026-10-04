"""把有效实例轮廓与单帧雷达投影点关联，并只绘制可信目标点团。"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import yaml

from Converter import ProjectedCloud
from Detector import DANGEROUS_CATEGORY, MONITORED_CATEGORY, Detection


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "fusion_config.yaml"


@dataclass(frozen=True)
class FusedTarget:
    track_id: int
    class_id: int
    label: str
    category: str
    pixels_xy: np.ndarray
    points_lidar: np.ndarray
    points_camera: np.ndarray
    depth_m: np.ndarray


@dataclass
class _TrackState:
    center_lidar: np.ndarray
    last_frame: int


class Fusion:
    def __init__(self, config_path: str | Path = DEFAULT_CONFIG_PATH):
        with Path(config_path).open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        if not isinstance(config, dict):
            raise ValueError("融合配置必须是 YAML 映射")

        self.min_component_pixels = int(config.get("min_component_pixels", 64))
        self.min_component_ratio = float(config.get("min_component_ratio", 0.02))
        self.mask_erosion_pixels = int(config.get("mask_erosion_pixels", 0))
        self.dbscan_eps_base_m = float(config.get("dbscan_eps_base_m", 0.12))
        self.dbscan_eps_per_depth_m = float(config.get("dbscan_eps_per_depth_m", 0.012))
        self.dbscan_eps_max_m = float(config.get("dbscan_eps_max_m", 0.50))
        self.dbscan_min_points = int(config.get("dbscan_min_points", 3))
        self.min_cluster_points = int(config.get("min_cluster_points", 3))
        self.satellite_max_gap_m = float(config.get("satellite_max_gap_m", 0.30))
        self.bootstrap_secondary_ratio = float(config.get("bootstrap_secondary_ratio", 0.15))
        self.max_center_jump_m = float(config.get("max_center_jump_m", 1.50))
        self.min_center_margin_m = float(config.get("min_center_margin_m", 0.15))
        self.max_track_gap_frames = int(config.get("max_track_gap_frames", 10))
        self.point_radius_pixels = int(config.get("point_radius_pixels", 2))
        if not (
            self.min_component_pixels >= 1
            and 0 <= self.min_component_ratio <= 1
            and self.mask_erosion_pixels >= 0
            and self.dbscan_eps_base_m > 0
            and self.dbscan_eps_per_depth_m >= 0
            and self.dbscan_eps_max_m >= self.dbscan_eps_base_m
            and self.dbscan_min_points >= 1
            and self.min_cluster_points >= self.dbscan_min_points
            and self.satellite_max_gap_m > 0
            and 0 <= self.bootstrap_secondary_ratio <= 1
            and self.max_center_jump_m > 0
            and self.min_center_margin_m >= 0
            and self.max_track_gap_frames >= 0
            and self.point_radius_pixels >= 1
        ):
            raise ValueError("融合参数无效，请检查 fusion_config.yaml")
        self._tracks: dict[tuple[int, int], _TrackState] = {}
        self._frame_index = 0
        self.last_rejections: dict[int, str] = {}

    def reset(self) -> None:
        """目标类别切换后清空旧的三维关联记录。"""
        self._tracks.clear()
        self._frame_index = 0
        self.last_rejections = {}

    def clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """删除主轮廓周围的小孤岛，保留面积足够的断开部位。"""
        binary = (mask > 0).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if count <= 1:
            return binary
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = int(np.argmax(areas)) + 1
        area_limit = max(
            self.min_component_pixels,
            int(areas[largest_label - 1] * self.min_component_ratio),
        )
        keep = [largest_label]
        keep.extend(
            label
            for label in range(1, count)
            if label != largest_label and stats[label, cv2.CC_STAT_AREA] >= area_limit
        )
        cleaned = np.isin(labels, keep).astype(np.uint8)
        if self.mask_erosion_pixels:
            size = 2 * self.mask_erosion_pixels + 1
            cleaned = cv2.erode(cleaned, np.ones((size, size), dtype=np.uint8))
        return cleaned

    def _cluster_indices(
        self, points_lidar: np.ndarray, depth_m: np.ndarray
    ) -> list[np.ndarray]:
        if len(points_lidar) < self.min_cluster_points:
            return []
        epsilon = min(
            self.dbscan_eps_max_m,
            self.dbscan_eps_base_m
            + self.dbscan_eps_per_depth_m * float(np.median(depth_m)),
        )
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points_lidar.astype(np.float64))
        labels = np.asarray(
            cloud.cluster_dbscan(eps=epsilon, min_points=self.dbscan_min_points),
            dtype=np.int32,
        )
        # 先找可靠的密集点团作为种子；再把与种子空间相连的稀疏点纳入目标。
        # 这样突出部位即使被 DBSCAN 标为噪点，也不会在测距前直接丢失。
        seed = np.zeros(len(points_lidar), dtype=bool)
        for label in np.unique(labels):
            if label >= 0 and np.count_nonzero(labels == label) >= self.min_cluster_points:
                seed |= labels == label
        if not np.any(seed):
            return []
        connected = np.asarray(
            cloud.cluster_dbscan(eps=max(epsilon, self.satellite_max_gap_m), min_points=1),
            dtype=np.int32,
        )
        clusters = []
        for label in np.unique(connected):
            if label < 0:
                continue
            indices = np.flatnonzero(connected == label)
            if np.any(seed[indices]):
                clusters.append(indices)
        return sorted(clusters, key=len, reverse=True)

    def _choose_cluster(
        self, clusters: list[np.ndarray], points_lidar: np.ndarray, key: tuple[int, int]
    ) -> np.ndarray | None:
        if not clusters:
            return None
        previous = self._tracks.get(key)
        if previous is None:
            if (
                len(clusters) > 1
                and len(clusters[1]) / len(clusters[0]) >= self.bootstrap_secondary_ratio
            ):
                return None
            return clusters[0]

        centers = np.asarray(
            [np.median(points_lidar[indices], axis=0) for indices in clusters]
        )
        distances = np.linalg.norm(centers - previous.center_lidar, axis=1)
        ordered = np.argsort(distances)
        best = int(ordered[0])
        if distances[best] > self.max_center_jump_m:
            return None
        if (
            len(ordered) > 1
            and distances[ordered[1]] - distances[best] < self.min_center_margin_m
        ):
            return None
        return clusters[best]

    def fuse(
        self, detections: list[Detection], projection: ProjectedCloud | None
    ) -> list[FusedTarget]:
        """从当前帧有效轮廓内提取点，并为每个目标选择一个三维点团。"""
        self._frame_index += 1
        self.last_rejections = {}
        self._tracks = {
            key: state
            for key, state in self._tracks.items()
            if self._frame_index - state.last_frame <= self.max_track_gap_frames
        }
        if projection is None or len(projection.pixels_xy) == 0:
            return []

        image_shape = projection.depth_map.shape
        valid = []
        for detection in detections:
            if not detection.geometry_valid or detection.track_id is None:
                continue
            if detection.mask.shape != image_shape:
                raise ValueError("目标轮廓与点云投影图像尺寸不一致")
            cleaned = self.clean_mask(detection.mask)
            if np.any(cleaned):
                valid.append((detection, cleaned))
        if not valid:
            return []

        pixel_x = projection.pixels_xy[:, 0]
        pixel_y = projection.pixels_xy[:, 1]
        # 多个实例轮廓同时覆盖的像素不能可靠归属，因此不分配给任何目标。
        memberships = np.zeros(len(pixel_x), dtype=np.uint16)
        for _, mask in valid:
            memberships += mask[pixel_y, pixel_x].astype(np.uint16)

        targets = []
        for detection, mask in valid:
            candidate_indices = np.flatnonzero(
                (memberships == 1) & (mask[pixel_y, pixel_x] > 0)
            )
            if len(candidate_indices) < self.min_cluster_points:
                self.last_rejections[detection.track_id] = "点数不足"
                continue
            candidate_xyz = projection.points_lidar[candidate_indices]
            clusters = self._cluster_indices(
                candidate_xyz, projection.depth_m[candidate_indices]
            )
            key = (detection.class_id, detection.track_id)
            chosen = self._choose_cluster(clusters, candidate_xyz, key)
            if chosen is None:
                self.last_rejections[detection.track_id] = (
                    "没有稳定点团" if not clusters else "多个点团无法确定归属"
                )
                continue
            indices = candidate_indices[chosen]
            target = FusedTarget(
                track_id=detection.track_id,
                class_id=detection.class_id,
                label=detection.label,
                category=detection.category,
                pixels_xy=projection.pixels_xy[indices],
                points_lidar=projection.points_lidar[indices],
                points_camera=projection.points_camera[indices],
                depth_m=projection.depth_m[indices],
            )
            targets.append(target)
            self._tracks[key] = _TrackState(
                center_lidar=np.median(target.points_lidar, axis=0),
                last_frame=self._frame_index,
            )
        return targets

    def draw_targets(self, image: np.ndarray, targets: list[FusedTarget]) -> np.ndarray:
        """只绘制完成归属的目标点：危险物红色，被监测物绿色。"""
        result = image.copy()
        colors = {
            DANGEROUS_CATEGORY: (0, 0, 255),
            MONITORED_CATEGORY: (0, 255, 0),
        }
        for target in targets:
            color = colors.get(target.category, (255, 255, 255))
            for x, y in target.pixels_xy:
                cv2.circle(
                    result,
                    (int(x), int(y)),
                    self.point_radius_pixels,
                    color,
                    -1,
                    cv2.LINE_AA,
                )
        return result
