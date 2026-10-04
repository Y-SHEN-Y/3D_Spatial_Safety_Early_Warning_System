"""计算被检测目标与危险物可见点云之间的三维最近点对。"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import yaml

from Detector import DANGEROUS_CATEGORY, MONITORED_CATEGORY
from Fusion import FusedTarget


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "distance_config.yaml"


@dataclass(frozen=True)
class DistanceMeasurement:
    monitored_class_id: int
    monitored_track_id: int
    dangerous_class_id: int
    dangerous_track_id: int
    monitored_label: str
    dangerous_label: str
    distance_m: float
    monitored_point_lidar: tuple[float, float, float]
    dangerous_point_lidar: tuple[float, float, float]
    monitored_pixel_xy: tuple[int, int]
    dangerous_pixel_xy: tuple[int, int]


class SpatialDistance:
    def __init__(self, config_path: str | Path = DEFAULT_CONFIG_PATH):
        with Path(config_path).open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        if not isinstance(config, dict):
            raise ValueError("测距配置必须是 YAML 映射")
        self.safety_distance_m = float(config.get("safety_distance_m", 1.0))
        self.marker_radius_pixels = int(config.get("marker_radius_pixels", 6))
        self.line_thickness = int(config.get("line_thickness", 2))
        if min(self.marker_radius_pixels, self.line_thickness) < 1:
            raise ValueError("测距显示参数必须大于 0")
        self.set_safety_distance(self.safety_distance_m)

    def set_safety_distance(self, value: float) -> None:
        """实时更新安全距离，单位米。"""
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("安全距离必须是大于 0 的有限米数")
        self.safety_distance_m = value

    def safety_status(self, measurements: list[DistanceMeasurement]) -> str:
        """有一组距离达到警戒阈值就报警；没有有效测距时状态未知。"""
        if not measurements:
            return "UNKNOWN"
        if any(item.distance_m <= self.safety_distance_m for item in measurements):
            return "WARNING"
        return "SAFE"

    @staticmethod
    def _check_target(target: FusedTarget) -> bool:
        """空点云表示本帧不能测距；不一致或无效坐标应立即报错。"""
        points = np.asarray(target.points_lidar)
        pixels = np.asarray(target.pixels_xy)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("目标三维点必须是 Nx3 数组")
        if pixels.ndim != 2 or pixels.shape[1] != 2 or len(pixels) != len(points):
            raise ValueError("目标投影像素与三维点不对应")
        if not np.all(np.isfinite(points)):
            raise ValueError("目标三维点包含无效数值")
        return len(points) > 0

    def measure(
        self, targets: list[FusedTarget],
        pair_keys: set[tuple[int, int, int, int]] | None = None,
    ) -> list[DistanceMeasurement]:
        """返回两团可见点云中的最近点对；可限定为指定监测对。"""
        monitored = [target for target in targets if target.category == MONITORED_CATEGORY]
        dangerous = [target for target in targets if target.category == DANGEROUS_CATEGORY]
        monitored = [target for target in monitored if self._check_target(target)]
        dangerous = [target for target in dangerous if self._check_target(target)]
        measurements = []
        for hazard in dangerous:
            hazard_cloud = o3d.geometry.PointCloud()
            hazard_cloud.points = o3d.utility.Vector3dVector(
                np.asarray(hazard.points_lidar, dtype=np.float64)
            )
            hazard_tree = o3d.geometry.KDTreeFlann(hazard_cloud)
            for subject in monitored:
                if pair_keys is not None and (
                    subject.class_id, subject.track_id,
                    hazard.class_id, hazard.track_id,
                ) not in pair_keys:
                    continue
                best_distance_squared = float("inf")
                best_subject_index = -1
                best_hazard_index = -1
                for subject_index, point in enumerate(subject.points_lidar):
                    count, indices, squared_distances = hazard_tree.search_knn_vector_3d(
                        np.asarray(point, dtype=np.float64), 1
                    )
                    if count and squared_distances[0] < best_distance_squared:
                        best_distance_squared = float(squared_distances[0])
                        best_subject_index = subject_index
                        best_hazard_index = int(indices[0])
                if best_subject_index < 0:
                    continue
                subject_point = subject.points_lidar[best_subject_index]
                hazard_point = hazard.points_lidar[best_hazard_index]
                subject_pixel = subject.pixels_xy[best_subject_index]
                hazard_pixel = hazard.pixels_xy[best_hazard_index]
                measurements.append(
                    DistanceMeasurement(
                        monitored_class_id=subject.class_id,
                        monitored_track_id=subject.track_id,
                        dangerous_class_id=hazard.class_id,
                        dangerous_track_id=hazard.track_id,
                        monitored_label=subject.label,
                        dangerous_label=hazard.label,
                        distance_m=float(np.sqrt(best_distance_squared)),
                        monitored_point_lidar=tuple(float(x) for x in subject_point),
                        dangerous_point_lidar=tuple(float(x) for x in hazard_point),
                        monitored_pixel_xy=tuple(int(x) for x in subject_pixel),
                        dangerous_pixel_xy=tuple(int(x) for x in hazard_pixel),
                    )
                )
        return sorted(measurements, key=lambda measurement: measurement.distance_m)

    def draw_measurements(
        self, image: np.ndarray, measurements: list[DistanceMeasurement]
    ) -> np.ndarray:
        """绘制传入的目标对距离线；mode1 传全部，mode2 只传选定配对。"""
        annotated = image.copy()
        for measurement in sorted(measurements, key=lambda item: item.distance_m):
            start = measurement.monitored_pixel_xy
            end = measurement.dangerous_pixel_xy
            # 每一组目标独立判断线色，避免远处安全目标也显示为警报红色。
            color = (
                (0, 0, 255)
                if measurement.distance_m <= self.safety_distance_m
                else (0, 255, 0)
            )
            cv2.line(annotated, start, end, color, self.line_thickness, cv2.LINE_AA)
            label = (
                f"#{measurement.monitored_track_id} - #{measurement.dangerous_track_id}"
                f"  {measurement.distance_m:.3f} m"
            )
            text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            x = max(0, min((start[0] + end[0]) // 2, annotated.shape[1] - text_size[0]))
            y = min(
                annotated.shape[0] - 1,
                max(text_size[1] + baseline, (start[1] + end[1]) // 2),
            )
            cv2.putText(
                annotated, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 0, 0), 4, cv2.LINE_AA,
            )
            cv2.putText(
                annotated, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, color, 2, cv2.LINE_AA,
            )
            # 最近点最后绘制，避免距离文字覆盖端点颜色。
            cv2.circle(annotated, start, self.marker_radius_pixels, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(annotated, end, self.marker_radius_pixels, (0, 0, 255), -1, cv2.LINE_AA)
        return annotated

    def draw_safety_status(
        self, image: np.ndarray, measurements: list[DistanceMeasurement],
        status_override: str | None = None,
    ) -> np.ndarray:
        """在画面左上角显示当前帧总体安全状态。"""
        annotated = image.copy()
        status = status_override or self.safety_status(measurements)
        colors = {"SAFE": (0, 255, 0), "WARNING": (0, 0, 255), "UNKNOWN": (0, 165, 255)}
        label = f"Safety: {status}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size, baseline = cv2.getTextSize(label, font, 0.9, 2)
        cv2.rectangle(
            annotated, (12, 12), (24 + text_size[0], 20 + text_size[1] + baseline),
            (0, 0, 0), -1,
        )
        cv2.putText(
            annotated, label, (18, 16 + text_size[1]), font,
            0.9, colors[status], 2, cv2.LINE_AA,
        )
        return annotated
