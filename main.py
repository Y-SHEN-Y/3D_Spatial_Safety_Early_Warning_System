"""相机、检测和雷达投影模块的运行入口。"""

from pathlib import Path

import cv2
import numpy as np

from Converter import Converter
from Detector import Detector
from Fusion import Fusion
from GroundFilter import GroundFilter
from Mode2 import Mode2Pairs
from Point_Cloud import PointCloudSubscriber
from SpatialDistance import SpatialDistance


PROJECT_DIR = Path(__file__).resolve().parent
CAMERA_CONFIG = PROJECT_DIR / "bin_cam_config.yaml"


def main() -> None:
    # 硬件和界面只在真正运行主程序时加载。
    from Capture import Capture
    from VisualUI import VisualUI

    camera = None
    point_cloud = None
    ui = None
    annotated = None
    depth_map = None
    latest_detections = []
    mode = "mode1"
    try:
        detector = Detector()
        if not detector.monitored_names or not detector.dangerous_names:
            raise ValueError("可视化界面需要配置至少一个被检测目标和一个危险物类别")
        detector.set_target_categories(
            detector.monitored_names, detector.dangerous_names
        )
        fusion = Fusion()
        spatial_distance = SpatialDistance()
        mode2_pairs = Mode2Pairs()
        camera = Capture(binocular_camera_cfg_path=str(CAMERA_CONFIG), camera_name="right")
        point_cloud = PointCloudSubscriber(topic="/cloudpoints", queue_size=3)
        ground_filter = GroundFilter()
        converter = Converter(
            point_cloud, config_path=CAMERA_CONFIG, ground_filter=ground_filter
        )
        point_cloud.start()

        def change_categories(monitored_labels: list[str], dangerous_labels: list[str]) -> None:
            nonlocal annotated, depth_map, latest_detections
            if (
                detector.monitored_names == monitored_labels
                and detector.dangerous_names == dangerous_labels
            ):
                return
            detector.set_target_categories(monitored_labels, dangerous_labels)
            fusion.reset()
            mode2_pairs.clear()
            latest_detections = []
            annotated = None
            depth_map = None
            ui.clear_frame()
            if mode == "mode2":
                ui.set_pair_prompt("类别已改变，旧监测对已清空；请重新添加")
                ui.show_mode2_pairs([], {}, spatial_distance.safety_distance_m, "UNKNOWN")

        def change_mode(selected_mode: str) -> None:
            nonlocal mode, annotated, depth_map, latest_detections
            if selected_mode not in ("mode1", "mode2"):
                raise ValueError("未知测距模式")
            mode = selected_mode
            mode2_pairs.cancel_add()
            fusion.reset()
            annotated = None
            depth_map = None
            latest_detections = []
            ui.clear_frame()

        def add_pair() -> None:
            mode2_pairs.start_add()

        def cancel_pair() -> None:
            mode2_pairs.cancel_add()

        def delete_pairs(pair_ids: list[int]) -> None:
            mode2_pairs.remove(pair_ids)
            if not mode2_pairs.pairs:
                fusion.reset()

        def choose_target(detection) -> None:
            try:
                pair = mode2_pairs.choose(detection)
            except ValueError as exc:
                ui.show_message(str(exc), error=True)
                return
            if pair is None:
                ui.set_pair_prompt(
                    f"已选 {detection.label} #{detection.track_id}；请点击危险物轮廓"
                )
            else:
                ui.set_pair_prompt(
                    f"已添加监测对 #{pair.pair_id}；可继续添加或删除"
                )
                ui.show_message(f"监测对 #{pair.pair_id} 已开始测距")

        def target_clicked(x: int, y: int, screen_x: int, screen_y: int) -> None:
            if mode2_pairs.expected_category is None:
                ui.show_message("请先点击“添加监测对”", error=True)
                return
            candidates = mode2_pairs.candidates_at(latest_detections, x, y)
            if not candidates:
                ui.show_message("未点中当前有效目标轮廓，请点击目标内部", error=True)
                return
            ui.choose_detection(candidates, screen_x, screen_y, choose_target)

        def save_current_frame() -> bool:
            if annotated is None:
                return False
            image_path = PROJECT_DIR / "segmentation_result.jpg"
            cv2.imwrite(str(image_path), annotated)
            print(f"已保存 {image_path}")
            if depth_map is not None:
                depth_path = PROJECT_DIR / "depth_projection.npy"
                np.save(depth_path, depth_map)
                print(f"已保存 {depth_path}（单位：米；0 表示没有点）")
            return True

        ui = VisualUI(
            labels=detector.available_labels,
            monitored_labels=detector.monitored_names,
            dangerous_labels=detector.dangerous_names,
            safety_distance_m=spatial_distance.safety_distance_m,
            on_distance_change=spatial_distance.set_safety_distance,
            on_categories_change=change_categories,
            on_save=save_current_frame,
            on_mode_change=change_mode,
            on_add_pair=add_pair,
            on_cancel_pair=cancel_pair,
            on_delete_pairs=delete_pairs,
            on_target_click=target_clicked,
        )

        def process_frame() -> None:
            nonlocal annotated, depth_map, latest_detections
            frame = camera.get_frame()
            if frame is None:
                return
            detections = detector.detect(frame)
            latest_detections = detections
            if mode == "mode2":
                annotated = detector.draw_masks(frame, detections, compact=True)
                if not mode2_pairs.pairs:
                    depth_map = None
                    ui.show_mode2_pairs([], {}, spatial_distance.safety_distance_m, "UNKNOWN")
                    ui.show_frame(annotated)
                    return
                projection = converter.project_latest_cloud(frame.shape[:2])
                depth_map = None if projection is None else projection.depth_map
                selected_keys = mode2_pairs.selected_keys()
                # 点云归属仍检查所有轮廓，避免未选目标的重叠点误归给已选目标。
                all_targets = fusion.fuse(detections, projection)
                targets = [
                    target for target in all_targets
                    if (target.class_id, target.track_id) in selected_keys
                ]
                selected_measurements = spatial_distance.measure(
                    targets, mode2_pairs.pair_keys()
                )
                matched = mode2_pairs.match_measurements(selected_measurements)
                measurements = list(matched.values())
                annotated = fusion.draw_targets(annotated, targets)
                annotated = spatial_distance.draw_measurements(annotated, measurements)
                status = mode2_pairs.safety_status(matched, spatial_distance.safety_distance_m)
                annotated = spatial_distance.draw_safety_status(
                    annotated, measurements, status_override=status
                )
                ui.show_mode2_pairs(
                    mode2_pairs.pairs, matched, spatial_distance.safety_distance_m, status
                )
                ui.show_frame(annotated)
                return

            projection = converter.project_latest_cloud(frame.shape[:2])
            depth_map = None if projection is None else projection.depth_map
            targets = fusion.fuse(detections, projection)
            measurements = spatial_distance.measure(targets)
            annotated = fusion.draw_targets(frame, targets)
            annotated = detector.draw_masks(annotated, detections)
            annotated = spatial_distance.draw_measurements(annotated, measurements)
            # 现场调参时可直接看出本帧是拟合成功、沿用旧平面，还是没有平面。
            ground_state = (
                f"Ground: {converter.last_ground_status}  removed: {converter.last_ground_count}"
                if depth_map is not None
                else "LiDAR: no recent cloud"
            )
            cv2.putText(
                annotated,
                ground_state,
                (16, annotated.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                f"Target clouds: {len(targets)}  unassigned: {len(fusion.last_rejections)}",
                (16, annotated.shape[0] - 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            distance_text = (
                f"Closest 3D distance: {measurements[0].distance_m:.3f} m"
                if measurements else "Closest 3D distance: unavailable"
            )
            cv2.putText(
                annotated, distance_text, (16, annotated.shape[0] - 76),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA,
            )
            status = spatial_distance.safety_status(measurements)
            annotated = spatial_distance.draw_safety_status(annotated, measurements)
            ui.show_frame(annotated)
            ui.show_measurements(measurements, spatial_distance.safety_distance_m, status)

        ui.run(process_frame)
    except KeyboardInterrupt:
        pass
    finally:
        if ui is not None:
            ui.close()
        if point_cloud is not None:
            point_cloud.close()
        if camera is not None:
            camera.release()


if __name__ == "__main__":
    main()
