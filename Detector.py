"""读取配置，用 COCO 实例分割模型检测、跟踪并筛选目标轮廓。"""

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "detector_config.yaml"

MONITORED_CATEGORY = "monitored"
DANGEROUS_CATEGORY = "dangerous"


@dataclass
class Detection:
    """单个目标；mask 仅包含当前帧可用于后续处理的像素。"""

    class_id: int
    label: str
    category: str  # monitored 或 dangerous
    track_id: int | None  # 跨帧目标编号；未跟踪时为 None
    confidence: float
    box_xyxy: tuple[int, int, int, int]
    mask: np.ndarray  # uint8，与输入图像同尺寸；无有效测距轮廓时全为 0
    geometry_valid: bool = True
    status: str = "current"  # current、weak 或 held
    display_mask: np.ndarray | None = None  # 仅用于显示，不用于点云提取


@dataclass
class _TrackState:
    """保存单个 track_id 的确认状态与最近的有效轮廓。"""

    class_id: int
    label: str
    category: str
    hits: int = 0
    misses: int = 0
    confirmed: bool = False
    last_mask: np.ndarray | None = None
    last_box: tuple[int, int, int, int] | None = None
    display_mask: np.ndarray | None = None
    history: list[tuple[np.ndarray, tuple[int, int, int, int]]] = field(default_factory=list)


class Detector:
    def __init__(self, config_path: str | Path | None = None):
        config_path = Path(config_path).expanduser().resolve() if config_path else DEFAULT_CONFIG_PATH
        config = self._load_config(config_path)

        weights_path = Path(str(config.get("weights", ""))).expanduser()
        if not weights_path.is_absolute():
            weights_path = config_path.parent / weights_path
        weights_path = weights_path.resolve()
        if not weights_path.is_file():
            raise FileNotFoundError(f"YOLO26 segmentation weights not found: {weights_path}")

        confidence = float(config.get("confidence", 0.25))
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")

        track_conf = float(config.get("track_conf", 0.1))
        if not 0 <= track_conf <= 1:
            raise ValueError("track_conf must be between 0 and 1")
        self.track_conf = track_conf

        self.tracker = str(config.get("tracker", "bytetrack.yaml"))

        lifecycle = config.get("lifecycle") or {}
        mask_quality = config.get("mask_quality") or {}
        if not isinstance(lifecycle, dict) or not isinstance(mask_quality, dict):
            raise ValueError("lifecycle 和 mask_quality 必须是 YAML 映射")
        self.min_confirmed_frames = int(lifecycle.get("min_confirmed_frames", 2))
        self.max_hold_frames = int(lifecycle.get("max_hold_frames", 3))
        self.start_confidence = float(lifecycle.get("start_confidence", 0.35))
        self.hold_confidence = float(lifecycle.get("hold_confidence", 0.16))
        self.min_area_pixels = int(mask_quality.get("min_area_pixels", 32))
        self.min_mask_iou = float(mask_quality.get("min_mask_iou", 0.3))
        self.min_area_ratio = float(mask_quality.get("min_area_ratio", 0.5))
        self.max_area_ratio = float(mask_quality.get("max_area_ratio", 2.0))
        self.temporal_window = int(mask_quality.get("temporal_window", 3))
        if self.min_confirmed_frames < 1 or self.max_hold_frames < 0:
            raise ValueError("确认帧数须大于 0，暂存帧数不能为负")
        if not 0 <= self.track_conf <= self.hold_confidence <= self.start_confidence <= 1:
            raise ValueError("置信度应满足 track_conf <= hold_confidence <= start_confidence <= 1")
        if self.min_area_pixels < 1 or not 0 <= self.min_mask_iou <= 1:
            raise ValueError("最小 mask 面积须大于 0，最小 IoU 须在 0～1 之间")
        if not 0 < self.min_area_ratio <= 1 <= self.max_area_ratio:
            raise ValueError("面积比例须满足 0 < min_area_ratio <= 1 <= max_area_ratio")
        if self.temporal_window < 1 or self.temporal_window % 2 != 1:
            raise ValueError("temporal_window 须为正奇数")
        self._tracks: dict[int, _TrackState] = {}
        self._frame_shape: tuple[int, int] | None = None

        targets = config.get("targets") or {}
        self.monitored_names = [str(name) for name in targets.get(MONITORED_CATEGORY, [])]
        self.dangerous_names = [str(name) for name in targets.get(DANGEROUS_CATEGORY, [])]
        self.target_names = self.monitored_names + self.dangerous_names
        if not self.target_names:
            raise ValueError("detector config must define at least one monitored or dangerous target")

        self.model = YOLO(str(weights_path))
        if self.model.task != "segment":
            raise ValueError(f"Expected segmentation weights, got task: {self.model.task}")

        names = self.model.names
        missing = [name for name in self.target_names if name not in names.values()]
        if missing:
            raise ValueError(f"Weights do not contain the configured classes: {missing}")
        self.available_labels = self._load_available_labels(config_path, config, names)
        unavailable = [name for name in self.target_names if name not in self.available_labels]
        if unavailable:
            raise ValueError(f"当前目标不在可选 label 中: {unavailable}")
        self.target_ids = [
            class_id for class_id, name in names.items() if name in self.target_names
        ]
        self.confidence = confidence
        if self.hold_confidence > self.confidence:
            raise ValueError("hold_confidence 不能高于用于测距的 confidence")

    @staticmethod
    def _load_config(config_path: Path) -> dict:
        if not config_path.is_file():
            raise FileNotFoundError(f"Detector config not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
        if not isinstance(config, dict):
            raise ValueError(f"Invalid detector config (expected a mapping): {config_path}")
        return config

    @staticmethod
    def _load_available_labels(config_path: Path, config: dict, model_names: dict) -> list[str]:
        """从 YAML 的 names 或 label 列表读取 UI 可选类别，并核对模型权重。"""
        label_file = config.get("label_file")
        if label_file is None:
            return [str(name) for name in model_names.values()]
        label_path = Path(str(label_file)).expanduser()
        if not label_path.is_absolute():
            label_path = config_path.parent / label_path
        with label_path.open("r", encoding="utf-8") as fh:
            label_config = yaml.safe_load(fh)
        if not isinstance(label_config, dict):
            raise ValueError("label 配置必须是 YAML 映射")
        labels = label_config.get("names", label_config.get("labels"))
        if isinstance(labels, dict):
            labels = [labels[key] for key in sorted(labels, key=lambda key: int(key))]
        if not isinstance(labels, list) or not labels:
            raise ValueError("label 配置需要非空的 names 或 labels 列表")
        options = [str(label) for label in labels]
        if len(options) != len(set(options)) or not set(options).issubset(set(model_names.values())):
            raise ValueError("label 配置有重复项或包含当前权重不支持的类别")
        return options

    def set_target_categories(
        self, monitored_labels: str | list[str], dangerous_labels: str | list[str]
    ) -> None:
        """运行时切换两组目标，同时丢弃旧类别的生命周期和追踪状态。"""
        monitored = [monitored_labels] if isinstance(monitored_labels, str) else list(monitored_labels)
        dangerous = [dangerous_labels] if isinstance(dangerous_labels, str) else list(dangerous_labels)
        if not monitored or not dangerous:
            raise ValueError("被检测目标和危险物都至少需要保留一个类别")
        if len(monitored) != len(set(monitored)) or len(dangerous) != len(set(dangerous)):
            raise ValueError("同一组内不能重复选择类别")
        if set(monitored) & set(dangerous):
            raise ValueError("同一类别不能同时作为被检测目标和危险物")
        if any(name not in self.available_labels for name in monitored + dangerous):
            raise ValueError("类别不在 label YAML 的可选列表中")
        if self.monitored_names == monitored and self.dangerous_names == dangerous:
            return
        self.monitored_names = monitored
        self.dangerous_names = dangerous
        self.target_names = monitored + dangerous
        self.target_ids = [
            class_id for class_id, name in self.model.names.items()
            if name in self.target_names
        ]
        self._tracks.clear()
        self._frame_shape = None
        predictor = getattr(self.model, "predictor", None)
        for tracker in (getattr(predictor, "trackers", None) or ()):
            tracker.reset()

    def detect(self, image: np.ndarray) -> list[Detection]:
        """接收 BGR 图像，返回已确认的目标及其当前轮廓状态。"""
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be a BGR array with shape (height, width, 3)")
        if image.dtype != np.uint8 or image.size == 0:
            raise ValueError("image must be a non-empty uint8 array")

        result = self.model.track(
            source=image,
            classes=self.target_ids,
            conf=self.track_conf,
            retina_masks=True,
            persist=True,
            tracker=self.tracker,
            verbose=False,
        )[0]
        if result.boxes is None or result.masks is None:
            return self._update_tracks([], image.shape[:2])

        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()
        masks = result.masks.data.cpu().numpy()
        track_ids = result.boxes.id
        if track_ids is None:
            track_ids = np.full(len(boxes), -1, dtype=int)
        else:
            track_ids = track_ids.cpu().numpy().astype(int)

        if len(boxes) != len(masks):
            raise RuntimeError("Segmentation boxes and masks have different counts")
        if len(boxes) != len(track_ids):
            raise RuntimeError("Tracking IDs and boxes have different counts")

        height, width = image.shape[:2]
        candidates = []
        for box, class_id, score, raw_mask, track_id in zip(
            boxes, classes, confidences, masks, track_ids
        ):
            if class_id not in self.target_ids:
                continue
            label = self.model.names[int(class_id)]
            category = self._category_for(label)
            if category is None:
                continue
            mask = (raw_mask > 0).astype(np.uint8)
            if mask.shape != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            candidates.append(
                Detection(
                    class_id=int(class_id),
                    label=label,
                    category=category,
                    track_id=int(track_id) if track_id >= 0 else None,
                    confidence=float(score),
                    box_xyxy=tuple(int(v) for v in box),
                    mask=mask,
                )
            )
        return self._update_tracks(candidates, (height, width))

    @staticmethod
    def _align_mask(
        mask: np.ndarray,
        old_box: tuple[int, int, int, int],
        new_box: tuple[int, int, int, int],
    ) -> np.ndarray:
        """按目标框的位置和尺寸，把上一帧轮廓对齐到当前帧。"""
        old_width = max(old_box[2] - old_box[0], 1)
        old_height = max(old_box[3] - old_box[1], 1)
        scale_x = max(new_box[2] - new_box[0], 1) / old_width
        scale_y = max(new_box[3] - new_box[1], 1) / old_height
        matrix = np.float32(
            [
                [scale_x, 0, new_box[0] - old_box[0] * scale_x],
                [0, scale_y, new_box[1] - old_box[1] * scale_y],
            ]
        )
        height, width = mask.shape
        return cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_NEAREST)

    def _mask_is_consistent(self, state: _TrackState, candidate: Detection) -> bool:
        """按面积和位置对齐后的重叠度排除突然跳变的轮廓。"""
        current_area = int(np.count_nonzero(candidate.mask))
        if current_area < self.min_area_pixels:
            return False
        if state.last_mask is None or state.last_box is None:
            return True
        previous = self._align_mask(state.last_mask, state.last_box, candidate.box_xyxy)
        previous_area = int(np.count_nonzero(previous))
        if previous_area < self.min_area_pixels:
            return False
        ratio = current_area / previous_area
        if not self.min_area_ratio <= ratio <= self.max_area_ratio:
            return False
        intersection = np.count_nonzero(previous & candidate.mask)
        union = previous_area + current_area - intersection
        return union > 0 and intersection / union >= self.min_mask_iou

    def _smooth_display_mask(self, state: _TrackState, candidate: Detection) -> np.ndarray:
        """在目标框对齐后对最近数帧取多数票，缓和显示轮廓的抖动。"""
        state.history.append((candidate.mask, candidate.box_xyxy))
        state.history = state.history[-self.temporal_window :]
        if len(state.history) < self.temporal_window:
            return candidate.mask
        votes = np.zeros_like(candidate.mask, dtype=np.uint8)
        for mask, box in state.history:
            aligned = mask if box == candidate.box_xyxy else self._align_mask(mask, box, candidate.box_xyxy)
            votes += aligned
        return (votes > self.temporal_window // 2).astype(np.uint8)

    @staticmethod
    def _display_detection(
        candidate: Detection, display_mask: np.ndarray, geometry_mask: np.ndarray, status: str
    ) -> Detection:
        """分别保存当前帧测距轮廓和仅供显示的轮廓。"""
        return Detection(
            class_id=candidate.class_id,
            label=candidate.label,
            category=candidate.category,
            track_id=candidate.track_id,
            confidence=candidate.confidence,
            box_xyxy=candidate.box_xyxy,
            mask=geometry_mask,
            geometry_valid=status == "current",
            status=status,
            display_mask=display_mask,
        )

    def _update_tracks(self, candidates: list[Detection], shape: tuple[int, int]) -> list[Detection]:
        """确认新目标，短时保留丢失目标，并隔离不可用于测距的轮廓。"""
        if getattr(self, "_frame_shape", None) != shape:
            self._tracks.clear()
            self._frame_shape = shape
        output = []
        seen_ids: set[int] = set()
        accepted_ids: set[int] = set()
        empty_mask = np.zeros(shape, dtype=np.uint8)
        for candidate in candidates:
            track_id = candidate.track_id
            if track_id is None or track_id in seen_ids:
                continue
            seen_ids.add(track_id)
            state = self._tracks.get(track_id)
            if state is not None and state.class_id != candidate.class_id:
                del self._tracks[track_id]
                state = None
            if state is None:
                if candidate.confidence < self.start_confidence:
                    continue
                state = _TrackState(candidate.class_id, candidate.label, candidate.category)
                self._tracks[track_id] = state
            if candidate.confidence < self.hold_confidence or not self._mask_is_consistent(state, candidate):
                if not state.confirmed:
                    del self._tracks[track_id]
                continue

            if not state.confirmed:
                if candidate.confidence < self.start_confidence:
                    del self._tracks[track_id]
                    continue
                state.hits += 1
            if state.misses:
                state.history.clear()
            state.misses = 0
            state.last_mask = candidate.mask
            state.last_box = candidate.box_xyxy
            state.display_mask = self._smooth_display_mask(state, candidate)
            state.confirmed = state.hits >= self.min_confirmed_frames
            accepted_ids.add(track_id)
            if not state.confirmed:
                continue

            # 测距轮廓始终与当前帧 mask 相交，避免时间平滑引入旧像素。
            geometry_mask = candidate.mask & state.display_mask
            geometry_valid = (
                candidate.confidence >= self.confidence
                and np.count_nonzero(geometry_mask) >= self.min_area_pixels
            )
            status = "current" if geometry_valid else "weak"
            output.append(
                self._display_detection(
                    candidate,
                    state.display_mask,
                    geometry_mask if geometry_valid else empty_mask.copy(),
                    status,
                )
            )

        for track_id, state in list(self._tracks.items()):
            if track_id in accepted_ids:
                continue
            if not state.confirmed:
                del self._tracks[track_id]
                continue
            state.misses += 1
            if state.misses > self.max_hold_frames:
                del self._tracks[track_id]
                continue
            if state.display_mask is None or state.last_box is None:
                continue
            # 暂存轮廓只供画面提示；此帧没有可用于点云提取的 mask。
            held = Detection(
                class_id=state.class_id,
                label=state.label,
                category=state.category,
                track_id=track_id,
                confidence=0.0,
                box_xyxy=state.last_box,
                mask=empty_mask.copy(),
                geometry_valid=False,
                status="held",
                display_mask=state.display_mask,
            )
            output.append(held)
        return output

    def _category_for(self, label: str) -> str | None:
        """把类别名称映射到监视对象或危险物类别。"""
        if label in self.monitored_names:
            return MONITORED_CATEGORY
        if label in self.dangerous_names:
            return DANGEROUS_CATEGORY
        return None

    @staticmethod
    def draw_masks(
        image: np.ndarray, detections: list[Detection], compact: bool = False
    ) -> np.ndarray:
        """在图像副本上画出轮廓与编号；精简模式只显示类别和 ID。"""
        annotated = image.copy()
        colors = {
            MONITORED_CATEGORY: (0, 255, 0),  # green: 监视对象
            DANGEROUS_CATEGORY: (0, 0, 255),  # red: 危险物
        }
        for detection in detections:
            color = colors.get(detection.category, (255, 255, 255))
            if detection.status == "held":
                color = (160, 160, 160)
            elif detection.status == "weak":
                color = (0, 165, 255)
            draw_mask = detection.display_mask if detection.display_mask is not None else detection.mask
            contours, _ = cv2.findContours(draw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated, contours, -1, color, 2)
            x1, y1, _, _ = detection.box_xyxy
            track_part = f"#{detection.track_id}" if detection.track_id is not None else ""
            label = f"{detection.label}{track_part}"
            if not compact:
                label += f" {detection.confidence:.2f} {detection.status}"
            cv2.putText(
                annotated,
                label,
                (max(x1, 0), max(y1 - 8, 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
        return annotated

    def process_frame(self, image: np.ndarray) -> np.ndarray:
        """检测并标注一帧 BGR 图像。"""
        detections = self.detect(image)
        return self.draw_masks(image, detections)

    def run(
        self,
        capture,
        window_name: str = "YOLO26 COCO instance segmentation",
        output_path: str | Path | None = None,
    ) -> None:
        """持续获取相机图像并显示；按 q 退出，按 s 保存当前标注帧。"""
        if output_path is None:
            output_path = Path("segmentation_result.jpg")
        else:
            output_path = Path(output_path)

        print("Detecting person and laptop. Press q to quit, s to save a frame.")
        annotated = None
        try:
            while True:
                frame = capture.get_frame()
                if frame is not None:
                    annotated = self.process_frame(frame)
                    cv2.imshow(window_name, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s") and annotated is not None:
                    cv2.imwrite(str(output_path), annotated)
                    print(f"Saved: {output_path}")
        except KeyboardInterrupt:
            pass
        finally:
            capture.release()
            cv2.destroyAllWindows()
