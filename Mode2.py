"""管理用户点选的监测对，并从全部测距结果中只保留指定配对。"""

from dataclasses import dataclass

from Detector import DANGEROUS_CATEGORY, MONITORED_CATEGORY, Detection
from SpatialDistance import DistanceMeasurement


@dataclass(frozen=True)
class SelectedTarget:
    class_id: int
    track_id: int
    label: str

    @property
    def key(self) -> tuple[int, int]:
        return self.class_id, self.track_id


@dataclass(frozen=True)
class MonitoringPair:
    pair_id: int
    monitored: SelectedTarget
    dangerous: SelectedTarget


class Mode2Pairs:
    def __init__(self):
        self.pairs: list[MonitoringPair] = []
        self.stage = "idle"
        self.pending_monitored: SelectedTarget | None = None
        self._next_pair_id = 1

    @property
    def expected_category(self) -> str | None:
        return {
            "monitored": MONITORED_CATEGORY,
            "dangerous": DANGEROUS_CATEGORY,
        }.get(self.stage)

    def start_add(self) -> None:
        """开始新配对；先点被检测目标，再点危险物。"""
        self.stage = "monitored"
        self.pending_monitored = None

    def cancel_add(self) -> None:
        self.stage = "idle"
        self.pending_monitored = None

    def clear(self) -> None:
        """类别配置改变后，旧 ID 的含义不再可靠。"""
        self.pairs.clear()
        self.cancel_add()
        self._next_pair_id = 1

    def candidates_at(
        self, detections: list[Detection], x: int, y: int
    ) -> list[Detection]:
        """按当前步骤和实例 mask 命中目标；框重叠本身不会造成歧义。"""
        category = self.expected_category
        if category is None:
            return []
        candidates = []
        for detection in detections:
            if (
                detection.category != category
                or detection.track_id is None
                or not detection.geometry_valid
            ):
                continue
            mask = detection.display_mask if detection.display_mask is not None else detection.mask
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x] > 0:
                candidates.append(detection)
        return candidates

    def choose(self, detection: Detection) -> MonitoringPair | None:
        """选中当前步骤的目标；选完危险物后完成一组监测对。"""
        if self.expected_category is None:
            raise ValueError("请先点击“添加监测对”")
        if detection.category != self.expected_category or detection.track_id is None:
            raise ValueError("点击的目标类别与当前选择步骤不符")
        if not detection.geometry_valid:
            raise ValueError("该目标当前轮廓无效，请等待有效检测帧")
        selected = SelectedTarget(detection.class_id, detection.track_id, detection.label)
        if self.stage == "monitored":
            self.pending_monitored = selected
            self.stage = "dangerous"
            return None
        if self.pending_monitored is None:
            raise RuntimeError("缺少被检测目标，无法完成配对")
        if any(
            pair.monitored.key == self.pending_monitored.key
            and pair.dangerous.key == selected.key
            for pair in self.pairs
        ):
            raise ValueError("该监测对已经存在")
        pair = MonitoringPair(self._next_pair_id, self.pending_monitored, selected)
        self._next_pair_id += 1
        self.pairs.append(pair)
        self.cancel_add()
        return pair

    def remove(self, pair_ids: list[int]) -> None:
        ids = set(pair_ids)
        self.pairs = [pair for pair in self.pairs if pair.pair_id not in ids]

    def selected_keys(self) -> set[tuple[int, int]]:
        return {
            target.key
            for pair in self.pairs
            for target in (pair.monitored, pair.dangerous)
        }

    def pair_keys(self) -> set[tuple[int, int, int, int]]:
        """只让测距模块计算用户明确选中的目标组合。"""
        return {
            (
                pair.monitored.class_id,
                pair.monitored.track_id,
                pair.dangerous.class_id,
                pair.dangerous.track_id,
            )
            for pair in self.pairs
        }

    def match_measurements(
        self, measurements: list[DistanceMeasurement]
    ) -> dict[int, DistanceMeasurement]:
        """忽略两团已选目标之间未被用户配成一组的交叉组合。"""
        by_key = {
            (
                result.monitored_class_id,
                result.monitored_track_id,
                result.dangerous_class_id,
                result.dangerous_track_id,
            ): result
            for result in measurements
        }
        return {
            pair.pair_id: by_key[key]
            for pair in self.pairs
            if (key := (
                pair.monitored.class_id,
                pair.monitored.track_id,
                pair.dangerous.class_id,
                pair.dangerous.track_id,
            )) in by_key
        }

    def safety_status(
        self, matched: dict[int, DistanceMeasurement], safety_distance_m: float
    ) -> str:
        """有警报则报警；若任一配对本帧缺测，不能宣布整体安全。"""
        if any(item.distance_m <= safety_distance_m for item in matched.values()):
            return "WARNING"
        if not self.pairs or len(matched) < len(self.pairs):
            return "UNKNOWN"
        return "SAFE"
