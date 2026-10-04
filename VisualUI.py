"""显示实时视频、全部目标对距离和运行时可调控件。"""

import tkinter as tk
from tkinter import font as tkfont, ttk
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageTk

from Detector import Detection
from Mode2 import MonitoringPair
from SpatialDistance import DistanceMeasurement


class VisualUI:
    def __init__(
        self,
        labels: list[str],
        monitored_labels: list[str],
        dangerous_labels: list[str],
        safety_distance_m: float,
        on_distance_change: Callable[[float], None],
        on_categories_change: Callable[[list[str], list[str]], None],
        on_save: Callable[[], bool],
        on_mode_change: Callable[[str], None],
        on_add_pair: Callable[[], None],
        on_cancel_pair: Callable[[], None],
        on_delete_pairs: Callable[[list[int]], None],
        on_target_click: Callable[[int, int, int, int], None],
    ):
        self.root = tk.Tk()
        self.root.title("三维安全预警系统")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._closed = False
        self._error: Exception | None = None
        self._image = None
        self._on_distance_change = on_distance_change
        self._on_categories_change = on_categories_change
        self._on_save = on_save
        self._on_mode_change = on_mode_change
        self._on_add_pair = on_add_pair
        self._on_cancel_pair = on_cancel_pair
        self._on_delete_pairs = on_delete_pairs
        self._on_target_click = on_target_click
        self._selected_monitored = list(monitored_labels)
        self._selected_dangerous = list(dangerous_labels)
        self._source_shape: tuple[int, int] | None = None
        self._display_size: tuple[int, int] | None = None

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self._panel_width = 420 if screen_width >= 1300 else min(380, max(330, screen_width // 2))
        self._video_width = max(320, min(1440, screen_width - self._panel_width - 60))
        self._video_height = max(300, min(1080, screen_height - 100))
        self.root.geometry(f"{self._video_width + self._panel_width + 40}x{self._video_height + 40}")
        self.root.minsize(min(850, screen_width - 20), min(500, screen_height - 20))

        # 配置栏统一使用约 1.5 倍于原来的字号，控件和表格行高同步调整。
        self._control_font = tkfont.Font(root=self.root, family="Sans", size=15)
        title_font = tkfont.Font(root=self.root, family="Sans", size=24, weight="bold")
        style = ttk.Style(self.root)
        style.configure("Alert.TButton", font=self._control_font, padding=(8, 5))
        style.configure("Alert.TEntry", padding=(6, 4))
        style.configure("Alert.TCombobox", padding=(6, 4))
        style.configure("Alert.Treeview", font=self._control_font, rowheight=34)
        style.configure("Alert.Treeview.Heading", font=self._control_font)
        self.root.option_add("*TCombobox*Listbox.font", self._control_font)

        container = ttk.Frame(self.root, padding=12)
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)

        self.video_label = tk.Label(container, bg="black", text="等待相机画面…", fg="white")
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        self.video_label.bind("<Button-1>", self._video_clicked)
        panel_holder = ttk.Frame(container, width=self._panel_width)
        panel_holder.grid(row=0, column=1, sticky="ns")
        panel_holder.pack_propagate(False)
        panel_canvas = tk.Canvas(panel_holder, highlightthickness=0, borderwidth=0)
        panel_scrollbar = ttk.Scrollbar(panel_holder, orient="vertical", command=panel_canvas.yview)
        panel_canvas.configure(yscrollcommand=panel_scrollbar.set)
        panel_canvas.pack(side="left", fill="both", expand=True)
        panel_scrollbar.pack(side="right", fill="y")
        panel = ttk.Frame(panel_canvas, padding=(0, 0, 8, 8))
        panel_window = panel_canvas.create_window((0, 0), window=panel, anchor="nw")
        panel.bind("<Configure>", lambda _event: panel_canvas.configure(scrollregion=panel_canvas.bbox("all")))
        panel_canvas.bind(
            "<Configure>",
            lambda event: panel_canvas.itemconfigure(panel_window, width=event.width),
        )

        ttk.Label(panel, text="实时控制", font=title_font).pack(anchor="w", pady=(2, 16))
        ttk.Label(panel, text="安全距离（米）", font=self._control_font).pack(anchor="w")
        self.distance_var = tk.StringVar(value=f"{safety_distance_m:g}")
        ttk.Entry(
            panel, textvariable=self.distance_var, font=self._control_font,
            style="Alert.TEntry",
        ).pack(fill="x", pady=(4, 4))
        self.distance_var.trace_add("write", self._distance_changed)

        self.monitored_add_var = tk.StringVar()
        self.monitored_listbox = self._build_category_controls(
            panel, "被检测目标类别", labels, self.monitored_add_var, "monitored"
        )
        self.dangerous_add_var = tk.StringVar()
        self.dangerous_listbox = self._build_category_controls(
            panel, "危险物类别", labels, self.dangerous_add_var, "dangerous"
        )
        self._sync_category_lists()

        ttk.Label(panel, text="测距模式", font=self._control_font).pack(anchor="w", pady=(14, 0))
        self.mode_var = tk.StringVar(value="mode1")
        self._active_mode = "mode1"
        mode_combo = ttk.Combobox(
            panel, textvariable=self.mode_var,
            values=("mode1", "mode2"), state="readonly",
            font=self._control_font, style="Alert.TCombobox",
        )
        mode_combo.pack(fill="x", pady=(4, 0))
        mode_combo.bind("<<ComboboxSelected>>", self._mode_changed)
        ttk.Label(
            panel, text="mode1：全部目标对；mode2：手动选配对",
            font=self._control_font,
        ).pack(anchor="w", pady=(4, 0))

        self.mode2_controls = ttk.Frame(panel)
        ttk.Button(
            self.mode2_controls, text="添加监测对", style="Alert.TButton",
            command=self._add_pair,
        ).pack(fill="x", pady=(10, 4))
        ttk.Button(
            self.mode2_controls, text="取消添加", style="Alert.TButton",
            command=self._cancel_pair,
        ).pack(fill="x", pady=(0, 4))
        self.pair_prompt_var = tk.StringVar(value="点击“添加监测对”开始")
        tk.Label(
            self.mode2_controls, textvariable=self.pair_prompt_var,
            font=self._control_font, anchor="w", justify="left",
            wraplength=self._panel_width - 40,
        ).pack(fill="x", pady=(4, 4))
        ttk.Label(
            self.mode2_controls, text="已添加的监测对", font=self._control_font
        ).pack(anchor="w")
        pair_table_frame = ttk.Frame(self.mode2_controls)
        pair_table_frame.pack(fill="x", pady=(4, 4))
        self.pair_table = ttk.Treeview(
            pair_table_frame, columns=("pair", "distance"),
            show="headings", height=5, selectmode="extended", style="Alert.Treeview",
        )
        self.pair_table.heading("pair", text="对象 → 危险物")
        self.pair_table.heading("distance", text="距离/m")
        self.pair_table.column("pair", width=270, anchor="w", stretch=False)
        self.pair_table.column("distance", width=90, anchor="center", stretch=False)
        self.pair_table.tag_configure("safe", foreground="#007a25")
        self.pair_table.tag_configure("warning", foreground="#c00000")
        self.pair_table.pack(side="left", fill="x", expand=True)
        pair_scrollbar = ttk.Scrollbar(pair_table_frame, command=self.pair_table.yview)
        pair_scrollbar.pack(side="right", fill="y")
        self.pair_table.configure(yscrollcommand=pair_scrollbar.set)
        ttk.Button(
            self.mode2_controls, text="删除选中监测对", style="Alert.TButton",
            command=self._delete_pairs,
        ).pack(fill="x", pady=(0, 8))

        self.message_var = tk.StringVar(value="设置修改后立即生效")
        self.message_label = tk.Label(
            panel, textvariable=self.message_var, anchor="w", font=self._control_font,
            wraplength=self._panel_width - 40, justify="left",
        )
        self.message_label.pack(fill="x", pady=(12, 4))

        self.status_label = tk.Label(
            panel, text="状态：UNKNOWN", anchor="w", fg="#a86400",
            font=self._control_font,
        )
        self.status_label.pack(fill="x", pady=(12, 6))
        self.mode1_table_title = ttk.Label(
            panel, text="目标对距离（米）", font=self._control_font
        )
        self.mode1_table_title.pack(anchor="w")
        self.mode1_table_frame = ttk.Frame(panel)
        self.mode1_table_frame.pack(fill="both", expand=True, pady=(4, 10))
        self.table = ttk.Treeview(
            self.mode1_table_frame, columns=("subject", "hazard", "distance"),
            show="headings", height=8, style="Alert.Treeview",
        )
        for column, title, width in (
            ("subject", "被检测目标", 140),
            ("hazard", "危险物", 120),
            ("distance", "距离", 85),
        ):
            self.table.heading(column, text=title)
            self.table.column(column, width=width, anchor="center", stretch=False)
        self.table.tag_configure("safe", foreground="#007a25")
        self.table.tag_configure("warning", foreground="#c00000")
        self.table.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(self.mode1_table_frame, command=self.table.yview)
        scrollbar.pack(side="right", fill="y")
        self.table.configure(yscrollcommand=scrollbar.set)

        self.save_button = ttk.Button(
            panel, text="保存当前画面", command=self._save, style="Alert.TButton"
        )
        self.save_button.pack(fill="x", pady=(0, 6))
        ttk.Button(panel, text="退出", command=self.close, style="Alert.TButton").pack(fill="x")

    def _build_category_controls(
        self, panel: ttk.Frame, title: str, labels: list[str],
        candidate_var: tk.StringVar, category: str,
    ) -> tk.Listbox:
        """下拉框用于添加；已选列表支持选中一项或多项后删除。"""
        ttk.Label(panel, text=title, font=self._control_font).pack(anchor="w", pady=(14, 0))
        add_row = ttk.Frame(panel)
        add_row.pack(fill="x", pady=(4, 4))
        ttk.Combobox(
            add_row, textvariable=candidate_var, values=labels, state="readonly",
            font=self._control_font, style="Alert.TCombobox",
        ).pack(side="left", fill="x", expand=True)
        ttk.Button(
            add_row, text="添加", style="Alert.TButton",
            command=lambda: self._add_category(category),
        ).pack(side="left", padx=(6, 0))
        ttk.Label(
            panel, text="已选类别（Ctrl/Shift 可多选）", font=self._control_font
        ).pack(anchor="w")
        selected = tk.Listbox(
            panel, selectmode=tk.EXTENDED, exportselection=False,
            height=3, font=self._control_font,
        )
        selected.pack(fill="x")
        ttk.Button(
            panel, text="删除选中", style="Alert.TButton",
            command=lambda: self._remove_category(category),
        ).pack(fill="x", pady=(4, 0))
        return selected

    def _sync_category_lists(self) -> None:
        """仅在一次有效配置变更后刷新已选类别。"""
        for listbox, names in (
            (self.monitored_listbox, self._selected_monitored),
            (self.dangerous_listbox, self._selected_dangerous),
        ):
            listbox.delete(0, tk.END)
            listbox.insert(tk.END, *names)

    def _show_message(self, message: str, error: bool = False) -> None:
        self.message_var.set(message)
        self.message_label.configure(fg="#c00000" if error else "#006e24")

    def _distance_changed(self, *_args) -> None:
        try:
            self._on_distance_change(float(self.distance_var.get()))
        except ValueError:
            self._show_message("请输入大于 0 的有效米数", error=True)
            return
        self._show_message(f"安全距离已更新为 {self.distance_var.get()} m")

    def _apply_categories(self, monitored: list[str], dangerous: list[str]) -> None:
        """验证通过后再刷新列表，避免界面和检测器状态不一致。"""
        try:
            self._on_categories_change(monitored, dangerous)
        except ValueError as exc:
            self._show_message(str(exc), error=True)
            return
        self._selected_monitored = list(monitored)
        self._selected_dangerous = list(dangerous)
        self._sync_category_lists()
        self._show_message(
            f"已更新：被检测目标 {len(monitored)} 类，危险物 {len(dangerous)} 类"
        )

    def _add_category(self, category: str) -> None:
        candidate_var = (
            self.monitored_add_var if category == "monitored" else self.dangerous_add_var
        )
        name = candidate_var.get()
        if not name:
            self._show_message("请先从下拉框选择类别", error=True)
            return
        monitored = list(self._selected_monitored)
        dangerous = list(self._selected_dangerous)
        selected = monitored if category == "monitored" else dangerous
        if name in selected:
            self._show_message("该类别已经添加", error=True)
            return
        selected.append(name)
        self._apply_categories(monitored, dangerous)

    def _remove_category(self, category: str) -> None:
        listbox = self.monitored_listbox if category == "monitored" else self.dangerous_listbox
        selected_indices = set(listbox.curselection())
        if not selected_indices:
            self._show_message("请先在已选列表中选中要删除的类别", error=True)
            return
        monitored = list(self._selected_monitored)
        dangerous = list(self._selected_dangerous)
        names = monitored if category == "monitored" else dangerous
        remaining = [name for index, name in enumerate(names) if index not in selected_indices]
        if not remaining:
            self._show_message("每组至少保留一个类别", error=True)
            return
        if category == "monitored":
            monitored = remaining
        else:
            dangerous = remaining
        self._apply_categories(monitored, dangerous)

    def _mode_changed(self, _event) -> None:
        mode = self.mode_var.get()
        if mode == self._active_mode:
            return
        self._on_mode_change(mode)
        self._active_mode = mode
        if mode == "mode2":
            self.mode1_table_title.pack_forget()
            self.mode1_table_frame.pack_forget()
            self.mode2_controls.pack(before=self.message_label, fill="x")
            self.set_pair_prompt("点击“添加监测对”开始")
            self.status_label.configure(text="状态：未添加监测对", fg="#a86400")
            self._show_message("mode2：点击“添加监测对”后在画面中选目标")
        else:
            self.mode2_controls.pack_forget()
            self.mode1_table_title.pack(before=self.save_button, anchor="w")
            self.mode1_table_frame.pack(
                before=self.save_button, fill="both", expand=True, pady=(4, 10)
            )
            self._show_message("mode1：显示全部目标对距离")

    def _add_pair(self) -> None:
        self._on_add_pair()
        self.set_pair_prompt("请在画面中点击被检测目标的轮廓")

    def _cancel_pair(self) -> None:
        self._on_cancel_pair()
        self.set_pair_prompt("已取消添加，现有监测对保持不变")

    def _delete_pairs(self) -> None:
        pair_ids = [int(item) for item in self.pair_table.selection()]
        if not pair_ids:
            self._show_message("请先选中要删除的监测对", error=True)
            return
        self._on_delete_pairs(pair_ids)
        self._show_message(f"已删除 {len(pair_ids)} 组监测对")

    def set_pair_prompt(self, text: str) -> None:
        self.pair_prompt_var.set(text)

    def show_message(self, text: str, error: bool = False) -> None:
        self._show_message(text, error)

    @staticmethod
    def map_video_click(
        click_x: int, click_y: int, widget_width: int, widget_height: int,
        display_width: int, display_height: int,
        source_width: int, source_height: int,
    ) -> tuple[int, int] | None:
        """把等比例缩放且居中的界面坐标还原为相机原图像素。"""
        left = (widget_width - display_width) // 2
        top = (widget_height - display_height) // 2
        local_x = click_x - left
        local_y = click_y - top
        if not (0 <= local_x < display_width and 0 <= local_y < display_height):
            return None
        return (
            min(source_width - 1, int(local_x * source_width / display_width)),
            min(source_height - 1, int(local_y * source_height / display_height)),
        )

    def _video_clicked(self, event) -> None:
        if self.mode_var.get() != "mode2" or self._source_shape is None or self._display_size is None:
            return
        source_height, source_width = self._source_shape
        display_width, display_height = self._display_size
        pixel = self.map_video_click(
            event.x, event.y,
            self.video_label.winfo_width(), self.video_label.winfo_height(),
            display_width, display_height, source_width, source_height,
        )
        if pixel is not None:
            self._on_target_click(pixel[0], pixel[1], event.x_root, event.y_root)

    def choose_detection(
        self, candidates: list[Detection], screen_x: int, screen_y: int,
        on_choice: Callable[[Detection], None],
    ) -> None:
        """多个 mask 同时命中时让用户按类别和 ID 明确选择。"""
        if len(candidates) == 1:
            on_choice(candidates[0])
            return
        menu = tk.Menu(self.root, tearoff=False, font=self._control_font)
        for detection in candidates:
            menu.add_command(
                label=f"{detection.label} #{detection.track_id}",
                command=lambda selected=detection: on_choice(selected),
            )
        self._show_message(f"此处有 {len(candidates)} 个重叠轮廓，请在弹出菜单中选 ID")
        try:
            menu.tk_popup(screen_x, screen_y)
        finally:
            menu.grab_release()

    def _save(self) -> None:
        if self._on_save():
            self._show_message("已保存当前画面")
        else:
            self._show_message("当前还没有可保存的画面", error=True)

    def show_frame(self, frame_bgr: np.ndarray) -> None:
        """把 BGR 视频帧等比例缩放后放进界面，不改变测距所用原图。"""
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        image.thumbnail((self._video_width, self._video_height), Image.Resampling.BILINEAR)
        self._source_shape = frame_bgr.shape[:2]
        self._display_size = image.size
        self._image = ImageTk.PhotoImage(image, master=self.root)
        self.video_label.configure(image=self._image, text="")

    def clear_frame(self) -> None:
        """模式或类别切换时清除上一帧，避免误点旧目标。"""
        self._image = None
        self._source_shape = None
        self._display_size = None
        self.video_label.configure(image="", text="等待当前配置的画面…")

    def show_mode2_pairs(
        self, pairs: list[MonitoringPair], matched: dict[int, DistanceMeasurement],
        safety_distance_m: float, status: str,
    ) -> None:
        """只显示用户选定的监测对；本帧缺测的配对显示为 --。"""
        colors = {"SAFE": "#007a25", "WARNING": "#c00000", "UNKNOWN": "#a86400"}
        self.status_label.configure(
            text=f"状态：{status}" if pairs else "状态：未添加监测对",
            fg=colors[status],
        )
        scroll_position = self.pair_table.yview()[0]
        selected = set(self.pair_table.selection())
        for item in self.pair_table.get_children():
            self.pair_table.delete(item)
        for pair in pairs:
            result = matched.get(pair.pair_id)
            tag = (
                "warning" if result and result.distance_m <= safety_distance_m
                else "safe" if result else ""
            )
            self.pair_table.insert(
                "", "end", iid=str(pair.pair_id),
                values=(
                    f"{pair.monitored.label}#{pair.monitored.track_id} → "
                    f"{pair.dangerous.label}#{pair.dangerous.track_id}",
                    f"{result.distance_m:.3f}" if result else "--",
                ),
                tags=(tag,) if tag else (),
            )
        remaining = [item for item in selected if self.pair_table.exists(item)]
        if remaining:
            self.pair_table.selection_set(remaining)
        self.pair_table.yview_moveto(scroll_position)

    def show_measurements(
        self, measurements: list[DistanceMeasurement], safety_distance_m: float, status: str
    ) -> None:
        """mode1 把所有目标对列在表格，状态由全部距离共同决定。"""
        colors = {"SAFE": "#007a25", "WARNING": "#c00000", "UNKNOWN": "#a86400"}
        self.status_label.configure(text=f"状态：{status}", fg=colors[status])
        scroll_position = self.table.yview()[0]
        for item in self.table.get_children():
            self.table.delete(item)
        for measurement in measurements:
            warning = measurement.distance_m <= safety_distance_m
            self.table.insert(
                "", "end",
                values=(
                    f"{measurement.monitored_label} #{measurement.monitored_track_id}",
                    f"{measurement.dangerous_label} #{measurement.dangerous_track_id}",
                    f"{measurement.distance_m:.3f}",
                ),
                tags=("warning" if warning else "safe",),
            )
        self.table.yview_moveto(scroll_position)

    def run(self, frame_callback: Callable[[], None]) -> None:
        """在 Tk 主线程更新视频和控件；异常会返回给主程序统一清理硬件。"""
        def tick() -> None:
            if self._closed:
                return
            try:
                frame_callback()
            except Exception as exc:
                self._error = exc
                self.close()
                return
            self.root.after(1, tick)

        self.root.after(0, tick)
        self.root.mainloop()
        if self._error is not None:
            raise self._error

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.root.quit()
        self.root.destroy()
