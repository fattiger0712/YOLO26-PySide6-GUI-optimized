from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PySide6.QtCore import QProcess, QSize, Qt, QThread, QUrl
from PySide6.QtGui import QCloseEvent, QDesktopServices, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.detection_worker import DetectionWorker
from core.history_store import HistoryStore
from core.models import DetectionConfig, FrameResult, SourceSpec, iter_supported_media
from core.review_dataset import build_labelimg_args, export_review_sample, resolve_labelimg_executable
from core.weight_registry import WeightRegistryStore
from ui.theme import APP_QSS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_WEIGHT_REGISTRY = CONFIG_DIR / "model_weights.json"


@dataclass
class FrameSnapshot:
    raw_jpeg: bytes
    annotated_jpeg: bytes
    fps: float
    class_count: int
    target_count: int
    class_counts: Dict[str, int]
    frame_index: int
    progress: int
    detections: List[Dict]
    class_names: List[str]
    source_path: str
    source_name: str

    def to_frame_result(self, raw_frame, annotated_frame) -> FrameResult:
        return FrameResult(
            raw_frame=raw_frame,
            annotated_frame=annotated_frame,
            fps=self.fps,
            class_count=self.class_count,
            target_count=self.target_count,
            class_counts=dict(self.class_counts),
            frame_index=self.frame_index,
            progress=self.progress,
            detections=[dict(item) for item in self.detections],
            class_names=list(self.class_names),
            source_path=self.source_path,
            source_name=self.source_name,
        )


class ImageLabel(QLabel):
    def __init__(self, text: str):
        super().__init__(text)
        self._zoom_title = text
        self._frame = None
        self._zoom_callback = None
        self.setObjectName("ImageView")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(QSize(360, 260))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setScaledContents(False)
        self.setCursor(Qt.ArrowCursor)

    def set_zoom_callback(self, callback) -> None:
        self._zoom_callback = callback

    def set_frame(self, frame) -> None:
        self._frame = frame
        self.setCursor(Qt.PointingHandCursor if frame is not None else Qt.ArrowCursor)

    def clear_frame(self) -> None:
        self._frame = None
        self.setCursor(Qt.ArrowCursor)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._frame is not None and self._zoom_callback:
            self._zoom_callback(self._zoom_title, self._frame)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class ZoomImageView(QGraphicsView):
    def __init__(self, pixmap: QPixmap, parent: QWidget | None = None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._fit_mode = True
        self._zoom_steps = 0

        self.setScene(self._scene)
        self.setAlignment(Qt.AlignCenter)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

    def fit_to_view(self) -> None:
        if self._pixmap_item.pixmap().isNull():
            return
        self._fit_mode = True
        self._zoom_steps = 0
        self.resetTransform()
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    def reset_zoom(self) -> None:
        self._fit_mode = False
        self._zoom_steps = 0
        self.resetTransform()

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return

        if delta > 0:
            if self._zoom_steps >= 12:
                event.accept()
                return
            factor = 1.25
            self._zoom_steps += 1
        else:
            if self._zoom_steps <= -8:
                event.accept()
                return
            factor = 0.8
            self._zoom_steps -= 1

        self._fit_mode = False
        self.scale(factor, factor)
        event.accept()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._fit_mode:
            self.fit_to_view()


class ImageZoomDialog(QDialog):
    def __init__(self, title: str, pixmap: QPixmap, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1000, 700)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        toolbar = QHBoxLayout()
        fit_btn = QPushButton("Fit")
        reset_btn = QPushButton("1:1")
        close_btn = QPushButton("Close")
        fit_btn.clicked.connect(self._fit)
        reset_btn.clicked.connect(self._reset)
        close_btn.clicked.connect(self.close)
        toolbar.addWidget(fit_btn)
        toolbar.addWidget(reset_btn)
        toolbar.addStretch(1)
        toolbar.addWidget(close_btn)

        self.view = ZoomImageView(pixmap, self)
        layout.addLayout(toolbar)
        layout.addWidget(self.view, 1)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.view.fit_to_view()

    def _fit(self) -> None:
        self.view.fit_to_view()

    def _reset(self) -> None:
        self.view.reset_zoom()


class ReviewIssueDialog(QDialog):
    REASONS = ["漏标", "误标", "框不准", "误检", "微小目标", "季节/光照", "遮挡", "模糊", "其他"]

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("复核补标")
        self.resize(420, 260)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        layout.addWidget(QLabel("问题原因"))
        self.reason_combo = QComboBox()
        self.reason_combo.addItems(self.REASONS)
        layout.addWidget(self.reason_combo)

        layout.addWidget(QLabel("备注"))
        self.note_edit = QPlainTextEdit()
        self.note_edit.setPlaceholderText("可记录微小目标、季节光照、遮挡、模糊等具体情况")
        layout.addWidget(self.note_edit, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def issue_reason(self) -> str:
        return self.reason_combo.currentText()

    @property
    def note(self) -> str:
        return self.note_edit.toPlainText().strip()


class WeightMetadataDialog(QDialog):
    def __init__(self, record: Dict, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("编辑权重备注")
        self.resize(520, 360)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        layout.addWidget(QLabel("显示名称"))
        self.display_name_edit = QLineEdit(str(record.get("display_name") or record.get("model_name", "")))
        layout.addWidget(self.display_name_edit)

        layout.addWidget(QLabel("推荐用途"))
        self.recommended_edit = QLineEdit(str(record.get("recommended_for", "")))
        self.recommended_edit.setPlaceholderText("例如：TT100K 交通标志、小目标检测、夜间场景")
        layout.addWidget(self.recommended_edit)

        layout.addWidget(QLabel("标签"))
        self.tags_edit = QLineEdit(str(record.get("tags", "")))
        self.tags_edit.setPlaceholderText("例如：tt100k, yolo26, high-recall")
        layout.addWidget(self.tags_edit)

        layout.addWidget(QLabel("备注"))
        self.notes_edit = QPlainTextEdit()
        self.notes_edit.setPlainText(str(record.get("notes", "")))
        self.notes_edit.setPlaceholderText("记录训练策略、适用场景、已知问题或选择建议")
        layout.addWidget(self.notes_edit, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> Dict[str, str]:
        return {
            "display_name": self.display_name_edit.text().strip(),
            "recommended_for": self.recommended_edit.text().strip(),
            "tags": self.tags_edit.text().strip(),
            "notes": self.notes_edit.toPlainText().strip(),
        }


class ArtifactPreview(QLabel):
    def __init__(self, title: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.title = title
        self.path = ""
        self._zoom_callback = None
        self.setObjectName("ImageView")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(QSize(180, 120))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setWordWrap(True)

    def set_artifact(self, path: str, zoom_callback) -> None:
        self.path = path or ""
        self._zoom_callback = zoom_callback
        if not self.path or not Path(self.path).exists():
            self.setPixmap(QPixmap())
            self.setText(f"{self.title}\n未生成")
            self.setCursor(Qt.ArrowCursor)
            return
        pixmap = QPixmap(self.path)
        if pixmap.isNull():
            self.setPixmap(QPixmap())
            self.setText(f"{self.title}\n无法预览")
            self.setCursor(Qt.ArrowCursor)
            return
        self.setText("")
        self.setPixmap(
            pixmap.scaled(
                max(1, self.width() - 10),
                max(1, self.height() - 10),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(self.path)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.path:
            self.set_artifact(self.path, self._zoom_callback)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self.path and self._zoom_callback:
            self._zoom_callback(self.title, self.path)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class WeightManagerDialog(QDialog):
    IMAGE_FIELDS = [
        ("训练曲线", "results_png"),
        ("PR 曲线", "pr_curve"),
        ("F1 曲线", "f1_curve"),
        ("Precision 曲线", "precision_curve"),
        ("Recall 曲线", "recall_curve"),
        ("混淆矩阵", "confusion_matrix"),
        ("归一化混淆矩阵", "confusion_matrix_normalized"),
    ]

    def __init__(
        self,
        store: WeightRegistryStore,
        models_dir: Path,
        project_root: Path,
        apply_callback,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.store = store
        self.models_dir = models_dir
        self.project_root = project_root
        self.apply_callback = apply_callback
        self.records: List[Dict] = []
        self._artifact_widgets: Dict[str, ArtifactPreview] = {}
        self._detail_labels: Dict[str, QLabel] = {}

        self.setWindowTitle("模型权重管理")
        self.resize(1180, 720)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_table_panel())
        splitter.addWidget(self._build_detail_panel())
        splitter.setSizes([620, 560])
        layout.addWidget(splitter, 1)

        self._refresh_records()

    def _build_table_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.weight_table = QTableWidget(0, 8)
        self.weight_table.setHorizontalHeaderLabels(
            ["权重", "训练名", "数据集", "mAP50", "mAP50-95", "Precision", "Recall", "推荐用途"]
        )
        self.weight_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.weight_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.weight_table.verticalHeader().setVisible(False)
        self.weight_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.weight_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.weight_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.weight_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.weight_table.currentCellChanged.connect(self._on_selection_changed)
        layout.addWidget(self.weight_table, 1)

        row_one = QHBoxLayout()
        import_btn = QPushButton("导入训练目录")
        import_btn.clicked.connect(self._import_training_run)
        associate_btn = QPushButton("手动关联权重")
        associate_btn.clicked.connect(self._associate_weight)
        edit_btn = QPushButton("编辑备注/用途")
        edit_btn.clicked.connect(self._edit_metadata)
        refresh_btn = QPushButton("刷新指标")
        refresh_btn.clicked.connect(self._refresh_metrics)
        remove_btn = QPushButton("删除导入记录")
        remove_btn.setObjectName("DangerButton")
        remove_btn.clicked.connect(self._remove_selected_record)
        row_one.addWidget(import_btn)
        row_one.addWidget(associate_btn)
        row_one.addWidget(edit_btn)
        row_one.addWidget(refresh_btn)
        row_one.addWidget(remove_btn)
        layout.addLayout(row_one)

        row_two = QHBoxLayout()
        apply_btn = QPushButton("应用为当前权重")
        apply_btn.setObjectName("PrimaryButton")
        apply_btn.clicked.connect(self._apply_selected_weight)
        open_btn = QPushButton("打开训练目录")
        open_btn.clicked.connect(self._open_training_dir)
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.accept)
        row_two.addWidget(apply_btn)
        row_two.addWidget(open_btn)
        row_two.addStretch(1)
        row_two.addWidget(close_btn)
        layout.addLayout(row_two)
        return panel

    def _build_detail_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 0, 0, 0)
        layout.setSpacing(10)

        summary = QFrame()
        summary.setObjectName("Panel")
        summary_layout = QGridLayout(summary)
        summary_layout.setContentsMargins(14, 14, 14, 14)
        summary_layout.setHorizontalSpacing(12)
        summary_layout.setVerticalSpacing(8)
        fields = [
            ("权重", "model_name"),
            ("显示名称", "display_name"),
            ("训练目录", "training_name"),
            ("数据集", "dataset"),
            ("基础模型", "base_model"),
            ("最佳 epoch", "best_epoch"),
            ("mAP50", "best_map50"),
            ("mAP50-95", "best_map5095"),
            ("Precision", "best_precision"),
            ("Recall", "best_recall"),
            ("训练参数", "train_args"),
            ("推荐用途", "recommended_for"),
            ("标签", "tags"),
        ]
        for index, (title, key) in enumerate(fields):
            title_label = QLabel(title)
            title_label.setObjectName("Muted")
            value_label = QLabel("--")
            value_label.setWordWrap(True)
            self._detail_labels[key] = value_label
            row = index // 2
            col = (index % 2) * 2
            summary_layout.addWidget(title_label, row, col)
            summary_layout.addWidget(value_label, row, col + 1)
        layout.addWidget(summary)

        layout.addWidget(QLabel("备注"))
        self.notes_view = QPlainTextEdit()
        self.notes_view.setReadOnly(True)
        self.notes_view.setMinimumHeight(90)
        layout.addWidget(self.notes_view)

        images = QFrame()
        images.setObjectName("Panel")
        image_grid = QGridLayout(images)
        image_grid.setContentsMargins(14, 14, 14, 14)
        image_grid.setHorizontalSpacing(10)
        image_grid.setVerticalSpacing(10)
        for index, (title, key) in enumerate(self.IMAGE_FIELDS):
            preview = ArtifactPreview(title)
            self._artifact_widgets[key] = preview
            image_grid.addWidget(preview, index // 2, index % 2)
        layout.addWidget(images)
        layout.addStretch(1)

        scroll.setWidget(container)
        return scroll

    def _refresh_records(self, select_model: str | None = None) -> None:
        current = select_model or self._current_model_name()
        hidden_models = self.store.hidden_model_names()
        records_by_model = {record.get("model_name"): dict(record) for record in self.store.load()}
        self.models_dir.mkdir(exist_ok=True)
        for model_path in sorted(self.models_dir.glob("*.pt"), key=lambda path: path.name.lower()):
            if model_path.name in hidden_models:
                continue
            if model_path.name not in records_by_model:
                records_by_model[model_path.name] = {
                    "model_name": model_path.name,
                    "model_path": str(model_path.resolve()),
                    "display_name": model_path.stem,
                    "training_name": "",
                    "training_dir": "",
                    "dataset": "",
                    "base_model": "",
                    "metrics": {},
                    "artifacts": {},
                    "recommended_for": "",
                    "tags": "",
                    "notes": "",
                }

        self.records = sorted(
            records_by_model.values(),
            key=lambda record: str(record.get("display_name") or record.get("model_name", "")).lower(),
        )
        self.weight_table.setRowCount(len(self.records))
        selected_row = 0
        for row, record in enumerate(self.records):
            metrics = record.get("metrics") or {}
            values = [
                record.get("model_name", ""),
                record.get("training_name", ""),
                record.get("dataset", ""),
                self._format_metric(metrics.get("best_map50")),
                self._format_metric(metrics.get("best_map5095")),
                self._format_metric(metrics.get("best_precision")),
                self._format_metric(metrics.get("best_recall")),
                record.get("recommended_for", ""),
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if col in {0, 2, 7}:
                    item.setToolTip(str(value))
                self.weight_table.setItem(row, col, item)
            if current and record.get("model_name") == current:
                selected_row = row

        if self.records:
            self.weight_table.selectRow(selected_row)
            self._show_record(self.records[selected_row])
        else:
            self._show_record({})

    def _on_selection_changed(self, current_row: int, _current_col: int, _previous_row: int, _previous_col: int) -> None:
        if 0 <= current_row < len(self.records):
            self._show_record(self.records[current_row])

    def _current_record(self) -> Dict:
        row = self.weight_table.currentRow()
        if 0 <= row < len(self.records):
            return self.records[row]
        return {}

    def _current_model_name(self) -> str:
        record = self._current_record()
        return str(record.get("model_name", ""))

    def _show_record(self, record: Dict) -> None:
        metrics = record.get("metrics") or {}
        self._detail_labels["model_name"].setText(str(record.get("model_name", "--") or "--"))
        self._detail_labels["display_name"].setText(str(record.get("display_name", "--") or "--"))
        self._detail_labels["training_name"].setText(str(record.get("training_name", "--") or "--"))
        self._detail_labels["dataset"].setText(str(record.get("dataset", "--") or "--"))
        self._detail_labels["base_model"].setText(str(record.get("base_model", "--") or "--"))
        self._detail_labels["best_epoch"].setText(str(metrics.get("best_epoch", "--") or "--"))
        self._detail_labels["best_map50"].setText(self._format_metric(metrics.get("best_map50")))
        self._detail_labels["best_map5095"].setText(self._format_metric(metrics.get("best_map5095")))
        self._detail_labels["best_precision"].setText(self._format_metric(metrics.get("best_precision")))
        self._detail_labels["best_recall"].setText(self._format_metric(metrics.get("best_recall")))
        args_text = f"epochs={record.get('epochs', '')}, imgsz={record.get('imgsz', '')}, batch={record.get('batch', '')}"
        self._detail_labels["train_args"].setText(args_text.strip(", "))
        self._detail_labels["recommended_for"].setText(str(record.get("recommended_for", "--") or "--"))
        self._detail_labels["tags"].setText(str(record.get("tags", "--") or "--"))
        self.notes_view.setPlainText(str(record.get("notes", "")))

        artifacts = record.get("artifacts") or {}
        for _title, key in self.IMAGE_FIELDS:
            self._artifact_widgets[key].set_artifact(str(artifacts.get(key, "")), self._open_artifact_zoom)

    def _import_training_run(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择 YOLO 训练输出目录", str(self.project_root.parent))
        if not folder:
            return
        try:
            record = self.store.import_training_run(folder, self.models_dir)
        except Exception as exc:
            QMessageBox.warning(self, "导入失败", str(exc))
            return
        self._refresh_records(record.get("model_name", ""))
        QMessageBox.information(self, "导入完成", f"已导入权重：{record.get('model_name', '')}")

    def _associate_weight(self) -> None:
        record = self._current_record()
        default_weight = str(record.get("model_path", "")) if record else ""
        open_dir = str(Path(default_weight).parent) if default_weight and Path(default_weight).exists() else str(self.models_dir)
        model_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择要关联的权重",
            open_dir,
            "YOLO weights (*.pt)",
        )
        if not model_path:
            return

        folder = QFileDialog.getExistingDirectory(self, "选择该权重对应的训练输出目录", str(self.project_root.parent))
        if not folder:
            return
        try:
            updated = self.store.register_training_run(folder, model_path)
        except Exception as exc:
            QMessageBox.warning(self, "关联失败", str(exc))
            return
        self._refresh_records(updated.get("model_name", ""))

    def _edit_metadata(self) -> None:
        record = self._current_record()
        if not record:
            return
        dialog = WeightMetadataDialog(record, self)
        if dialog.exec() != QDialog.Accepted:
            return
        updated = dict(record)
        updated.update(dialog.values())
        updated["display_name"] = updated.get("display_name") or updated.get("model_name", "")
        updated.setdefault("metrics", {})
        updated.setdefault("artifacts", {})
        saved = self.store.upsert(updated)
        self._refresh_records(saved.get("model_name", ""))

    def _remove_selected_record(self) -> None:
        record = self._current_record()
        model_name = str(record.get("model_name", ""))
        if not model_name:
            return

        reply = QMessageBox.question(
            self,
            "删除导入记录",
            f"确定从权重管理中删除「{model_name}」吗？\n\n本地 .pt 权重文件不会被删除，之后仍可通过“手动关联权重”或“导入训练目录”重新加入。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self.store.remove_from_manager(model_name)
        self._refresh_records()

    def _refresh_metrics(self) -> None:
        record = self._current_record()
        model_name = str(record.get("model_name", ""))
        if not model_name:
            return
        try:
            refreshed = self.store.refresh(model_name)
        except Exception as exc:
            QMessageBox.warning(self, "刷新失败", str(exc))
            return
        self._refresh_records(refreshed.get("model_name", ""))

    def _apply_selected_weight(self) -> None:
        record = self._current_record()
        model_name = str(record.get("model_name", ""))
        if not model_name:
            return
        model_path = Path(str(record.get("model_path") or self.models_dir / model_name))
        if not model_path.exists() and (self.models_dir / model_name).exists():
            model_path = self.models_dir / model_name
        if not model_path.exists():
            QMessageBox.warning(self, "权重不存在", f"找不到权重文件：{model_path}")
            return
        self.apply_callback(model_name)
        QMessageBox.information(self, "已应用", f"当前检测权重已切换为：{model_name}")

    def _open_training_dir(self) -> None:
        record = self._current_record()
        training_dir = str(record.get("training_dir", ""))
        if not training_dir or not Path(training_dir).exists():
            QMessageBox.information(self, "没有训练目录", "当前权重还没有关联训练输出目录。")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(training_dir))

    def _open_artifact_zoom(self, title: str, path: str) -> None:
        pixmap = QPixmap(path)
        if pixmap.isNull():
            return
        dialog = ImageZoomDialog(title, pixmap, self)
        dialog.exec()

    @staticmethod
    def _format_metric(value) -> str:
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return "--"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO26 检测工作台")
        self.resize(1320, 820)
        self.setMinimumSize(QSize(1120, 720))
        self.setStyleSheet(APP_QSS)

        self.history_store = HistoryStore(OUTPUT_DIR)
        self.weight_store = WeightRegistryStore(MODEL_WEIGHT_REGISTRY)
        self.source: Optional[SourceSpec] = None
        self.worker: Optional[DetectionWorker] = None
        self.worker_thread: Optional[QThread] = None
        self.current_output_dir = OUTPUT_DIR
        self._is_running = False
        self._is_paused = False
        self._last_raw = None
        self._last_annotated = None
        self._last_frame_result: Optional[FrameResult] = None
        self._frame_snapshots: List[FrameSnapshot] = []
        self._browser_index = -1
        self._auto_follow_preview = True
        self._browser_updating = False
        self.labelimg_process: Optional[QProcess] = None

        self._build_ui()
        self._load_models()
        self._load_settings()
        self._refresh_history_table()
        self._set_idle_state()

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        self.setCentralWidget(root)

        layout.addWidget(self._build_top_bar())

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setChildrenCollapsible(False)
        main_splitter.addWidget(self._build_left_panel())
        main_splitter.addWidget(self._build_preview_panel())
        main_splitter.addWidget(self._build_right_panel())
        main_splitter.setSizes([250, 760, 280])
        layout.addWidget(main_splitter, 1)

        layout.addWidget(self._build_history_panel())
        layout.addWidget(self._build_status_bar())

    def _build_top_bar(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("TopBar")
        row = QHBoxLayout(frame)
        row.setContentsMargins(18, 14, 18, 14)
        row.setSpacing(12)

        title_box = QVBoxLayout()
        title = QLabel("YOLO26 检测工作台")
        title.setObjectName("AppTitle")
        subtitle = QLabel("本地文件、摄像头和 HTTP/RTSP 统一后台推理")
        subtitle.setObjectName("SubTitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        row.addLayout(title_box, 1)

        self.open_outputs_btn = QPushButton("打开输出目录")
        self.open_outputs_btn.clicked.connect(self._open_outputs_dir)
        self.open_csv_btn = QPushButton("打开CSV历史")
        self.open_csv_btn.clicked.connect(self._open_history_csv)
        row.addWidget(self.open_outputs_btn)
        row.addWidget(self.open_csv_btn)
        return frame

    def _build_left_panel(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("Panel")
        frame.setMinimumWidth(230)
        frame.setMaximumWidth(300)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        layout.addWidget(self._section_title("检测来源"))
        self.source_label = QLabel("未选择来源")
        self.source_label.setObjectName("Muted")
        self.source_label.setWordWrap(True)
        layout.addWidget(self.source_label)

        self.file_btn = QPushButton("选择图片/视频")
        self.file_btn.clicked.connect(self._choose_file)
        self.batch_btn = QPushButton("批量处理文件夹")
        self.batch_btn.clicked.connect(self._choose_batch_folder)
        self.camera_btn = QPushButton("打开摄像头")
        self.camera_btn.clicked.connect(self._choose_camera)
        self.stream_btn = QPushButton("HTTP/RTSP")
        self.stream_btn.clicked.connect(self._choose_stream)
        layout.addWidget(self.file_btn)
        layout.addWidget(self.batch_btn)
        layout.addWidget(self.camera_btn)
        layout.addWidget(self.stream_btn)

        layout.addSpacing(10)
        layout.addWidget(self._section_title("运行控制"))
        self.run_btn = QPushButton("开始检测")
        self.run_btn.setObjectName("PrimaryButton")
        self.run_btn.clicked.connect(self._run_or_toggle_pause)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setObjectName("DangerButton")
        self.stop_btn.clicked.connect(self._stop_worker)
        layout.addWidget(self.run_btn)
        layout.addWidget(self.stop_btn)

        layout.addSpacing(10)
        self.save_result_check = QCheckBox("保存JPG/MP4结果")
        self.save_txt_check = QCheckBox("保存TXT标签")
        layout.addWidget(self.save_result_check)
        layout.addWidget(self.save_txt_check)

        layout.addSpacing(10)
        layout.addWidget(self._section_title("复核补标"))
        self.review_btn = QPushButton("复核当前帧")
        self.review_btn.clicked.connect(self._review_current_frame)
        layout.addWidget(self.review_btn)

        layout.addStretch(1)
        return frame

    def _build_preview_panel(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("PreviewPanel")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        metrics = QGridLayout()
        metrics.setHorizontalSpacing(10)
        metrics.setVerticalSpacing(10)
        self.class_card_value = QLabel("--")
        self.target_card_value = QLabel("--")
        self.fps_card_value = QLabel("--")
        self.model_card_value = QLabel("--")
        metrics.addWidget(self._metric_card("类别数", self.class_card_value), 0, 0)
        metrics.addWidget(self._metric_card("目标数", self.target_card_value), 0, 1)
        metrics.addWidget(self._metric_card("FPS", self.fps_card_value), 0, 2)
        metrics.addWidget(self._metric_card("当前模型", self.model_card_value), 0, 3)
        layout.addLayout(metrics)

        image_row = QSplitter(Qt.Horizontal)
        image_row.setChildrenCollapsible(False)
        self.raw_view = ImageLabel("原始画面")
        self.result_view = ImageLabel("检测结果")
        self.raw_view.set_zoom_callback(self._open_zoom_dialog)
        self.result_view.set_zoom_callback(self._open_zoom_dialog)
        image_row.addWidget(self.raw_view)
        image_row.addWidget(self.result_view)
        image_row.setSizes([1, 1])
        layout.addWidget(image_row, 1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        browser_row = QHBoxLayout()
        browser_row.setSpacing(8)
        self.prev_frame_btn = QPushButton("上一帧")
        self.prev_frame_btn.clicked.connect(self._show_previous_frame)
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.valueChanged.connect(self._on_frame_slider_changed)
        self.frame_counter_label = QLabel("0/0")
        self.frame_counter_label.setObjectName("Muted")
        self.frame_counter_label.setMinimumWidth(64)
        self.frame_counter_label.setAlignment(Qt.AlignCenter)
        self.next_frame_btn = QPushButton("下一帧")
        self.next_frame_btn.clicked.connect(self._show_next_frame)
        browser_row.addWidget(self.prev_frame_btn)
        browser_row.addWidget(self.frame_slider, 1)
        browser_row.addWidget(self.frame_counter_label)
        browser_row.addWidget(self.next_frame_btn)
        layout.addLayout(browser_row)
        self._update_frame_browser_controls()
        return frame

    def _build_right_panel(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("Panel")
        frame.setMinimumWidth(260)
        frame.setMaximumWidth(340)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        layout.addWidget(self._section_title("模型与参数"))
        self.model_combo = QComboBox()
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        layout.addWidget(QLabel("模型权重"))
        layout.addWidget(self.model_combo)
        self.weight_manager_btn = QPushButton("权重管理")
        self.weight_manager_btn.clicked.connect(self._open_weight_manager)
        layout.addWidget(self.weight_manager_btn)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.00)
        self.conf_spin.setSingleStep(0.01)
        self.conf_spin.setDecimals(2)
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 100)
        self._bind_double_slider(self.conf_spin, self.conf_slider)
        layout.addWidget(QLabel("Conf阈值"))
        layout.addWidget(self.conf_spin)
        layout.addWidget(self.conf_slider)

        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.01, 1.00)
        self.iou_spin.setSingleStep(0.01)
        self.iou_spin.setDecimals(2)
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(1, 100)
        self._bind_double_slider(self.iou_spin, self.iou_slider)
        layout.addWidget(QLabel("IoU阈值"))
        layout.addWidget(self.iou_spin)
        layout.addWidget(self.iou_slider)

        self.rate_spin = QSpinBox()
        self.rate_spin.setRange(0, 1000)
        self.rate_spin.setSingleStep(5)
        self.rate_slider = QSlider(Qt.Horizontal)
        self.rate_slider.setRange(0, 1000)
        self.rate_spin.valueChanged.connect(self.rate_slider.setValue)
        self.rate_slider.valueChanged.connect(self.rate_spin.setValue)
        layout.addWidget(QLabel("帧间隔(ms)"))
        layout.addWidget(self.rate_spin)
        layout.addWidget(self.rate_slider)

        hint = QLabel("提示：实时源在后台线程推理，停止时会释放摄像头或流。")
        hint.setObjectName("Muted")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch(1)
        return frame

    def _build_history_panel(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("Panel")
        frame.setMinimumHeight(160)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)

        header = QHBoxLayout()
        header.addWidget(self._section_title("检测历史"))
        header.addStretch(1)
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self._refresh_history_table)
        header.addWidget(refresh_btn)
        layout.addLayout(header)

        self.history_table = QTableWidget(0, 8)
        self.history_table.setHorizontalHeaderLabels(
            ["时间", "来源", "模型", "帧数", "平均FPS", "目标累计", "状态", "输出目录"]
        )
        self.history_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.history_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.history_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.history_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.history_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.history_table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeToContents)
        layout.addWidget(self.history_table)
        return frame

    def _build_status_bar(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("StatusBar")
        row = QHBoxLayout(frame)
        row.setContentsMargins(14, 8, 14, 8)
        self.status_label = QLabel("欢迎使用")
        self.status_label.setObjectName("StatusText")
        row.addWidget(self.status_label, 1)
        return frame

    @staticmethod
    def _section_title(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("SectionTitle")
        return label

    @staticmethod
    def _metric_card(title: str, value_label: QLabel) -> QFrame:
        frame = QFrame()
        frame.setObjectName("MetricCard")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 10, 12, 10)
        title_label = QLabel(title)
        title_label.setObjectName("Muted")
        value_label.setObjectName("MetricValue")
        value_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        value_label.setWordWrap(True)
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        return frame

    @staticmethod
    def _bind_double_slider(spin: QDoubleSpinBox, slider: QSlider) -> None:
        def spin_to_slider(value: float) -> None:
            slider.blockSignals(True)
            slider.setValue(int(round(value * 100)))
            slider.blockSignals(False)

        def slider_to_spin(value: int) -> None:
            spin.blockSignals(True)
            spin.setValue(value / 100)
            spin.blockSignals(False)

        spin.valueChanged.connect(spin_to_slider)
        slider.valueChanged.connect(slider_to_spin)

    def _load_models(self) -> None:
        MODELS_DIR.mkdir(exist_ok=True)
        current_model = self.model_combo.currentText() if hasattr(self, "model_combo") else ""
        hidden_models = self.weight_store.hidden_model_names()
        models = sorted(
            [
                path
                for path in MODELS_DIR.iterdir()
                if path.suffix.lower() == ".pt" and path.name not in hidden_models
            ],
            key=lambda path: path.stat().st_size,
        )
        self.model_combo.clear()
        self.model_combo.addItems([path.name for path in models])
        if models:
            selected = current_model if current_model in [path.name for path in models] else models[0].name
            self._select_model_by_name(selected)
        else:
            self.model_card_value.setText("未找到")
            self._set_status("models 目录下没有 .pt 权重文件")

    def _select_model_by_name(self, model_name: str) -> None:
        index = self.model_combo.findText(model_name)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)
            self.model_card_value.setText(model_name)

    def _load_settings(self) -> None:
        config = self._read_json(CONFIG_DIR / "setting.json", {})
        self.iou_spin.setValue(float(config.get("iou", 0.70)))
        self.conf_spin.setValue(float(config.get("conf", 0.25)))
        self.rate_spin.setValue(int(config.get("rate", 30)))
        self.save_result_check.setChecked(bool(config.get("save_res", 0)))
        self.save_txt_check.setChecked(bool(config.get("save_txt", 0)))

    def _save_settings(self) -> None:
        CONFIG_DIR.mkdir(exist_ok=True)
        config = {
            "iou": self.iou_spin.value(),
            "conf": self.conf_spin.value(),
            "rate": self.rate_spin.value(),
            "save_res": 2 if self.save_result_check.isChecked() else 0,
            "save_txt": 2 if self.save_txt_check.isChecked() else 0,
        }
        with (CONFIG_DIR / "setting.json").open("w", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2)

    def _choose_file(self) -> None:
        fold_config = CONFIG_DIR / "fold.json"
        open_dir = self._read_json(fold_config, {}).get("open_fold", str(PROJECT_ROOT))
        if not Path(open_dir).exists():
            open_dir = str(PROJECT_ROOT)
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片或视频",
            open_dir,
            "Media (*.mp4 *.mkv *.avi *.flv *.mov *.wmv *.jpg *.jpeg *.png *.bmp *.webp)",
        )
        if not filename:
            return
        try:
            self.source = SourceSpec.from_file(filename)
        except ValueError as exc:
            QMessageBox.warning(self, "不支持的文件", str(exc))
            return
        self.source_label.setText(self.source.display_name)
        self._set_status(f"已加载文件：{self.source.display_name}")
        CONFIG_DIR.mkdir(exist_ok=True)
        with fold_config.open("w", encoding="utf-8") as handle:
            json.dump({"open_fold": str(Path(filename).parent)}, handle, ensure_ascii=False, indent=2)

    def _choose_batch_folder(self) -> None:
        fold_config = CONFIG_DIR / "fold.json"
        open_dir = self._read_json(fold_config, {}).get("open_fold", str(PROJECT_ROOT))
        if not Path(open_dir).exists():
            open_dir = str(PROJECT_ROOT)
        folder = QFileDialog.getExistingDirectory(self, "选择批量处理文件夹", open_dir)
        if not folder:
            return
        try:
            media_files = iter_supported_media(folder)
        except ValueError as exc:
            QMessageBox.warning(self, "文件夹不可用", str(exc))
            return
        if not media_files:
            QMessageBox.information(
                self,
                "没有可处理文件",
                "该文件夹内没有支持的图片或视频文件。",
            )
            return
        self.source = SourceSpec.batch(folder)
        self.source_label.setText(f"批量：{Path(folder).name}（{len(media_files)} 个文件）")
        self._set_status(f"已选择批量文件夹：{folder}，共 {len(media_files)} 个文件")
        CONFIG_DIR.mkdir(exist_ok=True)
        with fold_config.open("w", encoding="utf-8") as handle:
            json.dump({"open_fold": folder}, handle, ensure_ascii=False, indent=2)

    def _choose_camera(self) -> None:
        self.source = SourceSpec.camera(0)
        self.source_label.setText("摄像头 0")
        self._set_status("已选择摄像头 0")

    def _choose_stream(self) -> None:
        ip_config = CONFIG_DIR / "ip.json"
        default_url = self._read_json(ip_config, {}).get("ip", "http://10.0.0.29:8080/test")
        url, ok = QInputDialog.getText(self, "HTTP/RTSP", "视频流地址：", text=default_url)
        url = url.strip()
        if not ok or not url:
            return
        self.source = SourceSpec.stream(url)
        self.source_label.setText(url)
        self._set_status(f"已选择视频流：{url}")
        CONFIG_DIR.mkdir(exist_ok=True)
        with ip_config.open("w", encoding="utf-8") as handle:
            json.dump({"ip": url}, handle, ensure_ascii=False, indent=2)

    def _run_or_toggle_pause(self) -> None:
        if self._is_running and self.worker:
            if self._is_paused:
                self.worker.resume()
                self._is_paused = False
                self.run_btn.setText("暂停")
            else:
                self.worker.pause()
                self._is_paused = True
                self.run_btn.setText("继续")
            return

        if not self.source:
            QMessageBox.information(self, "请选择来源", "请先选择图片、视频、摄像头或 HTTP/RTSP。")
            return
        if self.model_combo.count() == 0:
            QMessageBox.warning(self, "没有模型", "请将 .pt 权重文件放入 models 目录。")
            return

        self._reset_preview()
        config = self._current_config()
        self.worker_thread = QThread(self)
        self.worker = DetectionWorker(config=config, source=self.source, output_root=OUTPUT_DIR)
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.frame_ready.connect(self._on_frame_ready)
        self.worker.status_changed.connect(self._set_status)
        self.worker.progress_changed.connect(self.progress_bar.setValue)
        self.worker.error.connect(self._on_worker_error)
        self.worker.run_finished.connect(self._on_run_finished)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self._on_thread_finished)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        self._set_running_state()
        self.worker_thread.start()

    def _stop_worker(self) -> None:
        if self.worker:
            self.worker.stop()
            self.stop_btn.setEnabled(False)
            self.run_btn.setEnabled(False)
            self._set_status("正在停止，请稍候...")

    def _current_config(self) -> DetectionConfig:
        model_name = self.model_combo.currentText()
        return DetectionConfig(
            model_path=str(MODELS_DIR / model_name),
            conf=self.conf_spin.value(),
            iou=self.iou_spin.value(),
            rate_ms=self.rate_spin.value(),
            save_results=self.save_result_check.isChecked(),
            save_txt=self.save_txt_check.isChecked(),
        )

    def _open_weight_manager(self) -> None:
        dialog = WeightManagerDialog(
            self.weight_store,
            MODELS_DIR,
            PROJECT_ROOT,
            self._apply_model_from_manager,
            self,
        )
        dialog.exec()
        self._load_models()

    def _apply_model_from_manager(self, model_name: str) -> None:
        self._load_models()
        self._select_model_by_name(model_name)
        self._set_status(f"已切换模型权重：{model_name}")

    def _on_model_changed(self, model_name: str) -> None:
        self.model_card_value.setText(model_name or "--")

    def _on_frame_ready(self, result: FrameResult) -> None:
        self._append_frame_snapshot(result)
        if self._auto_follow_preview:
            self._display_frame_result(result)
        if result.progress:
            self.progress_bar.setValue(result.progress)

    def _display_frame_result(self, result: FrameResult) -> None:
        self._last_raw = result.raw_frame
        self._last_annotated = result.annotated_frame
        self._last_frame_result = result
        self._show_image(self.raw_view, result.raw_frame)
        self._show_image(self.result_view, result.annotated_frame)
        self.class_card_value.setText(str(result.class_count))
        self.target_card_value.setText(str(result.target_count))
        self.fps_card_value.setText(f"{result.fps:.1f}")
        self._update_review_button()

    def _on_worker_error(self, message: str) -> None:
        QMessageBox.warning(self, "检测失败", message)

    def _on_run_finished(self, summary) -> None:
        record = self.history_store.append(summary)
        if record.get("output_dir"):
            self.current_output_dir = Path(record["output_dir"])
        self._refresh_history_table()

    def _on_thread_finished(self) -> None:
        self.worker = None
        self.worker_thread = None
        self._set_idle_state()

    def _review_current_frame(self) -> None:
        if self._labelimg_is_running():
            QMessageBox.information(self, "labelImg 正在运行", "请先完成当前复核窗口，再创建新的补充样本。")
            return

        if self._last_raw is None or self._last_frame_result is None:
            QMessageBox.information(self, "没有可复核画面", "请先完成一次检测，或等待当前画面刷新。")
            return

        dialog = ReviewIssueDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return

        metadata = {
            "source_kind": self.source.kind if self.source else "",
            "source_name": self._last_frame_result.source_name or (self.source.display_name if self.source else ""),
            "source_path": self._last_frame_result.source_path or (self.source.path if self.source else ""),
            "model_name": self.model_combo.currentText(),
            "frame_index": self._last_frame_result.frame_index,
            "issue_reason": dialog.issue_reason,
            "note": dialog.note,
        }

        try:
            sample = export_review_sample(
                OUTPUT_DIR / "annotations",
                self._last_raw,
                self._last_frame_result.detections,
                self._last_frame_result.class_names,
                metadata,
            )
        except Exception as exc:
            QMessageBox.warning(self, "复核样本导出失败", str(exc))
            return

        executable = resolve_labelimg_executable()
        if executable is None:
            QMessageBox.warning(
                self,
                "未找到 labelImg",
                "未找到 labelImg.EXE，请确认 yolo26_gui 环境已安装 labelImg。",
            )
            return

        self.labelimg_process = QProcess(self)
        self.labelimg_process.finished.connect(self._on_labelimg_finished)
        self.labelimg_process.errorOccurred.connect(self._on_labelimg_error)
        self.labelimg_process.start(
            str(executable),
            build_labelimg_args(sample.image_path, sample.label_class_file, sample.labels_dir),
        )
        if not self.labelimg_process.waitForStarted(3000):
            message = self.labelimg_process.errorString()
            self.labelimg_process = None
            QMessageBox.warning(self, "labelImg 启动失败", message)
            self._update_review_button()
            return

        self._set_status(f"已创建复核样本：{sample.sample_id}，正在打开 labelImg")
        self._update_review_button()

    def _labelimg_is_running(self) -> bool:
        return self.labelimg_process is not None and self.labelimg_process.state() != QProcess.NotRunning

    def _on_labelimg_finished(self, exit_code: int, _exit_status) -> None:
        self.labelimg_process = None
        self._set_status(f"labelImg 已关闭，退出码：{exit_code}")
        self._update_review_button()

    def _on_labelimg_error(self, _error) -> None:
        message = self.labelimg_process.errorString() if self.labelimg_process else "未知错误"
        self._set_status(f"labelImg 运行异常：{message}")
        self.labelimg_process = None
        self._update_review_button()

    def _update_review_button(self) -> None:
        if not hasattr(self, "review_btn"):
            return
        has_frame = self._last_raw is not None and self._last_frame_result is not None
        self.review_btn.setEnabled(has_frame and not self._labelimg_is_running())

    def _append_frame_snapshot(self, result: FrameResult) -> None:
        if not self.source or self.source.kind not in {"image", "video", "batch"}:
            return

        raw_jpeg = self._encode_frame(result.raw_frame)
        annotated_jpeg = self._encode_frame(result.annotated_frame)
        if not raw_jpeg or not annotated_jpeg:
            return

        snapshot = FrameSnapshot(
            raw_jpeg=raw_jpeg,
            annotated_jpeg=annotated_jpeg,
            fps=result.fps,
            class_count=result.class_count,
            target_count=result.target_count,
            class_counts=dict(result.class_counts),
            frame_index=result.frame_index,
            progress=result.progress,
            detections=[dict(item) for item in result.detections],
            class_names=list(result.class_names),
            source_path=result.source_path,
            source_name=result.source_name,
        )
        self._frame_snapshots.append(snapshot)
        if self._auto_follow_preview or self._browser_index < 0:
            self._browser_index = len(self._frame_snapshots) - 1
        self._update_frame_browser_controls()

    def _on_frame_slider_changed(self, value: int) -> None:
        if self._browser_updating:
            return
        if not self._frame_snapshots:
            return
        self._auto_follow_preview = False
        self._show_snapshot(value)

    def _show_previous_frame(self) -> None:
        if self._browser_index > 0:
            self._auto_follow_preview = False
            self._show_snapshot(self._browser_index - 1)

    def _show_next_frame(self) -> None:
        if self._browser_index < len(self._frame_snapshots) - 1:
            self._auto_follow_preview = False
            self._show_snapshot(self._browser_index + 1)

    def _show_snapshot(self, index: int) -> None:
        if index < 0 or index >= len(self._frame_snapshots):
            return
        snapshot = self._frame_snapshots[index]
        raw = self._decode_frame(snapshot.raw_jpeg)
        annotated = self._decode_frame(snapshot.annotated_jpeg)
        if raw is None or annotated is None:
            return

        self._browser_index = index
        self._display_frame_result(snapshot.to_frame_result(raw, annotated))
        self._update_frame_browser_controls()

    def _reset_frame_browser(self) -> None:
        self._frame_snapshots.clear()
        self._browser_index = -1
        self._auto_follow_preview = True
        self._update_frame_browser_controls()

    def _update_frame_browser_controls(self) -> None:
        if not hasattr(self, "frame_slider"):
            return

        total = len(self._frame_snapshots)
        current = self._browser_index if 0 <= self._browser_index < total else 0
        self._browser_updating = True
        self.frame_slider.setRange(0, max(total - 1, 0))
        self.frame_slider.setValue(current)
        self._browser_updating = False

        self.frame_slider.setEnabled(total > 1)
        self.prev_frame_btn.setEnabled(total > 1 and self._browser_index > 0)
        self.next_frame_btn.setEnabled(total > 1 and self._browser_index < total - 1)
        self.frame_counter_label.setText(f"{current + 1}/{total}" if total else "0/0")

    @staticmethod
    def _encode_frame(frame) -> bytes:
        if frame is None:
            return b""
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        return encoded.tobytes() if ok else b""

    @staticmethod
    def _decode_frame(data: bytes):
        if not data:
            return None
        array = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(array, cv2.IMREAD_COLOR)

    def _set_running_state(self) -> None:
        self._is_running = True
        self._is_paused = False
        self.run_btn.setText("暂停")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        for widget in (
            self.file_btn,
            self.batch_btn,
            self.camera_btn,
            self.stream_btn,
            self.model_combo,
            self.weight_manager_btn,
            self.save_result_check,
            self.save_txt_check,
        ):
            widget.setEnabled(False)
        self._update_review_button()

    def _set_idle_state(self) -> None:
        self._is_running = False
        self._is_paused = False
        self.run_btn.setText("开始检测")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        for widget in (
            self.file_btn,
            self.batch_btn,
            self.camera_btn,
            self.stream_btn,
            self.model_combo,
            self.weight_manager_btn,
            self.save_result_check,
            self.save_txt_check,
        ):
            widget.setEnabled(True)
        self._update_review_button()

    def _reset_preview(self) -> None:
        self._last_raw = None
        self._last_annotated = None
        self._last_frame_result = None
        self._reset_frame_browser()
        self.raw_view.clear_frame()
        self.raw_view.setText("原始画面")
        self.raw_view.setPixmap(QPixmap())
        self.result_view.clear_frame()
        self.result_view.setText("检测结果")
        self.result_view.setPixmap(QPixmap())
        self.progress_bar.setValue(0)
        self.class_card_value.setText("--")
        self.target_card_value.setText("--")
        self.fps_card_value.setText("--")
        self._update_review_button()

    def _refresh_history_table(self) -> None:
        records = list(reversed(self.history_store.load()))
        self.history_table.setRowCount(len(records))
        for row, record in enumerate(records):
            values = [
                record.get("ended_at", ""),
                record.get("source_name", ""),
                record.get("model_name", ""),
                str(record.get("frames", "")),
                str(record.get("avg_fps", "")),
                str(record.get("total_target_events", "")),
                record.get("status", ""),
                record.get("output_dir", ""),
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col == 1:
                    item.setToolTip(json.dumps(record.get("class_counts", {}), ensure_ascii=False))
                self.history_table.setItem(row, col, item)

    def _open_outputs_dir(self) -> None:
        OUTPUT_DIR.mkdir(exist_ok=True)
        path = self.current_output_dir if self.current_output_dir.exists() else OUTPUT_DIR
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _open_history_csv(self) -> None:
        if not self.history_store.csv_path.exists():
            self.history_store.rewrite(self.history_store.load())
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.history_store.csv_path)))

    def _show_image(self, label: QLabel, frame) -> None:
        if frame is None:
            return
        pixmap = self._frame_to_pixmap(frame)
        if pixmap.isNull():
            return
        if isinstance(label, ImageLabel):
            label.set_frame(frame)
        label.setText("")
        label.setPixmap(
            pixmap.scaled(
                max(1, label.width() - 8),
                max(1, label.height() - 8),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def _open_zoom_dialog(self, title: str, frame) -> None:
        pixmap = self._frame_to_pixmap(frame)
        if pixmap.isNull():
            return
        dialog = ImageZoomDialog(title, pixmap, self)
        dialog.exec()

    @staticmethod
    def _frame_to_pixmap(frame) -> QPixmap:
        if frame is None:
            return QPixmap()

        if len(frame.shape) == 2:
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            image_format = QImage.Format_RGB888
            bytes_per_line = 3 * rgb.shape[1]
        elif frame.shape[2] == 4:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
            image_format = QImage.Format_RGBA8888
            bytes_per_line = 4 * rgb.shape[1]
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_format = QImage.Format_RGB888
            bytes_per_line = 3 * rgb.shape[1]

        height, width = rgb.shape[:2]
        image = QImage(rgb.data, width, height, bytes_per_line, image_format).copy()
        return QPixmap.fromImage(image)

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)

    @staticmethod
    def _read_json(path: Path, default: Dict) -> Dict:
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else default
        except (OSError, json.JSONDecodeError):
            return default

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._last_raw is not None:
            self._show_image(self.raw_view, self._last_raw)
        if self._last_annotated is not None:
            self._show_image(self.result_view, self._last_annotated)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._save_settings()
        if self.worker and self.worker_thread and self.worker_thread.isRunning():
            self.worker.stop()
            self.worker_thread.quit()
            self.worker_thread.wait(3000)
        event.accept()


def run_app() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()
