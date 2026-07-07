from __future__ import annotations

import threading
import time
import uuid
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
from PySide6.QtCore import QObject, Signal, Slot

from core.inference import create_predictor
from core.models import (
    DetectionConfig,
    FrameResult,
    RunSummary,
    SourceSpec,
    detect_file_kind,
    iter_supported_media,
    merge_counts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DetectionWorker(QObject):
    frame_ready = Signal(object)
    status_changed = Signal(str)
    progress_changed = Signal(int)
    run_finished = Signal(object)
    error = Signal(str)
    finished = Signal()

    def __init__(
        self,
        config: DetectionConfig,
        source: SourceSpec,
        output_root: str | Path | None = None,
        parent: QObject | None = None,
    ):
        super().__init__(parent)
        self.config = config
        self.source = source
        self.output_root = Path(output_root or PROJECT_ROOT / "outputs")
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._predictor = None
        self._writer = None
        self._run_dir = ""

    @Slot()
    def run(self) -> None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        started_at = datetime.now()
        frames = 0
        max_targets = 0
        final_class_count = 0
        final_target_count = 0
        total_target_events = 0
        aggregate_counts: Dict[str, int] = {}
        fps_sum = 0.0
        status = "completed"
        self._run_dir = ""

        try:
            self.output_root.mkdir(parents=True, exist_ok=True)
            if self.config.save_results or self.config.save_txt:
                run_dir = self.output_root / "runs" / run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                if self.config.save_txt:
                    (run_dir / "labels").mkdir(parents=True, exist_ok=True)
                self._run_dir = str(run_dir)

            self.status_changed.emit("正在加载模型...")
            self._predictor = create_predictor(self.config)
            self.status_changed.emit("正在检测...")

            if self.source.kind == "batch":
                (
                    frames,
                    fps_sum,
                    max_targets,
                    final_class_count,
                    final_target_count,
                    total_target_events,
                    aggregate_counts,
                    status,
                ) = self._run_batch_source()
            elif self.source.kind == "image":
                frame = cv2.imread(self.source.path)
                if frame is None:
                    raise RuntimeError("图片读取失败，请检查文件路径。")
                result, fps, counts, target_count = self._predict(frame)
                annotated = self._render_saved_result(frame, result)
                frames = 1
                max_targets = target_count
                final_target_count = target_count
                final_class_count = len(counts)
                total_target_events = target_count
                aggregate_counts = counts.copy()
                fps_sum = fps
                self._save_image_result(frame, result)
                self._save_labels(result, frames)
                self.progress_changed.emit(1000)
                self.frame_ready.emit(
                    FrameResult(
                        raw_frame=frame,
                        annotated_frame=annotated,
                        fps=fps,
                        class_count=final_class_count,
                        target_count=target_count,
                        class_counts=counts,
                        frame_index=frames,
                        progress=1000,
                        detections=self._frame_detections(result),
                        class_names=self._result_class_names(result),
                        source_path=self.source.path,
                        source_name=self.source.display_name,
                    )
                )
            else:
                (
                    frames,
                    fps_sum,
                    max_targets,
                    final_class_count,
                    final_target_count,
                    total_target_events,
                    aggregate_counts,
                    status,
                ) = self._run_capture_source()

            if status == "completed":
                self.status_changed.emit("检测完成")
            elif status == "stopped":
                self.status_changed.emit("检测已停止")

        except Exception as exc:
            status = "failed"
            self.error.emit(str(exc))
            self.status_changed.emit(f"检测失败：{exc}")
        finally:
            self._release_writer()
            ended_at = datetime.now()
            duration = max((ended_at - started_at).total_seconds(), 0.001)
            avg_fps = fps_sum / frames if frames else 0.0
            summary = RunSummary(
                run_id=run_id,
                started_at=started_at.strftime("%Y-%m-%d %H:%M:%S"),
                ended_at=ended_at.strftime("%Y-%m-%d %H:%M:%S"),
                source_type=self.source.kind,
                source_name=self.source.display_name or self.source.path,
                model_name=self.config.model_name,
                conf=self.config.conf,
                iou=self.config.iou,
                rate_ms=self.config.rate_ms,
                save_results=self.config.save_results,
                save_txt=self.config.save_txt,
                frames=frames,
                duration_seconds=duration,
                avg_fps=avg_fps,
                max_targets=max_targets,
                final_class_count=final_class_count,
                final_target_count=final_target_count,
                total_target_events=total_target_events,
                class_counts=aggregate_counts,
                status=status,
                output_dir=self._run_dir,
            )
            self.run_finished.emit(summary)
            self.finished.emit()

    def pause(self) -> None:
        if not self._stop_event.is_set():
            self._pause_event.set()
            self.status_changed.emit("已暂停")

    def resume(self) -> None:
        self._pause_event.clear()
        if not self._stop_event.is_set():
            self.status_changed.emit("正在检测...")

    def stop(self) -> None:
        self._stop_event.set()
        self._pause_event.clear()
        self.status_changed.emit("正在停止...")

    def _run_batch_source(
        self,
    ) -> Tuple[int, float, int, int, int, int, Dict[str, int], str]:
        batch_root = Path(self.source.path)
        files = iter_supported_media(batch_root)
        if not files:
            raise RuntimeError("No supported image or video files found in the selected folder.")

        frames = 0
        fps_sum = 0.0
        max_targets = 0
        final_class_count = 0
        final_target_count = 0
        total_target_events = 0
        aggregate_counts: Dict[str, int] = {}
        status = "completed"
        total_files = len(files)

        for file_index, media_path in enumerate(files):
            if self._stop_event.is_set():
                status = "stopped"
                break

            self._wait_if_paused()
            if self._stop_event.is_set():
                status = "stopped"
                break

            kind = detect_file_kind(media_path)
            relative_stem = self._batch_relative_stem(batch_root, media_path)
            self.status_changed.emit(
                f"Batch {file_index + 1}/{total_files}: {media_path.name}"
            )

            if kind == "image":
                frame = cv2.imread(str(media_path))
                if frame is None:
                    continue

                result, fps, counts, target_count = self._predict(frame)
                annotated = self._render_saved_result(frame, result)
                frames += 1
                fps_sum += fps
                final_target_count = target_count
                final_class_count = len(counts)
                total_target_events += target_count
                max_targets = max(max_targets, target_count)
                merge_counts(aggregate_counts, counts)

                self._save_batch_image_result(frame, result, relative_stem)
                self._save_batch_labels(result, relative_stem, None)

                progress = int((file_index + 1) / total_files * 1000)
                self.progress_changed.emit(progress)
                self.frame_ready.emit(
                    FrameResult(
                        raw_frame=frame,
                        annotated_frame=annotated,
                        fps=fps,
                        class_count=final_class_count,
                        target_count=target_count,
                        class_counts=counts,
                        frame_index=frames,
                        progress=progress,
                        detections=self._frame_detections(result),
                        class_names=self._result_class_names(result),
                        source_path=str(media_path),
                        source_name=media_path.name,
                    )
                )
                continue

            cap = cv2.VideoCapture(str(media_path))
            if not cap.isOpened():
                continue

            writer = None
            source_fps = cap.get(cv2.CAP_PROP_FPS) or 0
            video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            local_frame = 0
            try:
                while not self._stop_event.is_set():
                    self._wait_if_paused()
                    if self._stop_event.is_set():
                        status = "stopped"
                        break

                    tick = time.perf_counter()
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break

                    local_frame += 1
                    frames += 1
                    result, fps, counts, target_count = self._predict(frame)
                    annotated = self._render_saved_result(frame, result)
                    fps_sum += fps
                    final_target_count = target_count
                    final_class_count = len(counts)
                    total_target_events += target_count
                    max_targets = max(max_targets, target_count)
                    merge_counts(aggregate_counts, counts)

                    writer = self._write_batch_video_frame(
                        writer, frame, result, relative_stem, source_fps
                    )
                    self._save_batch_labels(result, relative_stem, local_frame)

                    if video_frame_count > 0:
                        file_progress = local_frame / video_frame_count
                    else:
                        file_progress = 0
                    progress = int(
                        min(1000, ((file_index + file_progress) / total_files) * 1000)
                    )
                    self.progress_changed.emit(progress)
                    self.frame_ready.emit(
                        FrameResult(
                            raw_frame=frame,
                            annotated_frame=annotated,
                            fps=fps,
                            class_count=final_class_count,
                            target_count=target_count,
                            class_counts=counts,
                            frame_index=frames,
                            progress=progress,
                            detections=self._frame_detections(result),
                            class_names=self._result_class_names(result),
                            source_path=str(media_path),
                            source_name=f"{media_path.name} frame {local_frame}",
                        )
                    )
                    self._sleep_for_rate(tick)
            finally:
                cap.release()
                if writer is not None:
                    writer.release()

            if status == "stopped":
                break

        if status == "completed":
            self.progress_changed.emit(1000)
        return (
            frames,
            fps_sum,
            max_targets,
            final_class_count,
            final_target_count,
            total_target_events,
            aggregate_counts,
            status,
        )

    def _run_capture_source(
        self,
    ) -> Tuple[int, float, int, int, int, int, Dict[str, int], str]:
        cap = self._open_capture()
        frames = 0
        fps_sum = 0.0
        max_targets = 0
        final_class_count = 0
        final_target_count = 0
        total_target_events = 0
        aggregate_counts: Dict[str, int] = {}
        status = "completed"

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 0

        try:
            while not self._stop_event.is_set():
                self._wait_if_paused()
                if self._stop_event.is_set():
                    status = "stopped"
                    break

                tick = time.perf_counter()
                ok, frame = cap.read()
                if not ok or frame is None:
                    if self.source.kind in {"camera", "stream"} and frames == 0:
                        raise RuntimeError("视频源没有返回画面，请检查摄像头或 RTSP 地址。")
                    break

                frames += 1
                result, fps, counts, target_count = self._predict(frame)
                annotated = self._render_saved_result(frame, result)
                fps_sum += fps
                final_target_count = target_count
                final_class_count = len(counts)
                total_target_events += target_count
                max_targets = max(max_targets, target_count)
                merge_counts(aggregate_counts, counts)

                self._save_video_frame(frame, result, source_fps)
                self._save_labels(result, frames)

                progress = 0
                if total_frames > 0:
                    progress = min(1000, int(frames / total_frames * 1000))
                    self.progress_changed.emit(progress)

                self.frame_ready.emit(
                    FrameResult(
                        raw_frame=frame,
                        annotated_frame=annotated,
                        fps=fps,
                        class_count=final_class_count,
                        target_count=target_count,
                        class_counts=counts,
                        frame_index=frames,
                        progress=progress,
                        detections=self._frame_detections(result),
                        class_names=self._result_class_names(result),
                        source_path=self.source.path,
                        source_name=self.source.display_name,
                    )
                )

                self._sleep_for_rate(tick)

            if self._stop_event.is_set():
                status = "stopped"
            elif total_frames > 0:
                self.progress_changed.emit(1000)
        finally:
            cap.release()

        return (
            frames,
            fps_sum,
            max_targets,
            final_class_count,
            final_target_count,
            total_target_events,
            aggregate_counts,
            status,
        )

    def _open_capture(self):
        if self.source.kind == "video":
            cap = cv2.VideoCapture(self.source.path)
        elif self.source.kind == "camera":
            cap = cv2.VideoCapture(self.source.camera_index)
        elif self.source.kind == "stream":
            cap = cv2.VideoCapture(self.source.path)
        else:
            raise RuntimeError(f"不支持的检测源：{self.source.kind}")

        if not cap.isOpened():
            raise RuntimeError("视频源打开失败，请检查文件、摄像头或网络地址。")
        return cap

    def _predict(self, frame):
        if self._predictor is None:
            raise RuntimeError("Model predictor is not loaded.")
        start = time.perf_counter()
        result = self._predictor.predict(frame)
        elapsed = max(time.perf_counter() - start, 1e-6)
        fps = 1.0 / elapsed
        counts = self._class_counts(result)
        target_count = sum(counts.values())
        return result, fps, counts, target_count

    def _frame_detections(self, result) -> List[Dict]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or getattr(boxes, "xyxy", None) is None:
            return []

        xyxy_values = self._as_plain_list(boxes.xyxy)
        class_values = self._as_plain_list(getattr(boxes, "cls", []))
        conf_values = self._as_plain_list(getattr(boxes, "conf", []))
        names = self._result_names(result)

        detections = []
        for index, box in enumerate(xyxy_values):
            if len(box) < 4:
                continue
            class_id = int(class_values[index]) if index < len(class_values) else 0
            confidence = float(conf_values[index]) if index < len(conf_values) else 0.0
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": self._class_name(names, class_id),
                    "confidence": confidence,
                    "xyxy": [float(value) for value in box[:4]],
                }
            )
        return detections

    def _result_class_names(self, result) -> List[str]:
        names = self._result_names(result)
        if isinstance(names, dict):
            indexed_names = []
            int_keys = []
            for key in names.keys():
                try:
                    int_keys.append(int(key))
                except (TypeError, ValueError):
                    continue
            if not int_keys:
                return [str(value) for value in names.values()]
            for index in range(max(int_keys) + 1):
                indexed_names.append(str(names.get(index, names.get(str(index), index))))
            return indexed_names
        if isinstance(names, (list, tuple)):
            return [str(name) for name in names]
        return []

    def _class_counts(self, result) -> Dict[str, int]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or getattr(boxes, "cls", None) is None:
            return {}

        names = self._result_names(result)
        counts: Dict[str, int] = {}
        for class_id in [int(value) for value in self._as_plain_list(boxes.cls)]:
            name = self._class_name(names, class_id)
            counts[name] = counts.get(name, 0) + 1
        return counts

    def _result_names(self, result):
        names = getattr(result, "names", None)
        if names:
            return names
        if self._predictor is not None:
            return getattr(self._predictor, "names", {})
        return {}

    @staticmethod
    def _class_name(names, class_id: int) -> str:
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)

    def _save_image_result(self, frame, result) -> None:
        if not self.config.save_results or not self._run_dir:
            return
        annotated = self._render_saved_result(frame, result)
        stem = Path(self.source.path).stem or "image"
        cv2.imwrite(str(Path(self._run_dir) / f"{stem}_detected.jpg"), annotated)

    def _save_video_frame(self, frame, result, source_fps: float) -> None:
        if not self.config.save_results or not self._run_dir:
            return

        annotated = self._render_saved_result(frame, result)
        if self._writer is None:
            height, width = annotated.shape[:2]
            if source_fps <= 0 or source_fps > 120:
                source_fps = max(1.0, min(60.0, 1000.0 / max(self.config.rate_ms, 1)))
            filename = self._source_stem() + "_detected.mp4"
            output_path = Path(self._run_dir) / filename
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(str(output_path), fourcc, source_fps, (width, height))

        self._writer.write(annotated)

    def _render_saved_result(self, frame, result):
        annotated = frame.copy()
        boxes = getattr(result, "boxes", None)
        if boxes is None or getattr(boxes, "xyxy", None) is None:
            return annotated

        xyxy_values = self._as_plain_list(boxes.xyxy)
        class_values = self._as_plain_list(getattr(boxes, "cls", []))
        conf_values = self._as_plain_list(getattr(boxes, "conf", []))
        if not xyxy_values:
            return annotated

        names = self._result_names(result)
        height, width = annotated.shape[:2]
        line_width = max(1, round((height + width) * 0.0012))
        line_width = max(2, line_width)
        font_scale = max(0.48, min(0.78, (height + width) / 1500.0))
        font_thickness = 2
        label_positions: List[Tuple[int, int, int, int]] = []

        for index, box in enumerate(xyxy_values):
            if len(box) < 4:
                continue

            class_id = int(class_values[index]) if index < len(class_values) else 0
            confidence = float(conf_values[index]) if index < len(conf_values) else 0.0
            color = self._label_color(class_id)

            x1, y1, x2, y2 = [int(round(value)) for value in box[:4]]
            x1 = max(0, min(width - 1, x1))
            x2 = max(0, min(width - 1, x2))
            y1 = max(0, min(height - 1, y1))
            y2 = max(0, min(height - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            overlay = annotated.copy()
            cv2.rectangle(
                overlay,
                (x1, y1),
                (x2, y2),
                color,
                thickness=line_width,
                lineType=cv2.LINE_AA,
            )
            cv2.addWeighted(overlay, 0.62, annotated, 0.38, 0, annotated)

            label = f"{self._class_name(names, class_id)} {confidence:.2f}"
            label_x, label_y, label_w, label_h = self._find_label_position(
                x1,
                y1,
                x2,
                y2,
                label,
                font_scale,
                font_thickness,
                width,
                height,
                label_positions,
            )

            label_cx = label_x + label_w // 2
            label_cy = label_y + label_h // 2
            box_cx = (x1 + x2) // 2
            box_cy = (y1 + y2) // 2
            distance = ((label_cx - box_cx) ** 2 + (label_cy - box_cy) ** 2) ** 0.5
            if distance > max(x2 - x1, y2 - y1) * 1.5:
                cv2.line(annotated, (box_cx, box_cy), (label_cx, label_cy), color, 1, cv2.LINE_AA)

            label_positions.append((label_x, label_y, label_w, label_h))
            self._draw_translucent_label(
                annotated,
                label,
                label_x,
                label_y,
                color,
                font_scale,
                font_thickness,
            )
        return annotated

    @staticmethod
    def _find_label_position(
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        label: str,
        font_scale: float,
        font_thickness: int,
        image_width: int,
        image_height: int,
        existing_labels: List[Tuple[int, int, int, int]],
    ) -> Tuple[int, int, int, int]:
        padding_x = 4
        padding_y = 3
        gap = 5
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            font_thickness,
        )
        label_width = min(max(1, image_width), text_width + padding_x * 2)
        label_height = min(max(1, image_height), text_height + baseline + padding_y * 2)
        max_x = max(0, image_width - label_width)
        max_y = max(0, image_height - label_height)

        candidates: List[Tuple[int, int]] = []
        for offset_x in (0, -label_width // 3, label_width // 3, -label_width // 2, label_width // 2):
            candidates.append((x1 + offset_x, y1 - label_height - gap))
        for offset_x in (0, -label_width // 3, label_width // 3, -label_width // 2, label_width // 2):
            candidates.append((x1 + offset_x, y2 + gap))
        for offset_y in (0, -label_height // 3, label_height // 3):
            candidates.append((x1 - label_width - gap, y1 + offset_y))
        for offset_y in (0, -label_height // 3, label_height // 3):
            candidates.append((x2 + gap, y1 + offset_y))
        candidates.extend(
            [
                (x1 + 2, y1 + 2),
                (x2 - label_width - 2, y1 + 2),
                (x1 + 2, y2 - label_height - 2),
                (x2 - label_width - 2, y2 - label_height - 2),
            ]
        )

        def clamp(candidate_x: int, candidate_y: int) -> Tuple[int, int]:
            return max(0, min(candidate_x, max_x)), max(0, min(candidate_y, max_y))

        def overlaps(candidate_x: int, candidate_y: int) -> bool:
            for existing_x, existing_y, existing_w, existing_h in existing_labels:
                if not (
                    candidate_x + label_width + 2 < existing_x
                    or candidate_x > existing_x + existing_w + 2
                    or candidate_y + label_height + 2 < existing_y
                    or candidate_y > existing_y + existing_h + 2
                ):
                    return True
            return False

        for candidate_x, candidate_y in candidates:
            checked_x, checked_y = clamp(int(candidate_x), int(candidate_y))
            if not overlaps(checked_x, checked_y):
                return checked_x, checked_y, label_width, label_height

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        for radius in range(0, max(image_width, image_height) + 20, 20):
            for angle in range(0, 360, 45):
                checked_x = int(center_x + radius * math.cos(math.radians(angle)) - label_width // 2)
                checked_y = int(center_y + radius * math.sin(math.radians(angle)) - label_height // 2)
                checked_x, checked_y = clamp(checked_x, checked_y)
                if not overlaps(checked_x, checked_y):
                    return checked_x, checked_y, label_width, label_height

        fallback_x, fallback_y = clamp(x1, y1 - label_height)
        return fallback_x, fallback_y, label_width, label_height

    @staticmethod
    def _draw_translucent_label(
        image,
        label: str,
        x: int,
        y: int,
        color: Tuple[int, int, int],
        font_scale: float,
        font_thickness: int,
    ) -> None:
        height, width = image.shape[:2]
        padding_x = 4
        padding_y = 3
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            font_thickness,
        )
        label_width = min(width, text_width + padding_x * 2)
        label_height = min(height, text_height + baseline + padding_y * 2)
        if label_width <= 0 or label_height <= 0:
            return

        x1 = max(0, min(x, width - label_width))
        y1 = max(0, min(y, height - label_height))
        x2 = min(width - 1, x1 + label_width)
        y2 = min(height - 1, y1 + label_height)

        overlay = image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.45, image, 0.55, 0, image)

        text_color = (20, 20, 20) if sum(color) > 420 else (255, 255, 255)
        text_origin = (x1 + padding_x, min(y2 - baseline, y1 + padding_y + text_height))
        cv2.putText(
            image,
            label,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            thickness=font_thickness,
            lineType=cv2.LINE_AA,
        )

    @staticmethod
    def _label_color(class_id: int) -> Tuple[int, int, int]:
        palette = (
            (56, 128, 255),
            (40, 180, 99),
            (243, 156, 18),
            (231, 76, 60),
            (155, 89, 182),
            (26, 188, 156),
            (52, 152, 219),
            (241, 196, 15),
        )
        return palette[class_id % len(palette)]

    @staticmethod
    def _as_plain_list(values):
        if values is None:
            return []
        if hasattr(values, "detach"):
            values = values.detach().cpu().numpy()
        if hasattr(values, "tolist"):
            return values.tolist()
        return list(values)

    def _save_labels(self, result, frame_index: int) -> None:
        if not self.config.save_txt or not self._run_dir:
            return
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return

        if self.source.kind == "image":
            label_name = Path(self.source.path).stem + ".txt"
        else:
            label_name = f"frame_{frame_index:06d}.txt"
        label_path = Path(self._run_dir) / "labels" / label_name

        classes = [int(value) for value in self._as_plain_list(getattr(boxes, "cls", []))]
        xywhn = self._as_plain_list(getattr(boxes, "xywhn", []))
        confs = self._as_plain_list(getattr(boxes, "conf", []))
        with label_path.open("w", encoding="utf-8") as handle:
            for class_id, box, conf in zip(classes, xywhn, confs):
                x, y, w, h = box
                handle.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {conf:.6f}\n")

    def _batch_relative_stem(self, batch_root: Path, media_path: Path) -> str:
        try:
            relative = media_path.relative_to(batch_root).with_suffix("")
        except ValueError:
            relative = media_path.with_suffix("")
        parts = [self._safe_name(part) for part in relative.parts if part]
        return "__".join(parts) or self._safe_name(media_path.stem)

    @staticmethod
    def _safe_name(value: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)
        return cleaned.strip("._") or "item"

    def _save_batch_image_result(self, frame, result, relative_stem: str) -> None:
        if not self.config.save_results or not self._run_dir:
            return
        annotated = self._render_saved_result(frame, result)
        output_path = Path(self._run_dir) / f"{relative_stem}_detected.jpg"
        cv2.imwrite(str(output_path), annotated)

    def _write_batch_video_frame(
        self,
        writer,
        frame,
        result,
        relative_stem: str,
        source_fps: float,
    ):
        if not self.config.save_results or not self._run_dir:
            return writer

        annotated = self._render_saved_result(frame, result)
        if writer is None:
            height, width = annotated.shape[:2]
            if source_fps <= 0 or source_fps > 120:
                source_fps = max(1.0, min(60.0, 1000.0 / max(self.config.rate_ms, 1)))
            output_path = Path(self._run_dir) / f"{relative_stem}_detected.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, source_fps, (width, height))
        writer.write(annotated)
        return writer

    def _save_batch_labels(self, result, relative_stem: str, frame_index: int | None) -> None:
        if not self.config.save_txt or not self._run_dir:
            return
        labels_dir = Path(self._run_dir) / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        if frame_index is None:
            label_path = labels_dir / f"{relative_stem}.txt"
        else:
            label_path = labels_dir / f"{relative_stem}_frame_{frame_index:06d}.txt"
        self._write_yolo_labels(result, label_path)

    def _write_yolo_labels(self, result, label_path: Path) -> None:
        boxes = getattr(result, "boxes", None)
        with label_path.open("w", encoding="utf-8") as handle:
            if boxes is None:
                return
            classes = [int(value) for value in self._as_plain_list(getattr(boxes, "cls", []))]
            xywhn = self._as_plain_list(getattr(boxes, "xywhn", []))
            confs = self._as_plain_list(getattr(boxes, "conf", []))
            for class_id, box, conf in zip(classes, xywhn, confs):
                x, y, w, h = box
                handle.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {conf:.6f}\n")

    def _source_stem(self) -> str:
        if self.source.kind == "camera":
            return f"camera_{self.source.camera_index}"
        if self.source.kind == "stream":
            return "stream"
        return Path(self.source.path).stem or self.source.kind

    def _sleep_for_rate(self, tick: float) -> None:
        if self.config.rate_ms <= 0:
            return
        elapsed_ms = (time.perf_counter() - tick) * 1000
        remain = self.config.rate_ms - elapsed_ms
        if remain > 0:
            time.sleep(remain / 1000)

    def _wait_if_paused(self) -> None:
        while self._pause_event.is_set() and not self._stop_event.is_set():
            time.sleep(0.05)

    def _release_writer(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
