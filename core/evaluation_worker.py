from __future__ import annotations

import csv
import json
import math
import shutil
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from core.inference import create_predictor
from core.dataset_checker import (
    SMALL_TARGET_PX,
    class_name,
    label_path_for_image,
    load_yolo_dataset_source,
    read_yolo_label,
    safe_name,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ULTRALYTICS = PROJECT_ROOT / "ultralytics"
if LOCAL_ULTRALYTICS.exists() and str(LOCAL_ULTRALYTICS) not in sys.path:
    sys.path.insert(0, str(LOCAL_ULTRALYTICS))


FAILURE_TYPES = {
    "false_negative": "漏检",
    "false_positive": "误检",
    "class_error": "类别错误",
    "small_target_failure": "小目标失败",
}
SCENE_TAGS = ["遮挡", "模糊", "夜间", "逆光"]
PER_CLASS_FIELDS = ("class_id", "class_name", "gt", "tp", "fp", "fn", "precision", "recall", "ap")
FAILURE_FIELDS = ("case_id", "case_type", "case_label", "image_path", "artifact_path", "scenario", "reason_note", "detail")


@dataclass
class EvaluationConfig:
    model_path: str
    dataset_path: str
    output_root: str
    conf: float = 0.25
    iou: float = 0.5
    imgsz: int = 640
    device: str = "auto"
    max_images: int = 200
    small_target_px: int = SMALL_TARGET_PX


class EvaluationWorker(QObject):
    status_changed = Signal(str)
    progress_changed = Signal(int)
    report_ready = Signal(object)
    error = Signal(str)
    finished = Signal()

    def __init__(self, config: EvaluationConfig, parent: QObject | None = None):
        super().__init__(parent)
        self.config = config
        self._stop_event = threading.Event()

    @Slot()
    def run(self) -> None:
        try:
            report = evaluate_dataset(
                self.config,
                progress_callback=self.progress_changed.emit,
                status_callback=self.status_changed.emit,
                stop_event=self._stop_event,
            )
            self.report_ready.emit(report)
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.finished.emit()

    def stop(self) -> None:
        self._stop_event.set()


def evaluate_dataset(
    config: EvaluationConfig,
    predictor: Optional[Callable[[np.ndarray, Path], Iterable[Dict[str, Any]]]] = None,
    progress_callback: Optional[Callable[[int], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    stop_event: Optional[threading.Event] = None,
) -> Dict[str, Any]:
    source = load_yolo_dataset_source(config.dataset_path)
    images = source.image_paths[: max(1, int(config.max_images))]
    output_dir = make_evaluation_dir(config.output_root, source.root)
    predictions_dir = output_dir / "predictions"
    errors_dir = output_dir / "errors"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    for folder_name in list(FAILURE_TYPES.keys()) + [safe_name(tag) for tag in SCENE_TAGS]:
        (errors_dir / folder_name).mkdir(parents=True, exist_ok=True)

    if predictor is None:
        status_callback and status_callback("正在加载评测模型...")
        predictor = ModelPredictor(config)

    per_class: Dict[int, Dict[str, Any]] = {}
    ranked_predictions: Dict[int, List[Tuple[float, bool]]] = {}
    failure_cases: List[Dict[str, str]] = []
    total_gt = 0
    total_predictions = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    processed_images = 0

    for index, image_path in enumerate(images):
        if stop_event and stop_event.is_set():
            break

        status_callback and status_callback(f"评测 {index + 1}/{len(images)}: {image_path.name}")
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue

        height, width = frame.shape[:2]
        truths = load_truths(image_path, source, width, height, config.small_target_px)
        raw_predictions = list(predictor(frame, image_path))
        predictions = normalize_predictions(raw_predictions, source.names, frame.shape)

        result = match_predictions(predictions, truths, config.iou)
        annotated = render_evaluation_image(frame, predictions, truths, source.names)
        pred_path = predictions_dir / f"{safe_name(image_path.stem)}_pred.jpg"
        cv2.imwrite(str(pred_path), annotated)

        processed_images += 1
        total_gt += len(truths)
        total_predictions += len(predictions)
        total_tp += len(result["matched_pairs"])
        total_fp += len(result["false_positive_indices"])
        total_fn += len(result["false_negative_indices"])

        update_class_metrics(per_class, ranked_predictions, predictions, truths, result)

        image_case_types = case_types_for_image(truths, result)
        for case_type in image_case_types:
            case_label = FAILURE_TYPES[case_type]
            artifact_path = errors_dir / case_type / f"{safe_name(image_path.stem)}_{case_type}.jpg"
            shutil.copy2(pred_path, artifact_path)
            failure_cases.append(
                {
                    "case_id": f"{len(failure_cases) + 1:04d}",
                    "case_type": case_type,
                    "case_label": case_label,
                    "image_path": str(image_path),
                    "artifact_path": str(artifact_path),
                    "scenario": "",
                    "reason_note": "",
                    "detail": failure_detail(case_type, predictions, truths, result),
                }
            )

        if progress_callback:
            progress_callback(int(((index + 1) / max(1, len(images))) * 1000))

    per_class_rows = finalize_class_metrics(per_class, ranked_predictions, source.names)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    map_value = sum(row["ap"] for row in per_class_rows) / len(per_class_rows) if per_class_rows else 0.0
    false_positive_rate = total_fp / total_predictions if total_predictions else 0.0
    false_negative_rate = total_fn / total_gt if total_gt else 0.0

    top_fp = max(per_class_rows, key=lambda row: row["fp"], default={})
    top_fn = max(per_class_rows, key=lambda row: row["fn"], default={})
    summary = {
        "model_path": config.model_path,
        "dataset_path": config.dataset_path,
        "dataset_root": str(source.root),
        "evaluated_images": processed_images,
        "ground_truth_boxes": total_gt,
        "predicted_boxes": total_predictions,
        "true_positive": total_tp,
        "false_positive": total_fp,
        "false_negative": total_fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "mAP": round(map_value, 6),
        "false_positive_rate": round(false_positive_rate, 6),
        "false_negative_rate": round(false_negative_rate, 6),
        "top_false_positive_class": top_fp.get("class_name", ""),
        "top_false_negative_class": top_fn.get("class_name", ""),
        "iou_threshold": config.iou,
        "confidence_threshold": config.conf,
        "small_target_px": config.small_target_px,
    }
    artifacts = {
        "predictions_dir": str(predictions_dir),
        "errors_dir": str(errors_dir),
        "report_json": str(output_dir / "evaluation_report.json"),
        "summary_txt": str(output_dir / "evaluation_summary.txt"),
        "per_class_csv": str(output_dir / "per_class_metrics.csv"),
        "failure_cases_csv": str(output_dir / "failure_cases.csv"),
    }
    report = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary": summary,
        "per_class": per_class_rows,
        "failure_cases": failure_cases,
        "scene_tags": SCENE_TAGS,
        "artifacts": artifacts,
        "output_dir": str(output_dir),
    }
    write_evaluation_report(report, output_dir)
    status_callback and status_callback("评测完成")
    progress_callback and progress_callback(1000)
    return report


class ModelPredictor:
    def __init__(self, config: EvaluationConfig):
        self.predictor = create_predictor(config)

    def __call__(self, frame: np.ndarray, _image_path: Path) -> Iterable[Dict[str, Any]]:
        return self.predictor.detections(frame)


def load_truths(
    image_path: Path,
    source,
    image_width: int,
    image_height: int,
    small_target_px: int,
) -> List[Dict[str, Any]]:
    label_path = label_path_for_image(image_path, source.root, source.image_roots)
    if not label_path.exists():
        return []
    objects, _issues = read_yolo_label(label_path)
    truths = []
    for obj in objects:
        box_width, box_height = obj.pixel_size(image_width, image_height)
        truths.append(
            {
                "class_id": obj.class_id,
                "class_name": class_name(source.names, obj.class_id),
                "xyxy": obj.to_xyxy(image_width, image_height),
                "small": box_width <= small_target_px and box_height <= small_target_px,
            }
        )
    return truths


def normalize_predictions(
    predictions: Iterable[Dict[str, Any]],
    names: Sequence[str],
    image_shape: Tuple[int, ...],
) -> List[Dict[str, Any]]:
    height, width = image_shape[:2]
    normalized = []
    for prediction in predictions:
        xyxy = prediction.get("xyxy") or []
        if len(xyxy) < 4:
            continue
        try:
            x1, y1, x2, y2 = [float(value) for value in xyxy[:4]]
            class_id = int(prediction.get("class_id", 0))
            confidence = float(prediction.get("confidence", prediction.get("conf", 0.0)))
        except (TypeError, ValueError):
            continue
        x1 = max(0.0, min(float(width), x1))
        x2 = max(0.0, min(float(width), x2))
        y1 = max(0.0, min(float(height), y1))
        y2 = max(0.0, min(float(height), y2))
        if x2 <= x1 or y2 <= y1:
            continue
        normalized.append(
            {
                "class_id": class_id,
                "class_name": str(prediction.get("class_name") or class_name(names, class_id)),
                "confidence": confidence,
                "xyxy": [x1, y1, x2, y2],
            }
        )
    return sorted(normalized, key=lambda item: float(item.get("confidence", 0.0)), reverse=True)


def match_predictions(predictions: Sequence[Dict[str, Any]], truths: Sequence[Dict[str, Any]], iou_threshold: float) -> Dict[str, Any]:
    matched_truths: set[int] = set()
    matched_pairs: List[Tuple[int, int, float]] = []
    false_positive_indices: List[int] = []

    for pred_index, prediction in enumerate(predictions):
        best_truth = -1
        best_iou = 0.0
        for truth_index, truth in enumerate(truths):
            if truth_index in matched_truths:
                continue
            if int(prediction["class_id"]) != int(truth["class_id"]):
                continue
            value = box_iou(prediction["xyxy"], truth["xyxy"])
            if value > best_iou:
                best_iou = value
                best_truth = truth_index
        if best_truth >= 0 and best_iou >= iou_threshold:
            matched_truths.add(best_truth)
            matched_pairs.append((pred_index, best_truth, best_iou))
        else:
            false_positive_indices.append(pred_index)

    false_negative_indices = [index for index in range(len(truths)) if index not in matched_truths]
    class_error_pairs = []
    for pred_index in false_positive_indices:
        best_truth = -1
        best_iou = 0.0
        for truth_index in false_negative_indices:
            value = box_iou(predictions[pred_index]["xyxy"], truths[truth_index]["xyxy"])
            if value > best_iou:
                best_iou = value
                best_truth = truth_index
        if best_truth >= 0 and best_iou >= iou_threshold and int(predictions[pred_index]["class_id"]) != int(truths[best_truth]["class_id"]):
            class_error_pairs.append((pred_index, best_truth, best_iou))

    return {
        "matched_pairs": matched_pairs,
        "false_positive_indices": false_positive_indices,
        "false_negative_indices": false_negative_indices,
        "class_error_pairs": class_error_pairs,
    }


def update_class_metrics(
    per_class: Dict[int, Dict[str, Any]],
    ranked_predictions: Dict[int, List[Tuple[float, bool]]],
    predictions: Sequence[Dict[str, Any]],
    truths: Sequence[Dict[str, Any]],
    result: Dict[str, Any],
) -> None:
    matched_pred_indices = {pred_index for pred_index, _truth_index, _iou in result["matched_pairs"]}
    false_positive_indices = set(result["false_positive_indices"])
    false_negative_indices = set(result["false_negative_indices"])

    for truth_index, truth in enumerate(truths):
        row = per_class.setdefault(int(truth["class_id"]), {"gt": 0, "tp": 0, "fp": 0, "fn": 0})
        row["gt"] += 1
        if truth_index in false_negative_indices:
            row["fn"] += 1

    for pred_index, prediction in enumerate(predictions):
        class_id = int(prediction["class_id"])
        row = per_class.setdefault(class_id, {"gt": 0, "tp": 0, "fp": 0, "fn": 0})
        is_tp = pred_index in matched_pred_indices
        if is_tp:
            row["tp"] += 1
        elif pred_index in false_positive_indices:
            row["fp"] += 1
        ranked_predictions.setdefault(class_id, []).append((float(prediction.get("confidence", 0.0)), is_tp))


def finalize_class_metrics(
    per_class: Dict[int, Dict[str, Any]],
    ranked_predictions: Dict[int, List[Tuple[float, bool]]],
    names: Sequence[str],
) -> List[Dict[str, Any]]:
    class_ids = sorted(set(per_class) | set(ranked_predictions) | set(range(len(names))))
    rows: List[Dict[str, Any]] = []
    for class_id in class_ids:
        counts = per_class.get(class_id, {"gt": 0, "tp": 0, "fp": 0, "fn": 0})
        tp = int(counts.get("tp", 0))
        fp = int(counts.get("fp", 0))
        fn = int(counts.get("fn", 0))
        gt_count = int(counts.get("gt", 0))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        ap = average_precision(ranked_predictions.get(class_id, []), gt_count)
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name(names, class_id),
                "gt": gt_count,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "ap": round(ap, 6),
            }
        )
    return rows


def average_precision(ranked: Sequence[Tuple[float, bool]], gt_count: int) -> float:
    if gt_count <= 0 or not ranked:
        return 0.0
    sorted_ranked = sorted(ranked, key=lambda item: item[0], reverse=True)
    tp = 0
    fp = 0
    precisions = [1.0]
    recalls = [0.0]
    for _score, is_tp in sorted_ranked:
        if is_tp:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp) if (tp + fp) else 0.0)
        recalls.append(tp / gt_count)
    precisions.append(0.0)
    recalls.append(1.0)

    for index in range(len(precisions) - 2, -1, -1):
        precisions[index] = max(precisions[index], precisions[index + 1])

    area = 0.0
    for index in range(1, len(recalls)):
        delta = recalls[index] - recalls[index - 1]
        if delta > 0:
            area += delta * precisions[index]
    return area


def case_types_for_image(truths: Sequence[Dict[str, Any]], result: Dict[str, Any]) -> List[str]:
    cases = []
    if result["false_negative_indices"]:
        cases.append("false_negative")
    if result["false_positive_indices"]:
        cases.append("false_positive")
    if result["class_error_pairs"]:
        cases.append("class_error")
    if any(truths[index].get("small") for index in result["false_negative_indices"]):
        cases.append("small_target_failure")
    return cases


def failure_detail(
    case_type: str,
    predictions: Sequence[Dict[str, Any]],
    truths: Sequence[Dict[str, Any]],
    result: Dict[str, Any],
) -> str:
    if case_type == "false_negative":
        names = [str(truths[index].get("class_name", "")) for index in result["false_negative_indices"]]
        return "missed: " + ", ".join(names[:8])
    if case_type == "false_positive":
        names = [str(predictions[index].get("class_name", "")) for index in result["false_positive_indices"]]
        return "false predictions: " + ", ".join(names[:8])
    if case_type == "class_error":
        details = []
        for pred_index, truth_index, iou_value in result["class_error_pairs"][:8]:
            details.append(
                f"pred={predictions[pred_index].get('class_name')} gt={truths[truth_index].get('class_name')} iou={iou_value:.2f}"
            )
        return "; ".join(details)
    if case_type == "small_target_failure":
        names = [
            str(truths[index].get("class_name", ""))
            for index in result["false_negative_indices"]
            if truths[index].get("small")
        ]
        return "small missed: " + ", ".join(names[:8])
    return ""


def render_evaluation_image(
    frame: np.ndarray,
    predictions: Sequence[Dict[str, Any]],
    truths: Sequence[Dict[str, Any]],
    names: Sequence[str],
) -> np.ndarray:
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    line_width = max(2, round((height + width) * 0.0012))
    font_scale = max(0.48, min(0.78, (height + width) / 1500.0))
    font_thickness = 2
    label_positions: List[Tuple[int, int, int, int]] = []

    for truth in truths:
        draw_box(
            annotated,
            truth["xyxy"],
            (40, 170, 80),
            f"GT {class_name(names, int(truth['class_id']))}",
            label_positions,
            font_scale,
            font_thickness,
            line_width,
        )
    for prediction in predictions:
        label = f"P {class_name(names, int(prediction['class_id']))} {float(prediction.get('confidence', 0.0)):.2f}"
        draw_box(
            annotated,
            prediction["xyxy"],
            label_color(int(prediction["class_id"])),
            label,
            label_positions,
            font_scale,
            font_thickness,
            line_width,
        )
    return annotated


def draw_box(
    image: np.ndarray,
    xyxy: Sequence[float],
    color: Tuple[int, int, int],
    label: str,
    label_positions: List[Tuple[int, int, int, int]] | None = None,
    font_scale: float | None = None,
    font_thickness: int = 2,
    line_width: int = 2,
) -> None:
    height, width = image.shape[:2]
    if font_scale is None:
        font_scale = max(0.48, min(0.78, (height + width) / 1500.0))
    if label_positions is None:
        label_positions = []

    x1, y1, x2, y2 = [int(round(value)) for value in xyxy[:4]]
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return

    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, line_width, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.62, image, 0.38, 0, image)

    label_x, label_y, label_width, label_height = find_label_position(
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
    label_cx = label_x + label_width // 2
    label_cy = label_y + label_height // 2
    box_cx = (x1 + x2) // 2
    box_cy = (y1 + y2) // 2
    distance = ((label_cx - box_cx) ** 2 + (label_cy - box_cy) ** 2) ** 0.5
    if distance > max(x2 - x1, y2 - y1) * 1.5:
        cv2.line(image, (box_cx, box_cy), (label_cx, label_cy), color, 1, cv2.LINE_AA)

    label_positions.append((label_x, label_y, label_width, label_height))
    draw_translucent_label(image, label, label_x, label_y, color, font_scale, font_thickness)


def find_label_position(
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


def draw_translucent_label(
    image: np.ndarray,
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


def label_color(class_id: int) -> Tuple[int, int, int]:
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


def write_evaluation_report(report: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    with (output_dir / "per_class_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PER_CLASS_FIELDS)
        writer.writeheader()
        for row in report.get("per_class", []):
            writer.writerow({field: row.get(field, "") for field in PER_CLASS_FIELDS})

    with (output_dir / "failure_cases.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FAILURE_FIELDS)
        writer.writeheader()
        for row in report.get("failure_cases", []):
            writer.writerow({field: row.get(field, "") for field in FAILURE_FIELDS})

    summary = report.get("summary", {})
    lines = [
        "YOLO evaluation summary",
        f"Created: {report.get('created_at', '')}",
        f"Model: {summary.get('model_path', '')}",
        f"Dataset: {summary.get('dataset_root', '')}",
        "",
        f"Images: {summary.get('evaluated_images', 0)}",
        f"GT boxes: {summary.get('ground_truth_boxes', 0)}",
        f"Pred boxes: {summary.get('predicted_boxes', 0)}",
        f"Precision: {summary.get('precision', 0):.4f}",
        f"Recall: {summary.get('recall', 0):.4f}",
        f"mAP: {summary.get('mAP', 0):.4f}",
        f"False positive rate: {summary.get('false_positive_rate', 0):.4f}",
        f"False negative rate: {summary.get('false_negative_rate', 0):.4f}",
        f"Most false positives: {summary.get('top_false_positive_class', '')}",
        f"Most false negatives: {summary.get('top_false_negative_class', '')}",
        "",
        "Failure cases can be annotated in failure_cases.csv using scenario/reason_note.",
    ]
    (output_dir / "evaluation_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in a[:4]]
    bx1, by1, bx2, by2 = [float(value) for value in b[:4]]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def make_evaluation_dir(output_root: str | Path, dataset_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(output_root) / "evaluations" / f"{stamp}_{safe_name(dataset_root.name or 'dataset')}"


def as_list(values):
    if values is None:
        return []
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)


def model_class_name(names, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, names.get(str(class_id), class_id)))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)
