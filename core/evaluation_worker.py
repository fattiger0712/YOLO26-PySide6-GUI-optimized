from __future__ import annotations

import csv
import json
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
    for folder_name in list(FAILURE_TYPES.values()) + SCENE_TAGS:
        (errors_dir / folder_name).mkdir(parents=True, exist_ok=True)

    if predictor is None:
        status_callback and status_callback("正在加载评测模型...")
        predictor = UltralyticsPredictor(config)

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
            artifact_path = errors_dir / case_label / f"{safe_name(image_path.stem)}_{case_type}.jpg"
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


class UltralyticsPredictor:
    def __init__(self, config: EvaluationConfig):
        from ultralytics import YOLO

        self.config = config
        self.model = YOLO(config.model_path)

    def __call__(self, frame: np.ndarray, _image_path: Path) -> Iterable[Dict[str, Any]]:
        result = self.model.predict(
            frame,
            save=False,
            save_txt=False,
            imgsz=self.config.imgsz,
            conf=self.config.conf,
            iou=self.config.iou,
            device=self._device(),
            verbose=False,
        )[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or getattr(boxes, "xyxy", None) is None:
            return []

        names = getattr(result, "names", None) or getattr(self.model, "names", {})
        xyxy_values = as_list(boxes.xyxy)
        class_values = as_list(getattr(boxes, "cls", []))
        conf_values = as_list(getattr(boxes, "conf", []))
        detections = []
        for index, box in enumerate(xyxy_values):
            if len(box) < 4:
                continue
            class_id = int(class_values[index]) if index < len(class_values) else 0
            confidence = float(conf_values[index]) if index < len(conf_values) else 0.0
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": model_class_name(names, class_id),
                    "confidence": confidence,
                    "xyxy": [float(value) for value in box[:4]],
                }
            )
        return detections

    def _device(self) -> str:
        if self.config.device != "auto":
            return self.config.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"


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
    for truth in truths:
        draw_box(annotated, truth["xyxy"], (40, 170, 80), f"GT {class_name(names, int(truth['class_id']))}")
    for prediction in predictions:
        label = f"P {class_name(names, int(prediction['class_id']))} {float(prediction.get('confidence', 0.0)):.2f}"
        draw_box(annotated, prediction["xyxy"], (35, 120, 240), label)
    return annotated


def draw_box(image: np.ndarray, xyxy: Sequence[float], color: Tuple[int, int, int], label: str) -> None:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in xyxy[:4]]
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    label_width = min(width - x1, text_width + 8)
    y0 = y1 - text_height - baseline - 6 if y1 > text_height + 8 else y1
    cv2.rectangle(image, (x1, y0), (x1 + label_width, y0 + text_height + baseline + 6), color, -1)
    cv2.putText(image, label, (x1 + 4, y0 + text_height + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)


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
