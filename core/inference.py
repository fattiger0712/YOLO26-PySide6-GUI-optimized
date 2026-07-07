from __future__ import annotations

import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ULTRALYTICS = PROJECT_ROOT / "ultralytics"
if LOCAL_ULTRALYTICS.exists() and str(LOCAL_ULTRALYTICS) not in sys.path:
    sys.path.insert(0, str(LOCAL_ULTRALYTICS))

SUPPORTED_MODEL_SUFFIXES = {".pt", ".onnx"}


@dataclass
class LetterboxInfo:
    ratio: float
    pad_x: float
    pad_y: float
    input_width: int
    input_height: int


class InferenceBoxes:
    def __init__(
        self,
        xyxy: Sequence[Sequence[float]] | np.ndarray,
        cls: Sequence[float] | np.ndarray,
        conf: Sequence[float] | np.ndarray,
        image_shape: Tuple[int, ...],
    ):
        xyxy_array = np.asarray(xyxy, dtype=np.float32)
        if xyxy_array.size == 0:
            xyxy_array = np.zeros((0, 4), dtype=np.float32)
        self.xyxy = xyxy_array.reshape(-1, 4)
        self.cls = np.asarray(cls, dtype=np.float32).reshape(-1)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.image_shape = image_shape

    @property
    def xywhn(self) -> np.ndarray:
        if len(self.xyxy) == 0:
            return np.zeros((0, 4), dtype=np.float32)

        height, width = self.image_shape[:2]
        xywh = np.zeros_like(self.xyxy, dtype=np.float32)
        xywh[:, 0] = (self.xyxy[:, 0] + self.xyxy[:, 2]) / 2.0 / max(float(width), 1.0)
        xywh[:, 1] = (self.xyxy[:, 1] + self.xyxy[:, 3]) / 2.0 / max(float(height), 1.0)
        xywh[:, 2] = (self.xyxy[:, 2] - self.xyxy[:, 0]) / max(float(width), 1.0)
        xywh[:, 3] = (self.xyxy[:, 3] - self.xyxy[:, 1]) / max(float(height), 1.0)
        return np.clip(xywh, 0.0, 1.0)


@dataclass
class InferenceResult:
    boxes: InferenceBoxes
    names: Dict[int, str]


class BasePredictor:
    names: Dict[int, str]

    def predict(self, frame: np.ndarray) -> Any:
        raise NotImplementedError

    def detections(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        return detections_from_result(self.predict(frame), self.names)


class UltralyticsPredictor(BasePredictor):
    def __init__(self, config: Any):
        from ultralytics import YOLO

        self.config = config
        self.model = YOLO(config.model_path)
        self.names = normalize_names(getattr(self.model, "names", {}))

    def predict(self, frame: np.ndarray) -> Any:
        result = self.model.predict(
            frame,
            save=False,
            save_txt=False,
            imgsz=int(getattr(self.config, "imgsz", 640)),
            conf=float(getattr(self.config, "conf", 0.25)),
            iou=float(getattr(self.config, "iou", 0.70)),
            device=self._device(),
            verbose=False,
        )[0]
        result_names = normalize_names(getattr(result, "names", None) or getattr(self.model, "names", {}))
        if result_names:
            self.names = result_names
        return result

    def _device(self) -> str:
        device = str(getattr(self.config, "device", "auto"))
        if device != "auto":
            return device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"


class ONNXRuntimePredictor(BasePredictor):
    def __init__(self, config: Any):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is required for .onnx inference. Install it in yolo26_gui first."
            ) from exc

        self.config = config
        self.model_path = Path(config.model_path)
        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        self.input_meta = self.session.get_inputs()[0]
        self.input_name = self.input_meta.name
        self.nhwc = False
        self.input_height, self.input_width = self._resolve_input_size()
        self.names = self._load_names()

    def predict(self, frame: np.ndarray) -> InferenceResult:
        image, info = letterbox(frame, (self.input_height, self.input_width))
        tensor = image.astype(np.float32) / 255.0
        if self.nhwc:
            tensor = tensor[None, ...]
        else:
            tensor = np.transpose(tensor, (2, 0, 1))[None, ...]

        outputs = self.session.run(None, {self.input_name: tensor})
        return postprocess_yolo_outputs(
            outputs,
            frame.shape,
            info,
            self.names,
            float(getattr(self.config, "conf", 0.25)),
            float(getattr(self.config, "iou", 0.70)),
        )

    def _resolve_input_size(self) -> Tuple[int, int]:
        default_size = int(getattr(self.config, "imgsz", 640) or 640)
        shape = list(getattr(self.input_meta, "shape", []) or [])
        if len(shape) != 4:
            return default_size, default_size

        if _dim_value(shape[1]) == 3:
            height = _dim_value(shape[2]) or default_size
            width = _dim_value(shape[3]) or default_size
            return int(height), int(width)
        if _dim_value(shape[3]) == 3:
            self.nhwc = True
            height = _dim_value(shape[1]) or default_size
            width = _dim_value(shape[2]) or default_size
            return int(height), int(width)
        return default_size, default_size

    def _load_names(self) -> Dict[int, str]:
        metadata = self.session.get_modelmeta().custom_metadata_map or {}
        for key in ("names", "classes", "class_names"):
            names = parse_names(metadata.get(key, ""))
            if names:
                return names

        sidecar = self.model_path.with_suffix(".json")
        if sidecar.exists():
            try:
                data = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
            for key in ("names", "classes", "class_names"):
                names = normalize_names(data.get(key))
                if names:
                    return names
        return {}


def create_predictor(config: Any) -> BasePredictor:
    backend = model_backend(config.model_path)
    if backend == "pt":
        return UltralyticsPredictor(config)
    if backend == "onnx":
        return ONNXRuntimePredictor(config)
    raise ValueError(f"Unsupported model format: {config.model_path}")


def model_backend(model_path: str | Path) -> str:
    suffix = Path(model_path).suffix.lower()
    if suffix == ".pt":
        return "pt"
    if suffix == ".onnx":
        return "onnx"
    raise ValueError(f"Unsupported model suffix: {suffix or '<none>'}")


def letterbox(frame: np.ndarray, new_shape: Tuple[int, int], color: Tuple[int, int, int] = (114, 114, 114)):
    height, width = frame.shape[:2]
    target_height, target_width = int(new_shape[0]), int(new_shape[1])
    ratio = min(target_width / max(width, 1), target_height / max(height, 1))
    resized_width = int(round(width * ratio))
    resized_height = int(round(height * ratio))
    pad_x = (target_width - resized_width) / 2.0
    pad_y = (target_height - resized_height) / 2.0

    if (width, height) != (resized_width, resized_height):
        resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    else:
        resized = frame

    top = int(round(pad_y - 0.1))
    bottom = int(round(pad_y + 0.1))
    left = int(round(pad_x - 0.1))
    right = int(round(pad_x + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    return rgb, LetterboxInfo(ratio, pad_x, pad_y, target_width, target_height)


def postprocess_yolo_outputs(
    outputs: Sequence[np.ndarray],
    image_shape: Tuple[int, ...],
    letterbox_info: LetterboxInfo,
    names: Dict[int, str] | None,
    conf_threshold: float,
    iou_threshold: float,
) -> InferenceResult:
    names = names or {}
    predictions = _select_detection_output(outputs)
    if predictions.size == 0:
        return _empty_result(image_shape, names)

    predictions = np.asarray(predictions, dtype=np.float32)
    if predictions.ndim == 3:
        predictions = predictions[0]
    if predictions.ndim == 1:
        predictions = predictions.reshape(1, -1)
    if predictions.ndim != 2:
        return _empty_result(image_shape, names)
    if predictions.shape[1] < 6 and predictions.shape[0] >= 6:
        predictions = predictions.T
    elif predictions.shape[0] < predictions.shape[1] and predictions.shape[0] <= 256:
        predictions = predictions.T
    if predictions.shape[1] < 6:
        return _empty_result(image_shape, names)

    attrs = predictions.shape[1]
    class_count = len(names)
    boxes: np.ndarray
    scores: np.ndarray
    classes: np.ndarray

    if attrs == 6 and _looks_like_nms_output(predictions):
        boxes = predictions[:, :4].copy()
        scores = predictions[:, 4].copy()
        classes = predictions[:, 5].astype(np.int32)
        boxes = _maybe_denormalize_xyxy(boxes, letterbox_info)
    else:
        if class_count and attrs >= 5 + class_count:
            class_scores = predictions[:, 5 : 5 + class_count] * predictions[:, 4:5]
        elif class_count and attrs >= 4 + class_count:
            class_scores = predictions[:, 4 : 4 + class_count]
        elif attrs > 6:
            class_scores = predictions[:, 4:]
        else:
            return _empty_result(image_shape, names)

        classes = np.argmax(class_scores, axis=1).astype(np.int32)
        scores = np.max(class_scores, axis=1)
        boxes = xywh_to_xyxy(predictions[:, :4])
        boxes = _maybe_denormalize_xyxy(boxes, letterbox_info)

    keep = scores >= float(conf_threshold)
    boxes = boxes[keep]
    scores = scores[keep]
    classes = classes[keep]
    if len(boxes) == 0:
        return _empty_result(image_shape, names)

    boxes = scale_boxes_from_letterbox(boxes, image_shape, letterbox_info)
    keep_indices = classwise_nms(boxes, scores, classes, float(iou_threshold))
    if keep_indices:
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        classes = classes[keep_indices]
    else:
        boxes = np.zeros((0, 4), dtype=np.float32)
        scores = np.zeros((0,), dtype=np.float32)
        classes = np.zeros((0,), dtype=np.float32)

    boxes_obj = InferenceBoxes(boxes, classes.astype(np.float32), scores.astype(np.float32), image_shape)
    return InferenceResult(boxes=boxes_obj, names=names)


def detections_from_result(result: Any, fallback_names: Dict[int, str] | None = None) -> List[Dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or getattr(boxes, "xyxy", None) is None:
        return []

    names = normalize_names(getattr(result, "names", None)) or (fallback_names or {})
    xyxy_values = as_plain_list(getattr(boxes, "xyxy", []))
    class_values = as_plain_list(getattr(boxes, "cls", []))
    conf_values = as_plain_list(getattr(boxes, "conf", []))
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


def result_class_names(names: Any) -> List[str]:
    normalized = normalize_names(names)
    if not normalized:
        return []
    return [str(normalized.get(index, index)) for index in range(max(normalized) + 1)]


def normalize_names(names: Any) -> Dict[int, str]:
    if not names:
        return {}
    if isinstance(names, dict):
        normalized: Dict[int, str] = {}
        for key, value in names.items():
            try:
                normalized[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
        return normalized
    if isinstance(names, (list, tuple)):
        return {index: str(value) for index, value in enumerate(names)}
    return parse_names(str(names))


def parse_names(raw: str) -> Dict[int, str]:
    if not raw:
        return {}
    for parser in (json.loads, ast.literal_eval):
        try:
            return normalize_names(parser(raw))
        except Exception:
            continue
    return {}


def model_class_name(names: Dict[int, str] | Sequence[str] | None, class_id: int) -> str:
    normalized = normalize_names(names)
    return str(normalized.get(class_id, class_id))


def as_plain_list(values: Any) -> List[Any]:
    if values is None:
        return []
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)


def xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    xywh = np.asarray(xywh, dtype=np.float32)
    xyxy = np.zeros_like(xywh[:, :4], dtype=np.float32)
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2.0
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2.0
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2.0
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2.0
    return xyxy


def scale_boxes_from_letterbox(
    boxes: np.ndarray,
    image_shape: Tuple[int, ...],
    letterbox_info: LetterboxInfo,
) -> np.ndarray:
    boxes = boxes.astype(np.float32).copy()
    boxes[:, [0, 2]] -= float(letterbox_info.pad_x)
    boxes[:, [1, 3]] -= float(letterbox_info.pad_y)
    boxes[:, :4] /= max(float(letterbox_info.ratio), 1e-9)

    height, width = image_shape[:2]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0.0, float(width))
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0.0, float(height))
    return boxes


def classwise_nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_threshold: float) -> List[int]:
    keep: List[int] = []
    for class_id in sorted(set(classes.astype(int).tolist())):
        indices = np.where(classes.astype(int) == class_id)[0]
        indices = indices[np.argsort(scores[indices])[::-1]]
        while len(indices) > 0:
            current = int(indices[0])
            keep.append(current)
            if len(indices) == 1:
                break
            ious = box_iou_numpy(boxes[current], boxes[indices[1:]])
            indices = indices[1:][ious <= iou_threshold]
    return sorted(keep, key=lambda index: float(scores[index]), reverse=True)


def box_iou_numpy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area1 + area2 - inter, 1e-9)


def _select_detection_output(outputs: Sequence[np.ndarray]) -> np.ndarray:
    for output in outputs:
        array = np.asarray(output)
        if array.ndim >= 2 and np.issubdtype(array.dtype, np.number):
            return array
    return np.zeros((0, 6), dtype=np.float32)


def _empty_result(image_shape: Tuple[int, ...], names: Dict[int, str]) -> InferenceResult:
    return InferenceResult(InferenceBoxes([], [], [], image_shape), names)


def _looks_like_nms_output(predictions: np.ndarray) -> bool:
    if predictions.shape[1] != 6:
        return False
    class_values = predictions[:, 5]
    return np.all(class_values >= 0) and np.allclose(class_values, np.round(class_values), atol=1e-3)


def _maybe_denormalize_xyxy(boxes: np.ndarray, letterbox_info: LetterboxInfo) -> np.ndarray:
    boxes = boxes.astype(np.float32).copy()
    if boxes.size and float(np.nanmax(np.abs(boxes))) <= 1.5:
        boxes[:, [0, 2]] *= float(letterbox_info.input_width)
        boxes[:, [1, 3]] *= float(letterbox_info.input_height)
    return boxes


def _dim_value(value: Any) -> int | None:
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None
