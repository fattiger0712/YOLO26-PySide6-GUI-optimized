from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import cv2


DEFAULT_LABELIMG_EXE = Path(r"E:\anaconda\envs\yolo26_gui\Scripts\labelImg.EXE")
MANIFEST_FIELDS = [
    "created_at",
    "sample_id",
    "image_path",
    "label_path",
    "source_kind",
    "source_name",
    "source_path",
    "model_name",
    "frame_index",
    "issue_reason",
    "note",
    "detections",
]


@dataclass
class ReviewSample:
    root: Path
    images_dir: Path
    labels_dir: Path
    image_path: Path
    label_path: Path
    root_class_file: Path
    label_class_file: Path
    data_yaml_path: Path
    manifest_path: Path
    sample_id: str


def export_review_sample(
    root: str | Path,
    frame,
    detections: Iterable[Dict[str, Any]],
    class_names: Iterable[str],
    metadata: Dict[str, Any],
) -> ReviewSample:
    root_path = Path(root)
    images_dir = root_path / "images"
    labels_dir = root_path / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    detections_list = list(detections or [])
    names = normalize_class_names(class_names, detections_list)
    sample_id = metadata.get("sample_id") or make_sample_id(metadata)

    image_path = images_dir / f"{sample_id}.jpg"
    label_path = labels_dir / f"{sample_id}.txt"
    root_class_file = root_path / "classes.txt"
    label_class_file = labels_dir / "classes.txt"
    data_yaml_path = root_path / "data.yaml"
    manifest_path = root_path / "manifest.csv"

    if not cv2.imwrite(str(image_path), frame):
        raise OSError(f"Failed to write review image: {image_path}")

    height, width = frame.shape[:2]
    write_class_file(root_class_file, names)
    write_class_file(label_class_file, names)
    write_yolo_label_file(label_path, detections_list, width, height)
    write_data_yaml(data_yaml_path, root_path, names)
    append_manifest(manifest_path, sample_id, image_path, label_path, detections_list, metadata)

    return ReviewSample(
        root=root_path,
        images_dir=images_dir,
        labels_dir=labels_dir,
        image_path=image_path,
        label_path=label_path,
        root_class_file=root_class_file,
        label_class_file=label_class_file,
        data_yaml_path=data_yaml_path,
        manifest_path=manifest_path,
        sample_id=sample_id,
    )


def normalize_class_names(class_names: Iterable[str], detections: Iterable[Dict[str, Any]]) -> List[str]:
    names = [str(name) for name in (class_names or [])]
    max_class_id = -1
    for detection in detections or []:
        try:
            max_class_id = max(max_class_id, int(detection.get("class_id", -1)))
        except (TypeError, ValueError):
            continue

    while len(names) <= max_class_id:
        names.append(str(len(names)))
    if not names:
        names.append("object")

    for detection in detections or []:
        try:
            class_id = int(detection.get("class_id", -1))
        except (TypeError, ValueError):
            continue
        if class_id < 0 or class_id >= len(names):
            continue
        class_name = str(detection.get("class_name") or "").strip()
        if class_name and names[class_id] in {"", str(class_id)}:
            names[class_id] = class_name

    return unique_class_names(names)


def unique_class_names(class_names: Iterable[str]) -> List[str]:
    seen: Dict[str, int] = {}
    unique_names: List[str] = []
    for index, raw_name in enumerate(class_names):
        name = str(raw_name).strip() or str(index)
        if name in seen:
            name = f"{name}__{index}"
        seen[name] = index
        unique_names.append(name)
    return unique_names


def write_class_file(path: Path, class_names: Iterable[str]) -> None:
    path.write_text("\n".join(class_names) + "\n", encoding="utf-8")


def write_yolo_label_file(
    path: Path,
    detections: Iterable[Dict[str, Any]],
    image_width: int,
    image_height: int,
) -> None:
    lines = []
    for detection in detections:
        line = detection_to_yolo_line(detection, image_width, image_height)
        if line is not None:
            lines.append(line)
    path.write_text("".join(lines), encoding="utf-8")


def detection_to_yolo_line(
    detection: Dict[str, Any],
    image_width: int,
    image_height: int,
) -> Optional[str]:
    if image_width <= 0 or image_height <= 0:
        return None

    xyxy = detection.get("xyxy") or []
    if len(xyxy) < 4:
        return None

    try:
        class_id = int(detection.get("class_id", 0))
        x1, y1, x2, y2 = [float(value) for value in xyxy[:4]]
    except (TypeError, ValueError):
        return None

    x1 = max(0.0, min(float(image_width), x1))
    x2 = max(0.0, min(float(image_width), x2))
    y1 = max(0.0, min(float(image_height), y1))
    y2 = max(0.0, min(float(image_height), y2))
    if x2 <= x1 or y2 <= y1:
        return None

    x_center = ((x1 + x2) / 2.0) / image_width
    y_center = ((y1 + y2) / 2.0) / image_height
    box_width = (x2 - x1) / image_width
    box_height = (y2 - y1) / image_height
    return f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n"


def write_data_yaml(path: Path, root: Path, class_names: Iterable[str]) -> None:
    names = list(class_names)
    lines = [
        f"path: {json.dumps(str(root), ensure_ascii=False)}",
        "train: images",
        "val: images",
        f"nc: {len(names)}",
        "names:",
    ]
    for index, name in enumerate(names):
        lines.append(f"  {index}: {json.dumps(name, ensure_ascii=False)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_manifest(
    path: Path,
    sample_id: str,
    image_path: Path,
    label_path: Path,
    detections: Iterable[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> None:
    row = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sample_id": sample_id,
        "image_path": str(image_path),
        "label_path": str(label_path),
        "source_kind": metadata.get("source_kind", ""),
        "source_name": metadata.get("source_name", ""),
        "source_path": metadata.get("source_path", ""),
        "model_name": metadata.get("model_name", ""),
        "frame_index": metadata.get("frame_index", ""),
        "issue_reason": metadata.get("issue_reason", ""),
        "note": metadata.get("note", ""),
        "detections": json.dumps(list(detections or []), ensure_ascii=False),
    }
    should_write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        if should_write_header:
            writer.writeheader()
        writer.writerow(row)


def make_sample_id(metadata: Dict[str, Any]) -> str:
    created = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    source = metadata.get("source_name") or metadata.get("source_path") or "frame"
    return f"{created}_{safe_name(Path(str(source)).stem)}"


def safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)
    return cleaned.strip("._") or "frame"


def resolve_labelimg_executable(preferred: str | Path | None = None) -> Optional[Path]:
    candidates = []
    if preferred:
        candidates.append(Path(preferred))
    candidates.append(DEFAULT_LABELIMG_EXE)
    for candidate in candidates:
        if candidate.exists():
            return candidate

    for command in ("labelImg", "labelimg"):
        found = shutil.which(command)
        if found:
            return Path(found)
    return None


def build_labelimg_args(image_path: str | Path, class_file: str | Path, labels_dir: str | Path) -> List[str]:
    return [str(image_path), str(class_file), str(labels_dir)]
