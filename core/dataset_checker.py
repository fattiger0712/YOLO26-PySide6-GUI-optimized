from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from core.models import IMG_FORMATS


SMALL_TARGET_PX = 32
SPLIT_KEYS = ("train", "val", "test")
ISSUE_FIELDS = ("issue_type", "path", "detail")


@dataclass
class YoloObject:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    raw_line: str = ""

    def to_xyxy(self, image_width: int, image_height: int) -> List[float]:
        box_width = self.width * image_width
        box_height = self.height * image_height
        x_center = self.x_center * image_width
        y_center = self.y_center * image_height
        return [
            x_center - box_width / 2.0,
            y_center - box_height / 2.0,
            x_center + box_width / 2.0,
            y_center + box_height / 2.0,
        ]

    def pixel_size(self, image_width: int, image_height: int) -> Tuple[float, float]:
        return self.width * image_width, self.height * image_height


@dataclass
class DatasetSource:
    input_path: str
    root: Path
    yaml_path: Optional[Path] = None
    names: List[str] = field(default_factory=list)
    image_paths: List[Path] = field(default_factory=list)
    image_roots: List[Path] = field(default_factory=list)

    @property
    def nc(self) -> int:
        return len(self.names)


def check_dataset(
    dataset_path: str | Path,
    output_root: str | Path,
    small_target_px: int = SMALL_TARGET_PX,
) -> Dict[str, Any]:
    source = load_yolo_dataset_source(dataset_path)
    output_dir = make_report_dir(output_root, source.root)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = build_dataset_report(source, output_dir, small_target_px=small_target_px)
    write_dataset_report(report, output_dir)
    return report


def load_yolo_dataset_source(dataset_path: str | Path) -> DatasetSource:
    input_path = Path(dataset_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {input_path}")

    yaml_path: Optional[Path] = input_path if input_path.is_file() and input_path.suffix.lower() in {".yaml", ".yml"} else None
    if input_path.is_dir() and (input_path / "data.yaml").exists():
        yaml_path = input_path / "data.yaml"

    data: Dict[str, Any] = {}
    if yaml_path:
        data = load_yaml_dict(yaml_path)

    if yaml_path and data.get("path"):
        root = resolve_path(data.get("path"), yaml_path.parent)
    else:
        root = input_path if input_path.is_dir() else input_path.parent

    names = normalize_names(data.get("names"), data.get("nc"))
    image_paths: List[Path] = []
    image_roots: List[Path] = []

    if data:
        for key in SPLIT_KEYS:
            value = data.get(key)
            if value in (None, ""):
                continue
            split_paths, split_roots = resolve_image_entries(value, root, yaml_path.parent if yaml_path else root)
            image_paths.extend(split_paths)
            image_roots.extend(split_roots)

    if not image_paths:
        default_roots = []
        if (root / "images").exists():
            default_roots.append(root / "images")
        else:
            default_roots.append(root)
        for image_root in default_roots:
            image_paths.extend(scan_images(image_root))
            image_roots.append(image_root)

    unique_images = sorted({path.resolve() for path in image_paths}, key=lambda path: str(path).lower())
    unique_roots = sorted({path.resolve() for path in image_roots if path.exists()}, key=lambda path: str(path).lower())
    return DatasetSource(
        input_path=str(input_path),
        root=root.resolve(),
        yaml_path=yaml_path.resolve() if yaml_path else None,
        names=names,
        image_paths=unique_images,
        image_roots=unique_roots,
    )


def build_dataset_report(
    source: DatasetSource,
    output_dir: str | Path,
    small_target_px: int = SMALL_TARGET_PX,
) -> Dict[str, Any]:
    output_path = Path(output_dir)
    class_box_counts: Dict[int, int] = {}
    class_image_counts: Dict[int, int] = {}
    small_target_counts: Dict[int, int] = {}
    issues: List[Dict[str, str]] = []
    expected_labels: set[Path] = set()

    total_boxes = 0
    small_targets = 0
    bad_images = 0
    missing_labels = 0
    invalid_label_lines = 0
    out_of_bounds_labels = 0

    for image_path in source.image_paths:
        label_path = label_path_for_image(image_path, source.root, source.image_roots)
        expected_labels.add(label_path.resolve())

        frame = cv2.imread(str(image_path))
        if frame is None:
            bad_images += 1
            issues.append(issue("bad_image", image_path, "OpenCV could not read this image."))
            continue

        height, width = frame.shape[:2]
        if not label_path.exists():
            missing_labels += 1
            issues.append(issue("missing_label", label_path, f"Missing label for image {image_path.name}."))
            continue

        objects, line_issues = read_yolo_label(label_path)
        for detail in line_issues:
            invalid_label_lines += 1
            issues.append(issue("invalid_label_line", label_path, detail))

        image_class_ids: set[int] = set()
        for obj in objects:
            if source.nc and (obj.class_id < 0 or obj.class_id >= source.nc):
                out_of_bounds_labels += 1
                issues.append(
                    issue(
                        "class_id_out_of_bounds",
                        label_path,
                        f"class_id={obj.class_id}, nc={source.nc}, line={obj.raw_line}",
                    )
                )
                continue

            total_boxes += 1
            class_box_counts[obj.class_id] = class_box_counts.get(obj.class_id, 0) + 1
            image_class_ids.add(obj.class_id)
            box_width, box_height = obj.pixel_size(width, height)
            if box_width <= small_target_px and box_height <= small_target_px:
                small_targets += 1
                small_target_counts[obj.class_id] = small_target_counts.get(obj.class_id, 0) + 1

        for class_id in image_class_ids:
            class_image_counts[class_id] = class_image_counts.get(class_id, 0) + 1

    orphan_labels = find_orphan_labels(source.root, expected_labels)
    for label_path in orphan_labels:
        issues.append(issue("orphan_label", label_path, "Label file has no matching image."))

    class_stats = build_class_stats(source.names, class_box_counts, class_image_counts, small_target_counts)
    avg_boxes = total_boxes / len(source.image_paths) if source.image_paths else 0.0
    small_ratio = small_targets / total_boxes if total_boxes else 0.0

    artifacts = {
        "class_image_hist": str(output_path / "class_image_hist.png"),
        "class_box_hist": str(output_path / "class_box_hist.png"),
        "small_target_hist": str(output_path / "small_target_hist.png"),
    }
    render_bar_chart(
        [(row["class_name"], int(row["image_count"])) for row in class_stats],
        Path(artifacts["class_image_hist"]),
        "Images per class",
    )
    render_bar_chart(
        [(row["class_name"], int(row["box_count"])) for row in class_stats],
        Path(artifacts["class_box_hist"]),
        "Boxes per class",
    )
    render_bar_chart(
        [(row["class_name"], int(row["small_target_count"])) for row in class_stats],
        Path(artifacts["small_target_hist"]),
        f"Small targets <= {small_target_px}px",
    )

    summary = {
        "dataset_root": str(source.root),
        "yaml_path": str(source.yaml_path) if source.yaml_path else "",
        "total_images": len(source.image_paths),
        "total_boxes": total_boxes,
        "avg_boxes_per_image": round(avg_boxes, 4),
        "small_targets": small_targets,
        "small_target_ratio": round(small_ratio, 4),
        "missing_labels": missing_labels,
        "orphan_labels": len(orphan_labels),
        "bad_images": bad_images,
        "class_id_out_of_bounds": out_of_bounds_labels,
        "invalid_label_lines": invalid_label_lines,
        "class_count": source.nc or len(class_stats),
        "small_target_px": small_target_px,
    }
    report = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary": summary,
        "class_stats": class_stats,
        "issues": issues,
        "artifacts": artifacts,
        "output_dir": str(output_path),
    }
    return report


def write_dataset_report(report: Dict[str, Any], output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "check_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (output_path / "check_report.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ISSUE_FIELDS)
        writer.writeheader()
        for item in report.get("issues", []):
            writer.writerow({field: item.get(field, "") for field in ISSUE_FIELDS})

    summary = report.get("summary", {})
    lines = [
        "YOLO dataset check report",
        f"Created: {report.get('created_at', '')}",
        f"Dataset: {summary.get('dataset_root', '')}",
        "",
        f"Images: {summary.get('total_images', 0)}",
        f"Boxes: {summary.get('total_boxes', 0)}",
        f"Avg boxes/image: {summary.get('avg_boxes_per_image', 0)}",
        f"Small targets: {summary.get('small_targets', 0)} ({summary.get('small_target_ratio', 0):.2%})",
        f"Missing labels: {summary.get('missing_labels', 0)}",
        f"Orphan labels: {summary.get('orphan_labels', 0)}",
        f"Bad images: {summary.get('bad_images', 0)}",
        f"Class id out of bounds: {summary.get('class_id_out_of_bounds', 0)}",
        f"Invalid label lines: {summary.get('invalid_label_lines', 0)}",
        "",
        "Issues:",
    ]
    for item in report.get("issues", []):
        lines.append(f"- [{item.get('issue_type', '')}] {item.get('path', '')}: {item.get('detail', '')}")
    (output_path / "check_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_yolo_label(label_path: str | Path) -> Tuple[List[YoloObject], List[str]]:
    objects: List[YoloObject] = []
    issues: List[str] = []
    try:
        lines = Path(label_path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return objects, [f"Could not read label: {exc}"]

    for line_number, raw_line in enumerate(lines, 1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 5:
            issues.append(f"line {line_number}: expected at least 5 values, got {len(parts)}")
            continue
        try:
            class_id = int(float(parts[0]))
            x_center, y_center, width, height = [float(value) for value in parts[1:5]]
        except ValueError:
            issues.append(f"line {line_number}: non-numeric YOLO values: {stripped}")
            continue
        if not all(math.isfinite(value) for value in (x_center, y_center, width, height)):
            issues.append(f"line {line_number}: non-finite box values: {stripped}")
            continue
        if width <= 0 or height <= 0:
            issues.append(f"line {line_number}: box width/height must be positive: {stripped}")
            continue
        if any(value < 0 or value > 1 for value in (x_center, y_center, width, height)):
            issues.append(f"line {line_number}: normalized box values should be in [0, 1]: {stripped}")
        objects.append(YoloObject(class_id, x_center, y_center, width, height, stripped))
    return objects, issues


def label_path_for_image(image_path: Path, dataset_root: Path, image_roots: Sequence[Path]) -> Path:
    parts = list(image_path.parts)
    lowered = [part.lower() for part in parts]
    if "images" in lowered:
        index = len(lowered) - 1 - list(reversed(lowered)).index("images")
        parts[index] = "labels"
        return Path(*parts).with_suffix(".txt")

    for image_root in image_roots:
        try:
            relative = image_path.relative_to(image_root)
        except ValueError:
            continue
        return (dataset_root / "labels" / relative).with_suffix(".txt")
    return (dataset_root / "labels" / image_path.name).with_suffix(".txt")


def find_orphan_labels(dataset_root: Path, expected_labels: set[Path]) -> List[Path]:
    labels_root = dataset_root / "labels"
    if not labels_root.exists():
        return []

    orphaned: List[Path] = []
    for label_path in sorted(labels_root.rglob("*.txt"), key=lambda path: str(path).lower()):
        if label_path.resolve() in expected_labels:
            continue
        if label_path.name.lower() == "classes.txt":
            continue
        orphaned.append(label_path.resolve())
    return orphaned


def build_class_stats(
    names: Sequence[str],
    class_box_counts: Dict[int, int],
    class_image_counts: Dict[int, int],
    small_target_counts: Dict[int, int],
) -> List[Dict[str, Any]]:
    class_ids = set(class_box_counts) | set(class_image_counts) | set(small_target_counts)
    class_ids.update(range(len(names)))
    stats = []
    for class_id in sorted(class_ids):
        stats.append(
            {
                "class_id": class_id,
                "class_name": class_name(names, class_id),
                "image_count": int(class_image_counts.get(class_id, 0)),
                "box_count": int(class_box_counts.get(class_id, 0)),
                "small_target_count": int(small_target_counts.get(class_id, 0)),
            }
        )
    return stats


def class_name(names: Sequence[str], class_id: int) -> str:
    if 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def make_report_dir(output_root: str | Path, dataset_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(output_root) / "dataset_checks" / f"{stamp}_{safe_name(dataset_root.name or 'dataset')}"


def resolve_image_entries(value: Any, root: Path, yaml_parent: Path) -> Tuple[List[Path], List[Path]]:
    image_paths: List[Path] = []
    image_roots: List[Path] = []
    entries = value if isinstance(value, list) else [value]
    for entry in entries:
        if entry in (None, ""):
            continue
        entry_path = resolve_path(str(entry), root)
        if not entry_path.exists() and not Path(str(entry)).is_absolute():
            entry_path = resolve_path(str(entry), yaml_parent)

        if entry_path.is_file() and entry_path.suffix.lower() == ".txt":
            listed = load_image_list_file(entry_path, yaml_parent)
            image_paths.extend(listed)
            image_roots.extend(infer_image_roots(listed))
        elif entry_path.is_file() and is_image_path(entry_path):
            image_paths.append(entry_path)
            image_roots.append(entry_path.parent)
        elif entry_path.is_dir():
            image_paths.extend(scan_images(entry_path))
            image_roots.append(entry_path)
    return image_paths, image_roots


def scan_images(folder: str | Path) -> List[Path]:
    folder_path = Path(folder)
    if not folder_path.exists() or not folder_path.is_dir():
        return []
    return sorted(
        [path for path in folder_path.rglob("*") if path.is_file() and is_image_path(path)],
        key=lambda path: str(path).lower(),
    )


def is_image_path(path: str | Path) -> bool:
    return Path(path).suffix.lower().lstrip(".") in IMG_FORMATS


def load_image_list_file(path: Path, yaml_parent: Path) -> List[Path]:
    images: List[Path] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return images
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        image_path = Path(stripped)
        if not image_path.is_absolute():
            image_path = (yaml_parent / image_path).resolve()
        if image_path.exists() and is_image_path(image_path):
            images.append(image_path)
    return images


def infer_image_roots(paths: Iterable[Path]) -> List[Path]:
    roots = set()
    for path in paths:
        parent = path.parent
        parts = [part.lower() for part in parent.parts]
        if "images" in parts:
            index = len(parts) - 1 - list(reversed(parts)).index("images")
            roots.add(Path(*parent.parts[: index + 1]))
        else:
            roots.add(parent)
    return sorted(roots, key=lambda path: str(path).lower())


def load_yaml_dict(path: str | Path) -> Dict[str, Any]:
    yaml_path = Path(path)
    try:
        import yaml  # type: ignore

        with yaml_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return load_simple_yaml(yaml_path)


def load_simple_yaml(path: Path) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    current_key = ""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return data

    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith((" ", "\t")) and current_key == "names" and ":" in line:
            key, raw_value = line.strip().split(":", 1)
            names = data.setdefault("names", {})
            if isinstance(names, dict):
                try:
                    names[int(key.strip())] = parse_scalar(raw_value.strip())
                except ValueError:
                    pass
            continue
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        current_key = key.strip()
        value = raw_value.strip()
        if current_key == "names" and not value:
            data[current_key] = {}
        else:
            data[current_key] = parse_scalar(value)
    return data


def parse_scalar(value: str) -> Any:
    if value in {"", "null", "None", "~"}:
        return ""
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [parse_scalar(part.strip()) for part in inner.split(",")]
    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
        return value[1:-1]
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def normalize_names(raw_names: Any, nc: Any = None) -> List[str]:
    if isinstance(raw_names, dict):
        int_keys = []
        for key in raw_names:
            try:
                int_keys.append(int(key))
            except (TypeError, ValueError):
                continue
        if not int_keys:
            return [str(value) for value in raw_names.values()]
        names = [str(index) for index in range(max(int_keys) + 1)]
        for key, value in raw_names.items():
            try:
                names[int(key)] = str(value)
            except (TypeError, ValueError, IndexError):
                continue
        return names
    if isinstance(raw_names, list):
        return [str(name) for name in raw_names]
    if isinstance(raw_names, tuple):
        return [str(name) for name in raw_names]
    try:
        count = int(nc)
    except (TypeError, ValueError):
        count = 0
    return [str(index) for index in range(count)]


def resolve_path(value: Any, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return (base / path).resolve()


def issue(issue_type: str, path: str | Path, detail: str) -> Dict[str, str]:
    return {"issue_type": issue_type, "path": str(path), "detail": detail}


def render_bar_chart(data: Sequence[Tuple[str, int]], output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    top = sorted(data, key=lambda item: item[1], reverse=True)[:30]
    if not top:
        top = [("none", 0)]

    width = 1100
    height = max(360, 80 + len(top) * 28)
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    margin_left = 240
    margin_right = 80
    margin_top = 56
    row_height = 26
    bar_area = width - margin_left - margin_right
    max_value = max(max(value for _label, value in top), 1)

    cv2.putText(image, title, (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (24, 34, 53), 2, cv2.LINE_AA)
    for index, (label, value) in enumerate(top):
        y = margin_top + index * row_height
        text = truncate_text(str(label), 28)
        cv2.putText(image, text, (24, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (62, 73, 91), 1, cv2.LINE_AA)
        bar_width = int((value / max_value) * bar_area) if max_value else 0
        color = (239, 94, 21) if index == 0 else (239, 137, 44)
        cv2.rectangle(image, (margin_left, y + 4), (margin_left + max(bar_width, 1), y + 21), color, -1)
        cv2.putText(
            image,
            str(value),
            (margin_left + min(bar_width + 8, bar_area - 32), y + 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (24, 34, 53),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), image)


def truncate_text(value: str, max_len: int) -> str:
    return value if len(value) <= max_len else value[: max_len - 3] + "..."


def safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)
    return cleaned.strip("._") or "dataset"
