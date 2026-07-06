from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List


IMG_FORMATS = {
    "bmp",
    "dng",
    "jpeg",
    "jpg",
    "mpo",
    "png",
    "tif",
    "tiff",
    "webp",
    "pfm",
}

VID_FORMATS = {
    "asf",
    "avi",
    "gif",
    "m4v",
    "mkv",
    "mov",
    "mp4",
    "mpeg",
    "mpg",
    "ts",
    "wmv",
    "webm",
    "flv",
}


def detect_file_kind(path: str | Path) -> str:
    suffix = Path(path).suffix.lower().lstrip(".")
    if suffix in IMG_FORMATS:
        return "image"
    if suffix in VID_FORMATS:
        return "video"
    raise ValueError(f"Unsupported file suffix: {suffix or '<none>'}")


def is_supported_media(path: str | Path) -> bool:
    suffix = Path(path).suffix.lower().lstrip(".")
    return suffix in IMG_FORMATS or suffix in VID_FORMATS


def iter_supported_media(folder: str | Path) -> List[Path]:
    folder_path = Path(folder)
    if not folder_path.exists() or not folder_path.is_dir():
        raise ValueError(f"Folder does not exist: {folder_path}")

    files = [
        path
        for path in folder_path.rglob("*")
        if path.is_file() and is_supported_media(path)
    ]
    return sorted(files, key=lambda path: str(path.relative_to(folder_path)).lower())


def merge_counts(total: Dict[str, int], current: Dict[str, int]) -> Dict[str, int]:
    for name, count in current.items():
        total[name] = total.get(name, 0) + int(count)
    return total


@dataclass
class DetectionConfig:
    model_path: str
    conf: float = 0.25
    iou: float = 0.70
    rate_ms: int = 30
    save_results: bool = False
    save_txt: bool = False
    imgsz: int = 640
    device: str = "auto"

    @property
    def model_name(self) -> str:
        return Path(self.model_path).name


@dataclass
class SourceSpec:
    kind: str
    path: str = ""
    camera_index: int = 0
    display_name: str = ""

    @classmethod
    def from_file(cls, path: str | Path) -> "SourceSpec":
        path_obj = Path(path)
        return cls(
            kind=detect_file_kind(path_obj),
            path=str(path_obj),
            display_name=path_obj.name,
        )

    @classmethod
    def camera(cls, index: int = 0) -> "SourceSpec":
        return cls(kind="camera", camera_index=index, display_name=f"Camera {index}")

    @classmethod
    def stream(cls, url: str) -> "SourceSpec":
        return cls(kind="stream", path=url, display_name=url)

    @classmethod
    def batch(cls, folder: str | Path) -> "SourceSpec":
        folder_obj = Path(folder)
        return cls(kind="batch", path=str(folder_obj), display_name=folder_obj.name)


@dataclass
class FrameResult:
    raw_frame: Any
    annotated_frame: Any
    fps: float
    class_count: int
    target_count: int
    class_counts: Dict[str, int] = field(default_factory=dict)
    frame_index: int = 0
    progress: int = 0
    detections: List[Dict[str, Any]] = field(default_factory=list)
    class_names: List[str] = field(default_factory=list)
    source_path: str = ""
    source_name: str = ""


@dataclass
class RunSummary:
    run_id: str
    started_at: str
    ended_at: str
    source_type: str
    source_name: str
    model_name: str
    conf: float
    iou: float
    rate_ms: int
    save_results: bool
    save_txt: bool
    frames: int
    duration_seconds: float
    avg_fps: float
    max_targets: int
    final_class_count: int
    final_target_count: int
    total_target_events: int
    class_counts: Dict[str, int] = field(default_factory=dict)
    status: str = "completed"
    output_dir: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["duration_seconds"] = round(float(self.duration_seconds), 3)
        data["avg_fps"] = round(float(self.avg_fps), 2)
        return data
