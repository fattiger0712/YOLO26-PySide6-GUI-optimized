from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


PRECISION_KEY = "metrics/precision(B)"
RECALL_KEY = "metrics/recall(B)"
MAP50_KEY = "metrics/mAP50(B)"
MAP5095_KEY = "metrics/mAP50-95(B)"
MANUAL_FIELDS = ("display_name", "recommended_for", "tags", "notes")
MODEL_SUFFIXES = {".pt", ".onnx"}


@dataclass
class WeightMetricSummary:
    best_epoch: int = 0
    best_precision: float = 0.0
    best_recall: float = 0.0
    best_map50: float = 0.0
    best_map5095: float = 0.0
    best_time_seconds: float = 0.0
    final_epoch: int = 0
    final_precision: float = 0.0
    final_recall: float = 0.0
    final_map50: float = 0.0
    final_map5095: float = 0.0
    total_epochs: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingArtifactPaths:
    results_csv: str = ""
    args_yaml: str = ""
    best_weight: str = ""
    results_png: str = ""
    pr_curve: str = ""
    f1_curve: str = ""
    precision_curve: str = ""
    recall_curve: str = ""
    confusion_matrix: str = ""
    confusion_matrix_normalized: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WeightTrainingRecord:
    model_name: str
    model_path: str
    display_name: str
    training_name: str = ""
    training_dir: str = ""
    dataset: str = ""
    base_model: str = ""
    task: str = ""
    epochs: int = 0
    imgsz: str = ""
    batch: str = ""
    device: str = ""
    imported_at: str = ""
    updated_at: str = ""
    file_size: int = 0
    modified_at: str = ""
    model_format: str = "pt"
    source_model_path: str = ""
    export_imgsz: int = 0
    export_opset: int = 0
    exported_at: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    recommended_for: str = ""
    tags: str = ""
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WeightRegistryStore:
    def __init__(self, registry_path: str | Path):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

    def load(self, include_removed: bool = False) -> List[Dict[str, Any]]:
        records = self._read_all()
        if include_removed:
            return records
        return [record for record in records if not record.get("removed_from_manager")]

    def _read_all(self) -> List[Dict[str, Any]]:
        if not self.registry_path.exists():
            return []
        try:
            with self.registry_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, list):
                return []
            return [item for item in data if isinstance(item, dict)]
        except (OSError, json.JSONDecodeError):
            return []

    def upsert(self, record: WeightTrainingRecord | Dict[str, Any]) -> Dict[str, Any]:
        data = record.to_dict() if isinstance(record, WeightTrainingRecord) else dict(record)
        model_name = str(data.get("model_name", "")).strip()
        if not model_name:
            raise ValueError("model_name is required")

        data.setdefault("display_name", model_name)
        data.setdefault("metrics", {})
        data.setdefault("artifacts", {})
        data.pop("removed_from_manager", None)
        data.pop("removed_at", None)
        data["updated_at"] = _now_text()

        records = [item for item in self._read_all() if item.get("model_name") != model_name]
        records.append(data)
        records.sort(key=lambda item: str(item.get("display_name") or item.get("model_name", "")).lower())
        self._write(records)
        return data

    def get_by_model_name(self, model_name: str) -> Dict[str, Any] | None:
        for record in self.load():
            if record.get("model_name") == model_name:
                return record
        return None

    def hidden_model_names(self) -> set[str]:
        return {
            str(record.get("model_name"))
            for record in self._read_all()
            if record.get("removed_from_manager") and record.get("model_name")
        }

    def remove_from_manager(self, model_name: str) -> bool:
        model_name = str(model_name).strip()
        if not model_name:
            raise ValueError("model_name is required")

        records = self._read_all()
        existing = next((item for item in records if item.get("model_name") == model_name), {})
        tombstone = {
            "model_name": model_name,
            "model_path": existing.get("model_path", ""),
            "display_name": existing.get("display_name") or Path(model_name).stem,
            "removed_from_manager": True,
            "removed_at": _now_text(),
        }
        remaining = [item for item in records if item.get("model_name") != model_name]
        remaining.append(tombstone)
        remaining.sort(key=lambda item: str(item.get("display_name") or item.get("model_name", "")).lower())
        self._write(remaining)
        return bool(existing)

    def import_training_run(self, run_dir: str | Path, models_dir: str | Path) -> Dict[str, Any]:
        run_path = Path(run_dir)
        if not run_path.exists() or not run_path.is_dir():
            raise ValueError(f"Training directory does not exist: {run_path}")

        source_weight = run_path / "weights" / "best.pt"
        if not source_weight.exists():
            raise ValueError(f"Missing trained weight: {source_weight}")

        models_path = Path(models_dir)
        models_path.mkdir(parents=True, exist_ok=True)
        model_path = models_path / f"{_safe_name(run_path.name)}_best.pt"
        if not model_path.exists():
            shutil.copy2(source_weight, model_path)

        return self.register_training_run(run_path, model_path)

    def register_training_run(self, run_dir: str | Path, model_path: str | Path) -> Dict[str, Any]:
        record = build_training_record(run_dir, model_path)
        existing = self.get_by_model_name(record.model_name)
        if existing:
            _preserve_manual_fields(record, existing)
            record.imported_at = str(existing.get("imported_at") or record.imported_at)
        return self.upsert(record)

    def register_exported_onnx(
        self,
        onnx_path: str | Path,
        source_model_path: str | Path = "",
        imgsz: int = 0,
        opset: int = 0,
    ) -> Dict[str, Any]:
        model = Path(onnx_path)
        if model.suffix.lower() != ".onnx":
            raise ValueError(f"Expected an ONNX file: {model}")
        if not model.exists():
            raise ValueError(f"ONNX file does not exist: {model}")

        source = Path(source_model_path) if source_model_path else None
        source_record = self.get_by_model_name(source.name) if source and source.name else None
        existing = self.get_by_model_name(model.name)
        stat = model.stat()
        now = _now_text()

        data: Dict[str, Any] = {}
        if source_record:
            for key in (
                "training_name",
                "training_dir",
                "dataset",
                "base_model",
                "task",
                "epochs",
                "imgsz",
                "batch",
                "device",
                "metrics",
                "artifacts",
                "recommended_for",
                "tags",
                "notes",
            ):
                if key in source_record:
                    data[key] = source_record[key]

        data.update(
            {
                "model_name": model.name,
                "model_path": _path_text(model),
                "display_name": (existing or {}).get("display_name") or model.stem,
                "imgsz": str(imgsz or data.get("imgsz", "")),
                "imported_at": str((existing or {}).get("imported_at") or now),
                "file_size": int(stat.st_size),
                "modified_at": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "model_format": "onnx",
                "source_model_path": _path_text(source) if source else "",
                "source_model_name": source.name if source else "",
                "export_imgsz": int(imgsz or 0),
                "export_opset": int(opset or 0),
                "exported_at": str((existing or {}).get("exported_at") or now),
            }
        )
        data.setdefault("metrics", {})
        data.setdefault("artifacts", {})
        if existing:
            for field_name in MANUAL_FIELDS:
                if existing.get(field_name):
                    data[field_name] = str(existing[field_name])
        return self.upsert(data)

    def refresh(self, model_name: str) -> Dict[str, Any]:
        existing = self.get_by_model_name(model_name)
        if not existing:
            raise ValueError(f"Unknown model record: {model_name}")
        training_dir = existing.get("training_dir")
        model_path = existing.get("model_path")
        if not training_dir or not Path(training_dir).exists():
            raise ValueError("Training directory is not available for refresh")
        if not model_path:
            model_path = str(Path(training_dir) / "weights" / "best.pt")

        record = build_training_record(training_dir, model_path)
        _preserve_manual_fields(record, existing)
        record.imported_at = str(existing.get("imported_at") or record.imported_at)
        return self.upsert(record)

    def _write(self, records: List[Dict[str, Any]]) -> None:
        with self.registry_path.open("w", encoding="utf-8") as handle:
            json.dump(records, handle, ensure_ascii=False, indent=2)


def build_training_record(run_dir: str | Path, model_path: str | Path) -> WeightTrainingRecord:
    run_path = Path(run_dir)
    model = Path(model_path)
    args = _load_args(run_path / "args.yaml")
    metrics = _load_metrics(run_path / "results.csv")
    artifacts = _collect_artifacts(run_path)
    artifacts.best_weight = _path_text(run_path / "weights" / "best.pt")

    stat = model.stat() if model.exists() else None
    now = _now_text()
    display_name = model.stem
    return WeightTrainingRecord(
        model_name=model.name,
        model_path=_path_text(model),
        display_name=display_name,
        training_name=run_path.name,
        training_dir=_path_text(run_path),
        dataset=str(args.get("data", "")),
        base_model=str(args.get("model", "")),
        task=str(args.get("task", "")),
        epochs=_to_int(args.get("epochs"), 0),
        imgsz=str(args.get("imgsz", "")),
        batch=str(args.get("batch", "")),
        device=str(args.get("device", "")),
        imported_at=now,
        updated_at=now,
        file_size=int(stat.st_size) if stat else 0,
        modified_at=datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S") if stat else "",
        model_format="pt",
        metrics=metrics.to_dict(),
        artifacts=artifacts.to_dict(),
    )


def _load_args(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return _load_simple_yaml(path)


def _load_simple_yaml(path: Path) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return data

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        value = raw_value.strip()
        if not key:
            continue
        data[key] = _parse_scalar(value)
    return data


def _parse_scalar(value: str) -> Any:
    if value in {"", "null", "None", "~"}:
        return ""
    if (value.startswith("'") and value.endswith("'")) or (
        value.startswith('"') and value.endswith('"')
    ):
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


def _load_metrics(path: Path) -> WeightMetricSummary:
    if not path.exists():
        return WeightMetricSummary()

    rows: List[Dict[str, str]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append({str(key).strip(): str(value).strip() for key, value in row.items() if key is not None})
    except OSError:
        return WeightMetricSummary()

    if not rows:
        return WeightMetricSummary()

    best_row = max(rows, key=lambda row: _to_float(row.get(MAP5095_KEY), -1.0))
    final_row = rows[-1]
    return WeightMetricSummary(
        best_epoch=_to_int(best_row.get("epoch"), 0),
        best_precision=_to_float(best_row.get(PRECISION_KEY), 0.0),
        best_recall=_to_float(best_row.get(RECALL_KEY), 0.0),
        best_map50=_to_float(best_row.get(MAP50_KEY), 0.0),
        best_map5095=_to_float(best_row.get(MAP5095_KEY), 0.0),
        best_time_seconds=_to_float(best_row.get("time"), 0.0),
        final_epoch=_to_int(final_row.get("epoch"), 0),
        final_precision=_to_float(final_row.get(PRECISION_KEY), 0.0),
        final_recall=_to_float(final_row.get(RECALL_KEY), 0.0),
        final_map50=_to_float(final_row.get(MAP50_KEY), 0.0),
        final_map5095=_to_float(final_row.get(MAP5095_KEY), 0.0),
        total_epochs=len(rows),
    )


def _collect_artifacts(run_path: Path) -> TrainingArtifactPaths:
    return TrainingArtifactPaths(
        results_csv=_path_if_exists(run_path / "results.csv"),
        args_yaml=_path_if_exists(run_path / "args.yaml"),
        results_png=_path_if_exists(run_path / "results.png"),
        pr_curve=_path_if_exists(run_path / "BoxPR_curve.png"),
        f1_curve=_path_if_exists(run_path / "BoxF1_curve.png"),
        precision_curve=_path_if_exists(run_path / "BoxP_curve.png"),
        recall_curve=_path_if_exists(run_path / "BoxR_curve.png"),
        confusion_matrix=_path_if_exists(run_path / "confusion_matrix.png"),
        confusion_matrix_normalized=_path_if_exists(run_path / "confusion_matrix_normalized.png"),
    )


def _preserve_manual_fields(record: WeightTrainingRecord, existing: Dict[str, Any]) -> None:
    for field_name in MANUAL_FIELDS:
        value = existing.get(field_name)
        if value:
            setattr(record, field_name, str(value))


def _path_if_exists(path: Path) -> str:
    return _path_text(path) if path.exists() else ""


def _path_text(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)
    return cleaned.strip("._") or "model"


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_training_run_weights(
    store: WeightRegistryStore,
    models_dir: str | Path,
    run_roots: List[str | Path],
) -> List[Dict[str, Any]]:
    """Import available YOLO run directories so the GUI has local weights to load."""
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    imported: List[Dict[str, Any]] = []
    for run_dir in discover_training_runs(run_roots):
        try:
            record = store.import_training_run(run_dir, models_path)
        except Exception:
            continue
        imported.append(record)
    return imported


def discover_training_runs(run_roots: List[str | Path]) -> List[Path]:
    runs: List[Path] = []
    seen: set[str] = set()
    for root in run_roots:
        root_path = Path(root)
        if not root_path.exists() or not root_path.is_dir():
            continue
        for run_dir in sorted(root_path.iterdir(), key=lambda path: path.name.lower()):
            if not run_dir.is_dir():
                continue
            best_weight = run_dir / "weights" / "best.pt"
            if not best_weight.exists():
                continue
            key = str(run_dir.resolve())
            if key in seen:
                continue
            seen.add(key)
            runs.append(run_dir)
    return runs
