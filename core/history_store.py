from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

from core.models import RunSummary


class HistoryStore:
    CSV_FIELDS = [
        "run_id",
        "started_at",
        "ended_at",
        "source_type",
        "source_name",
        "model_name",
        "conf",
        "iou",
        "rate_ms",
        "save_results",
        "save_txt",
        "frames",
        "duration_seconds",
        "avg_fps",
        "max_targets",
        "final_class_count",
        "final_target_count",
        "total_target_events",
        "class_counts",
        "status",
        "output_dir",
    ]

    def __init__(self, output_root: str | Path):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.json_path = self.output_root / "history.json"
        self.csv_path = self.output_root / "history.csv"

    def load(self) -> List[Dict]:
        if not self.json_path.exists():
            return []
        try:
            with self.json_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, list) else []
        except (OSError, json.JSONDecodeError):
            return []

    def append(self, summary: RunSummary) -> Dict:
        record = summary.to_dict()
        history = self.load()
        history.append(record)
        self._write_json(history)
        self._write_csv(history)
        return record

    def rewrite(self, records: Iterable[Dict]) -> None:
        data = list(records)
        self._write_json(data)
        self._write_csv(data)

    def _write_json(self, records: List[Dict]) -> None:
        with self.json_path.open("w", encoding="utf-8") as handle:
            json.dump(records, handle, ensure_ascii=False, indent=2)

    def _write_csv(self, records: List[Dict]) -> None:
        with self.csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.CSV_FIELDS)
            writer.writeheader()
            for record in records:
                row = {field: record.get(field, "") for field in self.CSV_FIELDS}
                row["class_counts"] = json.dumps(
                    row.get("class_counts") or {}, ensure_ascii=False
                )
                writer.writerow(row)

