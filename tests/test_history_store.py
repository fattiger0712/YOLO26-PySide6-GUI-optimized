import csv
import json
import shutil
import unittest
import uuid
from pathlib import Path

from core.history_store import HistoryStore
from core.models import RunSummary


class HistoryStoreTest(unittest.TestCase):
    def _summary(self):
        return RunSummary(
            run_id="run-1",
            started_at="2026-07-05 12:00:00",
            ended_at="2026-07-05 12:00:02",
            source_type="image",
            source_name="sample.jpg",
            model_name="best.pt",
            conf=0.25,
            iou=0.7,
            rate_ms=30,
            save_results=True,
            save_txt=False,
            frames=1,
            duration_seconds=2.0,
            avg_fps=12.345,
            max_targets=3,
            final_class_count=2,
            final_target_count=3,
            total_target_events=3,
            class_counts={"stop": 2, "speed": 1},
            status="completed",
            output_dir="outputs/runs/run-1",
        )

    def test_append_writes_json_and_csv(self):
        temp_root = Path.cwd() / "outputs"
        temp_root.mkdir(exist_ok=True)
        tmp = temp_root / f"test_history_{uuid.uuid4().hex}"
        tmp.mkdir()
        try:
            store = HistoryStore(tmp)
            record = store.append(self._summary())

            self.assertEqual(record["run_id"], "run-1")
            self.assertTrue(store.json_path.exists())
            self.assertTrue(store.csv_path.exists())

            data = json.loads(store.json_path.read_text(encoding="utf-8"))
            self.assertEqual(data[0]["class_counts"]["stop"], 2)

            with store.csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["source_name"], "sample.jpg")
            self.assertEqual(json.loads(rows[0]["class_counts"])["speed"], 1)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_load_returns_empty_for_missing_file(self):
        temp_root = Path.cwd() / "outputs"
        temp_root.mkdir(exist_ok=True)
        tmp = temp_root / f"test_history_{uuid.uuid4().hex}"
        tmp.mkdir()
        try:
            self.assertEqual(HistoryStore(Path(tmp)).load(), [])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
