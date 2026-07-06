import csv
import shutil
import unittest
import uuid
from importlib.util import find_spec
from pathlib import Path

import numpy as np

if find_spec("cv2") is None:
    raise unittest.SkipTest("opencv-python is required for review dataset tests")

from core.review_dataset import (
    build_labelimg_args,
    detection_to_yolo_line,
    export_review_sample,
    normalize_class_names,
    resolve_labelimg_executable,
)


class ReviewDatasetTest(unittest.TestCase):
    def test_detection_to_yolo_line_clips_and_normalizes_box(self):
        line = detection_to_yolo_line(
            {"class_id": 2, "xyxy": [-10, 20, 110, 80]},
            image_width=100,
            image_height=100,
        )

        self.assertEqual(line, "2 0.500000 0.500000 1.000000 0.600000\n")

    def test_normalize_class_names_keeps_indices_unique(self):
        names = normalize_class_names(
            ["pl5", "pl10", "pl5"],
            [{"class_id": 2, "class_name": "pl5"}],
        )

        self.assertEqual(names, ["pl5", "pl10", "pl5__2"])

    def test_export_review_sample_writes_yolo_dataset_files_and_manifest(self):
        root = Path.cwd() / "outputs" / f"test_review_{uuid.uuid4().hex}"
        frame = np.full((100, 200, 3), 220, dtype=np.uint8)
        detections = [
            {
                "class_id": 1,
                "class_name": "speed",
                "confidence": 0.91,
                "xyxy": [20, 10, 120, 60],
            }
        ]
        metadata = {
            "sample_id": "sample_001",
            "source_kind": "image",
            "source_name": "demo.jpg",
            "source_path": "demo.jpg",
            "model_name": "best.pt",
            "frame_index": 1,
            "issue_reason": "漏标",
            "note": "small sign",
        }

        try:
            sample = export_review_sample(root, frame, detections, ["stop", "speed"], metadata)

            self.assertTrue(sample.image_path.exists())
            self.assertEqual(sample.label_path.read_text(encoding="utf-8"), "1 0.350000 0.350000 0.500000 0.500000\n")
            self.assertEqual((root / "classes.txt").read_text(encoding="utf-8"), "stop\nspeed\n")
            self.assertEqual((root / "labels" / "classes.txt").read_text(encoding="utf-8"), "stop\nspeed\n")
            self.assertIn("train: images", sample.data_yaml_path.read_text(encoding="utf-8"))

            with sample.manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["sample_id"], "sample_001")
            self.assertEqual(rows[0]["issue_reason"], "漏标")
            self.assertEqual(rows[0]["note"], "small sign")
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_export_review_sample_writes_empty_label_file_without_boxes(self):
        root = Path.cwd() / "outputs" / f"test_review_{uuid.uuid4().hex}"
        frame = np.zeros((30, 40, 3), dtype=np.uint8)

        try:
            sample = export_review_sample(
                root,
                frame,
                [],
                ["object"],
                {"sample_id": "empty_001", "issue_reason": "漏标"},
            )

            self.assertTrue(sample.label_path.exists())
            self.assertEqual(sample.label_path.read_text(encoding="utf-8"), "")
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_labelimg_command_helpers(self):
        root = Path.cwd()
        args = build_labelimg_args(root / "a.jpg", root / "classes.txt", root / "labels")
        fake_exe = root / "outputs" / f"fake_labelimg_{uuid.uuid4().hex}.exe"

        try:
            fake_exe.parent.mkdir(exist_ok=True)
            fake_exe.write_text("", encoding="utf-8")

            self.assertEqual(args, [str(root / "a.jpg"), str(root / "classes.txt"), str(root / "labels")])
            self.assertEqual(resolve_labelimg_executable(fake_exe), fake_exe)
        finally:
            fake_exe.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
