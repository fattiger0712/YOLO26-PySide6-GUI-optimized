import shutil
import unittest
import uuid
from importlib.util import find_spec
from pathlib import Path

import numpy as np

if find_spec("cv2") is None:
    raise unittest.SkipTest("opencv-python is required for dataset checker tests")

import cv2

from core.dataset_checker import check_dataset, load_yolo_dataset_source


class DatasetCheckerTest(unittest.TestCase):
    def setUp(self):
        self.root = Path.cwd() / "outputs" / f"test_dataset_checker_{uuid.uuid4().hex}"
        self.dataset = self.root / "dataset"
        (self.dataset / "images" / "train").mkdir(parents=True)
        (self.dataset / "labels" / "train").mkdir(parents=True)
        self.output_root = self.root / "reports"

        cv2.imwrite(str(self.dataset / "images" / "train" / "ok.jpg"), np.full((100, 100, 3), 240, dtype=np.uint8))
        cv2.imwrite(str(self.dataset / "images" / "train" / "missing.jpg"), np.full((100, 100, 3), 220, dtype=np.uint8))
        cv2.imwrite(str(self.dataset / "images" / "train" / "bad_class.jpg"), np.full((100, 100, 3), 200, dtype=np.uint8))
        (self.dataset / "images" / "train" / "broken.jpg").write_bytes(b"not an image")

        (self.dataset / "labels" / "train" / "ok.txt").write_text(
            "\n".join(
                [
                    "0 0.500000 0.500000 0.200000 0.200000",
                    "1 0.500000 0.500000 0.800000 0.800000",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (self.dataset / "labels" / "train" / "bad_class.txt").write_text(
            "2 0.500000 0.500000 0.100000 0.100000\nbad line\n",
            encoding="utf-8",
        )
        (self.dataset / "labels" / "train" / "orphan.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        (self.dataset / "data.yaml").write_text(
            "\n".join(
                [
                    "path: .",
                    "train: images/train",
                    "nc: 2",
                    "names:",
                    "  0: stop",
                    "  1: speed",
                ]
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_load_source_from_data_yaml(self):
        source = load_yolo_dataset_source(self.dataset / "data.yaml")

        self.assertEqual(source.names, ["stop", "speed"])
        self.assertEqual(len(source.image_paths), 4)

    def test_check_dataset_reports_counts_and_artifacts(self):
        report = check_dataset(self.dataset / "data.yaml", self.output_root)
        summary = report["summary"]

        self.assertEqual(summary["total_images"], 4)
        self.assertEqual(summary["total_boxes"], 2)
        self.assertEqual(summary["missing_labels"], 1)
        self.assertEqual(summary["bad_images"], 1)
        self.assertEqual(summary["class_id_out_of_bounds"], 1)
        self.assertEqual(summary["invalid_label_lines"], 1)
        self.assertEqual(summary["orphan_labels"], 1)
        self.assertEqual(summary["small_targets"], 1)

        issue_types = {issue["issue_type"] for issue in report["issues"]}
        self.assertIn("missing_label", issue_types)
        self.assertIn("bad_image", issue_types)
        self.assertIn("class_id_out_of_bounds", issue_types)
        self.assertIn("orphan_label", issue_types)

        output_dir = Path(report["output_dir"])
        self.assertTrue((output_dir / "check_report.json").exists())
        self.assertTrue((output_dir / "check_report.csv").exists())
        self.assertTrue((output_dir / "check_report.txt").exists())
        for path in report["artifacts"].values():
            self.assertTrue(Path(path).exists())


if __name__ == "__main__":
    unittest.main()
