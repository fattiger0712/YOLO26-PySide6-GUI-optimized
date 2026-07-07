import shutil
import unittest
import uuid
from importlib.util import find_spec
from pathlib import Path

import numpy as np

if find_spec("cv2") is None:
    raise unittest.SkipTest("opencv-python is required for evaluation tests")

import cv2

from core.evaluation_worker import EvaluationConfig, box_iou, evaluate_dataset, render_evaluation_image


class EvaluationWorkerTest(unittest.TestCase):
    def setUp(self):
        self.root = Path.cwd() / "outputs" / f"test_evaluation_{uuid.uuid4().hex}"
        self.dataset = self.root / "dataset"
        (self.dataset / "images").mkdir(parents=True)
        (self.dataset / "labels").mkdir(parents=True)
        cv2.imwrite(str(self.dataset / "images" / "good.jpg"), np.full((100, 100, 3), 230, dtype=np.uint8))
        cv2.imwrite(str(self.dataset / "images" / "missed_small.jpg"), np.full((100, 100, 3), 210, dtype=np.uint8))
        (self.dataset / "labels" / "good.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
        (self.dataset / "labels" / "missed_small.txt").write_text("1 0.2 0.2 0.2 0.2\n", encoding="utf-8")
        (self.dataset / "data.yaml").write_text(
            "\n".join(["path: .", "train: images", "nc: 2", "names: [stop, speed]"]),
            encoding="utf-8",
        )

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_box_iou(self):
        self.assertAlmostEqual(box_iou([0, 0, 10, 10], [5, 5, 15, 15]), 25 / 175)

    def test_evaluate_dataset_writes_metrics_and_failure_cases(self):
        def predictor(_frame, image_path):
            if image_path.name == "good.jpg":
                return [{"class_id": 0, "confidence": 0.9, "xyxy": [30, 30, 70, 70]}]
            return [
                {"class_id": 0, "confidence": 0.8, "xyxy": [10, 10, 30, 30]},
                {"class_id": 1, "confidence": 0.5, "xyxy": [70, 70, 80, 80]},
            ]

        config = EvaluationConfig(
            model_path="fake.pt",
            dataset_path=str(self.dataset / "data.yaml"),
            output_root=str(self.root / "reports"),
            iou=0.5,
            max_images=2,
        )
        report = evaluate_dataset(config, predictor=predictor)
        summary = report["summary"]

        self.assertEqual(summary["evaluated_images"], 2)
        self.assertEqual(summary["ground_truth_boxes"], 2)
        self.assertEqual(summary["predicted_boxes"], 3)
        self.assertEqual(summary["true_positive"], 1)
        self.assertEqual(summary["false_positive"], 2)
        self.assertEqual(summary["false_negative"], 1)
        self.assertAlmostEqual(summary["precision"], 1 / 3, places=5)
        self.assertAlmostEqual(summary["recall"], 0.5, places=5)
        self.assertEqual(summary["top_false_negative_class"], "speed")

        case_types = {case["case_type"] for case in report["failure_cases"]}
        self.assertIn("false_positive", case_types)
        self.assertIn("false_negative", case_types)
        self.assertIn("class_error", case_types)
        self.assertIn("small_target_failure", case_types)

        output_dir = Path(report["output_dir"])
        self.assertTrue((output_dir / "evaluation_report.json").exists())
        self.assertTrue((output_dir / "per_class_metrics.csv").exists())
        self.assertTrue((output_dir / "failure_cases.csv").exists())
        self.assertTrue(Path(report["artifacts"]["predictions_dir"]).exists())
        self.assertTrue(Path(report["artifacts"]["errors_dir"]).exists())

        prediction_images = list(Path(report["artifacts"]["predictions_dir"]).glob("*.jpg"))
        error_images = list(Path(report["artifacts"]["errors_dir"]).rglob("*.jpg"))
        self.assertTrue(prediction_images)
        self.assertTrue(error_images)
        self.assertIsNotNone(cv2.imread(str(prediction_images[0])))
        self.assertIsNotNone(cv2.imread(str(error_images[0])))

    def test_render_evaluation_image_uses_translucent_non_overlapping_style(self):
        frame = np.full((140, 180, 3), 180, dtype=np.uint8)
        truths = [
            {"class_id": 0, "class_name": "stop", "xyxy": [20, 40, 100, 110], "small": False},
            {"class_id": 1, "class_name": "speed", "xyxy": [24, 42, 104, 112], "small": False},
        ]
        predictions = [
            {"class_id": 0, "class_name": "stop", "confidence": 0.91, "xyxy": [22, 42, 102, 112]},
            {"class_id": 1, "class_name": "speed", "confidence": 0.83, "xyxy": [28, 46, 108, 116]},
        ]

        rendered = render_evaluation_image(frame, predictions, truths, ["stop", "speed"])

        self.assertEqual(rendered.shape, frame.shape)
        self.assertFalse(np.array_equal(rendered, frame))
        changed_pixels = np.count_nonzero(np.any(rendered != frame, axis=2))
        self.assertGreater(changed_pixels, 100)


if __name__ == "__main__":
    unittest.main()
