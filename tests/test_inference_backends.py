import unittest
from importlib.util import find_spec

import numpy as np

if find_spec("cv2") is None:
    raise unittest.SkipTest("opencv-python is required for inference backend tests")

from core.inference import (
    LetterboxInfo,
    classwise_nms,
    model_backend,
    postprocess_yolo_outputs,
)


class InferenceBackendTest(unittest.TestCase):
    def test_model_backend_uses_suffix(self):
        self.assertEqual(model_backend("best.pt"), "pt")
        self.assertEqual(model_backend("best.onnx"), "onnx")
        with self.assertRaises(ValueError):
            model_backend("best.engine")

    def test_postprocess_restores_letterboxed_xywh_coordinates(self):
        image_shape = (100, 200, 3)
        info = LetterboxInfo(ratio=1.6, pad_x=0.0, pad_y=80.0, input_width=320, input_height=320)
        raw = np.zeros((1, 6, 1), dtype=np.float32)
        raw[0, :, 0] = [160.0, 160.0, 160.0, 96.0, 0.9, 0.1]

        result = postprocess_yolo_outputs([raw], image_shape, info, {0: "stop", 1: "speed"}, 0.25, 0.7)

        boxes = result.boxes.xyxy.tolist()
        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(boxes[0][0], 50.0, places=3)
        self.assertAlmostEqual(boxes[0][1], 20.0, places=3)
        self.assertAlmostEqual(boxes[0][2], 150.0, places=3)
        self.assertAlmostEqual(boxes[0][3], 80.0, places=3)
        self.assertEqual(int(result.boxes.cls[0]), 0)

    def test_classwise_nms_suppresses_same_class_overlap_only(self):
        boxes = np.array(
            [
                [10, 10, 60, 60],
                [12, 12, 62, 62],
                [12, 12, 62, 62],
            ],
            dtype=np.float32,
        )
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        classes = np.array([0, 0, 1], dtype=np.float32)

        keep = classwise_nms(boxes, scores, classes, 0.5)

        self.assertEqual(keep, [0, 2])


if __name__ == "__main__":
    unittest.main()
