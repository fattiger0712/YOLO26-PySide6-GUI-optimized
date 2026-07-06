import unittest
from importlib.util import find_spec

import numpy as np

if find_spec("cv2") is None:
    raise unittest.SkipTest("opencv-python is required for saved rendering tests")

from core.detection_worker import DetectionWorker
from core.models import DetectionConfig, SourceSpec


class FakeBoxes:
    def __init__(self, xyxy, cls=None, conf=None):
        self.xyxy = np.asarray(xyxy, dtype=float)
        self.cls = np.asarray(cls if cls is not None else [], dtype=float)
        self.conf = np.asarray(conf if conf is not None else [], dtype=float)


class FakeResult:
    def __init__(self, boxes=None):
        self.boxes = boxes
        self.names = {0: "stop", 1: "speed"}


class SavedRenderingTest(unittest.TestCase):
    def _worker(self):
        return DetectionWorker(
            DetectionConfig(model_path="dummy.pt", save_results=True),
            SourceSpec.from_file("sample.jpg"),
        )

    def test_render_saved_result_keeps_shape_and_does_not_mutate_source(self):
        worker = self._worker()
        frame = np.full((80, 100, 3), 240, dtype=np.uint8)
        original = frame.copy()
        result = FakeResult(FakeBoxes([[10, 20, 60, 55]], [0], [0.87]))

        rendered = worker._render_saved_result(frame, result)

        self.assertEqual(rendered.shape, frame.shape)
        self.assertTrue(np.array_equal(frame, original))
        self.assertFalse(np.array_equal(rendered, original))

    def test_render_saved_result_handles_no_boxes(self):
        worker = self._worker()
        frame = np.full((40, 50, 3), 120, dtype=np.uint8)

        rendered = worker._render_saved_result(frame, FakeResult())

        self.assertEqual(rendered.shape, frame.shape)
        self.assertIsNot(rendered, frame)
        self.assertTrue(np.array_equal(rendered, frame))

    def test_render_saved_result_handles_empty_boxes(self):
        worker = self._worker()
        frame = np.full((40, 50, 3), 90, dtype=np.uint8)
        result = FakeResult(FakeBoxes([], [], []))

        rendered = worker._render_saved_result(frame, result)

        self.assertEqual(rendered.shape, frame.shape)
        self.assertTrue(np.array_equal(rendered, frame))


if __name__ == "__main__":
    unittest.main()
