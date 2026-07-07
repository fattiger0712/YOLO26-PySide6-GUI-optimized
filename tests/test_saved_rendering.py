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

    def test_find_label_position_avoids_existing_labels(self):
        first = DetectionWorker._find_label_position(
            20,
            30,
            90,
            90,
            "stop 0.90",
            0.55,
            2,
            160,
            120,
            [],
        )
        second = DetectionWorker._find_label_position(
            22,
            32,
            92,
            92,
            "speed 0.80",
            0.55,
            2,
            160,
            120,
            [first],
        )

        self.assertFalse(self._rects_overlap(first, second))

    def test_render_saved_result_uses_translucent_overlays(self):
        worker = self._worker()
        frame = np.full((140, 180, 3), 180, dtype=np.uint8)
        result = FakeResult(
            FakeBoxes(
                [[20, 40, 100, 110], [24, 42, 104, 112]],
                [0, 1],
                [0.91, 0.82],
            )
        )

        rendered = worker._render_saved_result(frame, result)

        self.assertEqual(rendered.shape, frame.shape)
        self.assertFalse(np.array_equal(rendered, frame))
        changed_pixels = np.count_nonzero(np.any(rendered != frame, axis=2))
        self.assertGreater(changed_pixels, 100)

    @staticmethod
    def _rects_overlap(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        return not (ax + aw + 2 < bx or ax > bx + bw + 2 or ay + ah + 2 < by or ay > by + bh + 2)


if __name__ == "__main__":
    unittest.main()
