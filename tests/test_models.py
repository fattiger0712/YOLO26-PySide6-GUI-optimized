import unittest
import shutil
import uuid
from pathlib import Path

from core.models import SourceSpec, detect_file_kind, iter_supported_media, merge_counts


class ModelHelpersTest(unittest.TestCase):
    def test_detect_file_kind(self):
        self.assertEqual(detect_file_kind("demo.JPG"), "image")
        self.assertEqual(detect_file_kind(Path("demo.mp4")), "video")

    def test_detect_file_kind_rejects_unknown_suffix(self):
        with self.assertRaises(ValueError):
            detect_file_kind("weights.pt")

    def test_source_spec_from_file(self):
        source = SourceSpec.from_file("sample.png")
        self.assertEqual(source.kind, "image")
        self.assertEqual(source.display_name, "sample.png")

    def test_source_spec_batch(self):
        source = SourceSpec.batch("dataset")
        self.assertEqual(source.kind, "batch")
        self.assertEqual(source.display_name, "dataset")

    def test_iter_supported_media_is_recursive_and_sorted(self):
        root = Path.cwd() / "outputs" / f"test_media_{uuid.uuid4().hex}"
        nested = root / "nested"
        nested.mkdir(parents=True)
        try:
            (root / "b.mp4").write_bytes(b"video")
            (root / "a.jpg").write_bytes(b"image")
            (nested / "c.png").write_bytes(b"image")
            (root / "ignore.txt").write_text("skip", encoding="utf-8")
            files = [path.relative_to(root).as_posix() for path in iter_supported_media(root)]
            self.assertEqual(files, ["a.jpg", "b.mp4", "nested/c.png"])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_merge_counts(self):
        total = {"car": 2}
        self.assertEqual(merge_counts(total, {"car": 3, "bus": 1}), {"car": 5, "bus": 1})


if __name__ == "__main__":
    unittest.main()
