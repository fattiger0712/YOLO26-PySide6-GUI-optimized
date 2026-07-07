import json
import shutil
import unittest
import uuid
from pathlib import Path

from core.weight_registry import WeightRegistryStore, build_training_record, ensure_training_run_weights


class WeightRegistryTest(unittest.TestCase):
    def setUp(self):
        self.root = Path.cwd() / "outputs" / f"test_weight_registry_{uuid.uuid4().hex}"
        self.run_dir = self.root / "runs" / "tt100k_demo"
        self.models_dir = self.root / "models"
        (self.run_dir / "weights").mkdir(parents=True)
        self.models_dir.mkdir(parents=True)
        (self.run_dir / "weights" / "best.pt").write_bytes(b"first-weight")
        (self.run_dir / "args.yaml").write_text(
            "\n".join(
                [
                    "task: detect",
                    "model: yolo26s.pt",
                    "data: tt100k.yaml",
                    "epochs: 3",
                    "batch: 8",
                    "imgsz: 1024",
                    "device: '0'",
                ]
            ),
            encoding="utf-8",
        )
        (self.run_dir / "results.csv").write_text(
            "\n".join(
                [
                    "epoch,time,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)",
                    "1,10,0.50,0.40,0.60,0.30",
                    "2,20,0.70,0.60,0.80,0.55",
                    "3,30,0.65,0.62,0.79,0.50",
                ]
            ),
            encoding="utf-8",
        )
        (self.run_dir / "results.png").write_bytes(b"png")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_build_training_record_uses_best_map5095_epoch(self):
        model_path = self.models_dir / "demo_best.pt"
        model_path.write_bytes(b"model")

        record = build_training_record(self.run_dir, model_path)

        self.assertEqual(record.model_name, "demo_best.pt")
        self.assertEqual(record.training_name, "tt100k_demo")
        self.assertEqual(record.dataset, "tt100k.yaml")
        self.assertEqual(record.base_model, "yolo26s.pt")
        self.assertEqual(record.epochs, 3)
        self.assertEqual(record.imgsz, "1024")
        self.assertEqual(record.metrics["best_epoch"], 2)
        self.assertEqual(record.metrics["final_epoch"], 3)
        self.assertAlmostEqual(record.metrics["best_map5095"], 0.55)
        self.assertTrue(record.artifacts["results_png"].endswith("results.png"))
        self.assertEqual(record.artifacts["pr_curve"], "")

    def test_import_training_run_copies_weight_without_overwriting(self):
        store = WeightRegistryStore(self.root / "model_weights.json")

        first = store.import_training_run(self.run_dir, self.models_dir)
        target = self.models_dir / "tt100k_demo_best.pt"
        self.assertTrue(target.exists())
        self.assertEqual(target.read_bytes(), b"first-weight")

        first["notes"] = "works well for traffic signs"
        first["recommended_for"] = "TT100K detection"
        store.upsert(first)
        (self.run_dir / "weights" / "best.pt").write_bytes(b"second-weight")
        (self.run_dir / "results.csv").write_text(
            "\n".join(
                [
                    "epoch,time,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)",
                    "1,10,0.80,0.70,0.90,0.66",
                ]
            ),
            encoding="utf-8",
        )

        second = store.import_training_run(self.run_dir, self.models_dir)

        self.assertEqual(target.read_bytes(), b"first-weight")
        self.assertEqual(second["notes"], "works well for traffic signs")
        self.assertEqual(second["recommended_for"], "TT100K detection")
        self.assertAlmostEqual(second["metrics"]["best_map5095"], 0.66)

        data = json.loads((self.root / "model_weights.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["model_name"], "tt100k_demo_best.pt")

    def test_remove_from_manager_hides_record_without_deleting_weight(self):
        store = WeightRegistryStore(self.root / "model_weights.json")
        first = store.import_training_run(self.run_dir, self.models_dir)
        target = self.models_dir / first["model_name"]

        removed_existing = store.remove_from_manager(first["model_name"])

        self.assertTrue(removed_existing)
        self.assertTrue(target.exists())
        self.assertEqual(store.load(), [])
        self.assertEqual(store.hidden_model_names(), {first["model_name"]})
        hidden = store.load(include_removed=True)
        self.assertEqual(len(hidden), 1)
        self.assertTrue(hidden[0]["removed_from_manager"])

        restored = store.import_training_run(self.run_dir, self.models_dir)
        self.assertEqual(restored["model_name"], first["model_name"])
        self.assertFalse(restored.get("removed_from_manager", False))
        self.assertEqual(store.hidden_model_names(), set())
        self.assertEqual(len(store.load()), 1)

    def test_ensure_training_run_weights_imports_runs_into_models_dir(self):
        store = WeightRegistryStore(self.root / "model_weights.json")

        imported = ensure_training_run_weights(store, self.models_dir, [self.root / "runs"])

        self.assertEqual(len(imported), 1)
        self.assertTrue((self.models_dir / "tt100k_demo_best.pt").exists())
        self.assertEqual(store.load()[0]["model_name"], "tt100k_demo_best.pt")
        self.assertEqual(Path(store.load()[0]["model_path"]), self.models_dir / "tt100k_demo_best.pt")


if __name__ == "__main__":
    unittest.main()
