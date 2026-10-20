import json
import unittest
from pathlib import Path
from unittest.mock import patch

from food_ai.config import CATEGORIES, FRESHNESS_CLASSES
from scripts.resplit_dataset import prepare_unique_records, split_clusters


class ContractTests(unittest.TestCase):
    def test_category_order_matches_dataset_order(self):
        self.assertEqual(tuple(CATEGORIES), tuple(sorted(CATEGORIES)))

    def test_freshness_order_is_stable(self):
        self.assertEqual(FRESHNESS_CLASSES, ("FRESH", "HALF-FRESH", "SPOILED"))


class DatasetSplitTests(unittest.TestCase):
    @staticmethod
    def record(path, class_name, sha256, phash=0):
        return {
            "path": path,
            "class": class_name,
            "sha256": sha256,
            "phash": phash,
        }

    def test_exact_duplicate_same_label_keeps_one_representative(self):
        records = [
            self.record("b.jpg", "qua/FRESH/apple", "same"),
            self.record("a.jpg", "qua/FRESH/apple", "same"),
        ]

        kept, excluded = prepare_unique_records(records)

        self.assertEqual([record["path"] for record in kept], ["a.jpg"])
        self.assertEqual(excluded[0]["status"], "exact_duplicate")
        self.assertEqual(excluded[0]["kept_path"], "a.jpg")

    def test_exact_duplicate_with_conflicting_labels_is_quarantined(self):
        records = [
            self.record("fresh.jpg", "qua/FRESH/orange", "same"),
            self.record("spoiled.jpg", "qua/SPOILED/orange", "same"),
        ]

        kept, excluded = prepare_unique_records(records)

        self.assertEqual(kept, [])
        self.assertEqual(len(excluded), 2)
        self.assertTrue(all(row["status"] == "label_conflict" for row in excluded))

    def test_cluster_is_never_split_between_partitions(self):
        cluster = [
            self.record("one.jpg", "cu/FRESH/carrot", "one"),
            self.record("two.jpg", "cu/FRESH/carrot", "two"),
        ]
        singles = [
            [self.record(f"single-{index}.jpg", "cu/FRESH/carrot", str(index))]
            for index in range(20)
        ]

        plan, _, _ = split_clusters(
            [cluster, *singles], (0.7, 0.15, 0.15), seed=42
        )

        destinations = {
            split
            for split, records in plan.items()
            if any(record["path"] in {"one.jpg", "two.jpg"} for record in records)
        }
        self.assertEqual(len(destinations), 1)


class PreprocessingTests(unittest.TestCase):
    def test_eval_transform_has_model_input_shape(self):
        try:
            import numpy as np
            from PIL import Image
            from food_ai.data import build_transforms
        except ImportError as error:
            self.skipTest(f"optional ML dependency unavailable: {error}")

        _, eval_transform = build_transforms()
        image = Image.new("RGB", (640, 360), color=(100, 120, 140))
        tensor = eval_transform(image)

        self.assertEqual(tuple(tensor.shape), (3, 224, 224))
        self.assertEqual(tensor.dtype, __import__("torch").float32)
        self.assertEqual(tensor.numpy().dtype, np.float32)

    def test_loaders_read_three_independent_splits(self):
        try:
            import torch
            from food_ai.data import create_loaders
        except ImportError as error:
            self.skipTest(f"optional ML dependency unavailable: {error}")

        roots_seen = []

        class FakeDataset:
            classes = ["FRESH"]

            def __init__(self, root, transform=None):
                roots_seen.append(Path(root))

            def __len__(self):
                return 1

            def __getitem__(self, index):
                return torch.zeros(3, 224, 224), 0

        with patch("food_ai.data.NonEmptyImageFolder", FakeDataset):
            train, valid, test, classes = create_loaders(
                "train-root", "valid-root", "test-root", 1, 0
            )

        self.assertEqual(
            roots_seen,
            [Path("train-root"), Path("valid-root"), Path("test-root")],
        )
        self.assertEqual(classes, ["FRESH"])
        self.assertEqual(len(train.dataset), 1)
        self.assertEqual(len(valid.dataset), 1)
        self.assertEqual(len(test.dataset), 1)


class UnifiedModelTests(unittest.TestCase):
    def test_sqrt_inverse_food_weights_follow_class_order(self):
        try:
            from food_ai.unified_loss import compute_class_weights
        except ImportError as error:
            self.skipTest(f"optional ML dependency unavailable: {error}")

        weights = compute_class_weights(
            {"SUPPORTED": 66115, "UNSUPPORTED": 3500, "NON_FOOD": 3500},
            ("SUPPORTED", "UNSUPPORTED", "NON_FOOD"),
            "sqrt_inverse",
        )

        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        self.assertLess(float(weights[0]), float(weights[1]))
        self.assertAlmostEqual(float(weights[1]), float(weights[2]), places=6)

    def test_food_weights_reject_missing_class(self):
        try:
            from food_ai.unified_loss import compute_class_weights
        except ImportError as error:
            self.skipTest(f"optional ML dependency unavailable: {error}")

        with self.assertRaises(ValueError):
            compute_class_weights(
                {"SUPPORTED": 10},
                ("SUPPORTED", "UNSUPPORTED", "NON_FOOD"),
                "sqrt_inverse",
            )

    def test_unified_model_has_three_expected_outputs(self):
        try:
            import torch
            from food_ai.unified_model import create_unified_model
        except ImportError as error:
            self.skipTest(f"optional ML dependency unavailable: {error}")

        model = create_unified_model("mobilenet_v3_small", pretrained=False)
        model.eval()
        with torch.inference_mode():
            outputs = model(torch.zeros(2, 3, 224, 224))
        self.assertEqual(
            [tuple(output.shape) for output in outputs],
            [(2, 3), (2, 5), (2, 3)],
        )

    def test_masked_loss_ignores_missing_category_and_freshness(self):
        try:
            import torch
            from food_ai.unified_loss import MaskedMultiTaskLoss
        except ImportError as error:
            self.skipTest(f"optional ML dependency unavailable: {error}")

        outputs = (
            torch.randn(2, 3, requires_grad=True),
            torch.randn(2, 5, requires_grad=True),
            torch.randn(2, 3, requires_grad=True),
        )
        targets = {
            "food_status": torch.tensor([1, 2]),
            "category": torch.tensor([-100, -100]),
            "freshness": torch.tensor([-100, -100]),
        }
        loss, components = MaskedMultiTaskLoss()(outputs, targets)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(components["category"].detach()), 0.0)
        self.assertEqual(float(components["freshness"].detach()), 0.0)


class LocalWebNegativeTests(unittest.TestCase):
    def test_invalid_model_mode_is_rejected(self):
        from local_web.server import LocalEvaluationApp

        with self.assertRaises(ValueError):
            LocalEvaluationApp.validate_model_mode("unknown")

    def test_negative_path_maps_to_food_status_only(self):
        from local_web.server import negative_sample_metadata

        sample = negative_sample_metadata("NON_FOOD/example.jpg")

        self.assertEqual(sample["food_status"], "NON_FOOD")
        self.assertEqual(sample["dataset"], "negative")
        self.assertEqual(sample["category"], "")
        self.assertEqual(sample["freshness"], "")

    def test_mixed_metrics_only_score_category_on_supported(self):
        import io
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from local_web.server import LocalEvaluationApp

        samples = [
            {
                "path": "cu/FRESH/carrot/a.jpg", "dataset": "supported",
                "food_status": "SUPPORTED", "category": "cu",
                "freshness": "FRESH", "product": "carrot",
            },
            {
                "path": "UNSUPPORTED/b.jpg", "dataset": "negative",
                "food_status": "UNSUPPORTED", "category": "",
                "freshness": "", "product": "",
            },
            {
                "path": "NON_FOOD/c.jpg", "dataset": "negative",
                "food_status": "NON_FOOD", "category": "",
                "freshness": "", "product": "",
            },
        ]
        predictions = []
        for status in ("SUPPORTED", "UNSUPPORTED", "SUPPORTED"):
            predictions.append({
                "food_status": status,
                "food_confidence": 0.9,
                "food_gate_ready": True,
                "category": "cu",
                "category_confidence": 0.9,
                "freshness": "FRESH",
                "freshness_confidence": 0.9,
                "estimated_freshness_score": 9.0,
                "probabilities": {
                    "food": {"SUPPORTED": 0.8, "UNSUPPORTED": 0.1, "NON_FOOD": 0.1},
                    "category": {name: (0.8 if name == "cu" else 0.05) for name in CATEGORIES},
                    "freshness": {name: (0.8 if name == "FRESH" else 0.1) for name in FRESHNESS_CLASSES},
                },
            })

        app = LocalEvaluationApp.__new__(LocalEvaluationApp)
        app.samples = samples
        app.by_path = {sample["path"]: sample for sample in samples}
        app.model = SimpleNamespace(
            ready=True,
            error=None,
            food_gate_ready=True,
            model_id="test-model",
            display_name="Test model",
            architecture="test_arch",
            descriptor=lambda: {
                "id": "test-model",
                "display_name": "Test model",
                "architecture": "test_arch",
            },
            predict_batch=lambda paths: predictions[:len(paths)],
        )
        app.unified_model = lambda model_id=None: app.model
        app.legacy_model = SimpleNamespace(ready=False, error="disabled")
        app.resolve_sample_path = lambda path: Path(path)

        csv_path = MagicMock()
        csv_path.name = "test.csv"
        csv_path.open.return_value.__enter__.return_value = io.StringIO()
        summary_path = MagicMock()
        summary_path.name = "test.json"
        app._report_paths = lambda: (csv_path, summary_path)
        result = app.evaluate(None, None, None, 0, 42)

        metrics = result["models"]["unified"]["metrics"]
        self.assertAlmostEqual(metrics["food_accuracy"], 2 / 3)
        self.assertEqual(metrics["category_accuracy"], 1.0)
        self.assertEqual(metrics["freshness_accuracy"], 1.0)
        self.assertEqual(metrics["food_recall"]["NON_FOOD"], 0.0)
        self.assertEqual(metrics["supported_count"], 1)


class UnifiedExportTests(unittest.TestCase):
    def test_artifact_name_is_filesystem_safe(self):
        from export_unified_onnx import safe_artifact_name

        self.assertEqual(
            safe_artifact_name("MobileNet V3 Small / v2"),
            "MobileNet_V3_Small_v2",
        )

    def test_empty_artifact_name_is_rejected(self):
        from export_unified_onnx import safe_artifact_name

        with self.assertRaises(ValueError):
            safe_artifact_name(" / ")


class ManifestTests(unittest.TestCase):
    def test_manifest_has_required_contract_fields(self):
        manifest_path = Path("models/manifest.json")
        if not manifest_path.exists():
            self.skipTest("manifest chưa được tạo; chạy package_models.py trước")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["input"]["shape"], [1, 3, 224, 224])
        self.assertEqual(manifest["input"]["dtype"], "float32")
        self.assertIn("preprocessing", manifest)
        self.assertTrue(manifest["models"])
        for model in manifest["models"]:
            self.assertRegex(model["checkpoint"]["sha256"], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
