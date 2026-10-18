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
