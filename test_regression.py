import json
import tempfile
import unittest
from pathlib import Path

from food_ai.config import CATEGORIES, FRESHNESS_CLASSES
from scripts.check_duplicates import find_cross_split_duplicates


class ContractTests(unittest.TestCase):
    def test_category_order_matches_dataset_order(self):
        self.assertEqual(tuple(CATEGORIES), tuple(sorted(CATEGORIES)))

    def test_freshness_order_is_stable(self):
        self.assertEqual(FRESHNESS_CLASSES, ("FRESH", "HALF-FRESH", "SPOILED"))


class DuplicateDetectionTests(unittest.TestCase):
    def test_detects_identical_bytes_across_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train"
            valid = root / "valid"
            train.mkdir()
            valid.mkdir()
            (train / "one.jpg").write_bytes(b"same image")
            (valid / "renamed.png").write_bytes(b"same image")

            duplicates = find_cross_split_duplicates({"train": train, "valid": valid})

            self.assertEqual(len(duplicates), 1)
            self.assertEqual({split for split, _ in duplicates[0][1]}, {"train", "valid"})

    def test_does_not_report_unique_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train"
            valid = root / "valid"
            train.mkdir()
            valid.mkdir()
            (train / "one.jpg").write_bytes(b"train image")
            (valid / "one.jpg").write_bytes(b"valid image")

            self.assertEqual(find_cross_split_duplicates({"train": train, "valid": valid}), [])


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
