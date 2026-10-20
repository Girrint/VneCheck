import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from food_ai.data import build_transforms
from local_web.server import (
    LocalEvaluationApp,
    LegacyOnnxRuntimeModel,
    OnnxRuntimeModel,
    REPORT_FIELDS,
    preprocess_pil_image,
    sample_metadata,
)


class LocalWebTests(unittest.TestCase):
    def test_numpy_preprocessing_matches_training_eval_transform(self):
        pixels = np.arange(317 * 241 * 3, dtype=np.uint32)
        pixels = (pixels % 256).astype(np.uint8).reshape(241, 317, 3)
        image = Image.fromarray(pixels)
        _, transform = build_transforms()
        expected = transform(image.copy()).numpy()
        actual = preprocess_pil_image(image.copy())
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)

    def test_metadata_uses_dataset_label_structure(self):
        sample = sample_metadata("qua/FRESH/apple/sample.png")
        self.assertEqual(sample["category"], "qua")
        self.assertEqual(sample["freshness"], "FRESH")
        self.assertIsNone(sample_metadata("unknown/FRESH/apple/ignored.png"))

    def test_resolve_rejects_paths_outside_index(self):
        app = LocalEvaluationApp.__new__(LocalEvaluationApp)
        app.test_root = Path.cwd()
        app.by_path = {}
        with self.assertRaises(FileNotFoundError):
            app.resolve_sample_path("../private.txt")

    def test_report_path_rejects_directory_traversal(self):
        app = LocalEvaluationApp.__new__(LocalEvaluationApp)
        app.report_root = Path.cwd() / "reports" / "web_evaluations"
        with self.assertRaises(FileNotFoundError):
            app.resolve_report_path("../private.csv")

    def test_decode_masks_half_fresh_for_two_class_category(self):
        model = OnnxRuntimeModel.__new__(OnnxRuntimeModel)
        model.food_gate_ready = False
        result = model._decode(
            np.array([8.0, 0.0, 0.0]),
            np.array([8.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([1.0, 10.0, 2.0]),
        )
        self.assertEqual(result["category"], "cu")
        self.assertEqual(result["freshness"], "SPOILED")
        self.assertNotIn("HALF-FRESH", result["probabilities"]["freshness"])

    def test_legacy_two_class_expert_decoding(self):
        model = LegacyOnnxRuntimeModel.__new__(LegacyOnnxRuntimeModel)
        model.freshness_classes = {"cu": ["FRESH", "SPOILED"]}
        result = model._decode_freshness("cu", np.array([1.0, 4.0]))
        self.assertEqual(result["freshness"], "SPOILED")
        self.assertEqual(result["class_count"], 2)
        self.assertEqual(
            set(result["freshness_probabilities"]), {"FRESH", "SPOILED"}
        )

    def test_csv_report_contract_contains_detailed_predictions(self):
        self.assertEqual(len(REPORT_FIELDS), len(set(REPORT_FIELDS)))
        for field in (
            "image_path",
            "truth_freshness",
            "unified_category_confidence",
            "legacy_pipeline_freshness",
            "legacy_oracle_correct",
            "legacy_router_probability_cu",
        ):
            self.assertIn(field, REPORT_FIELDS)


if __name__ == "__main__":
    unittest.main()
