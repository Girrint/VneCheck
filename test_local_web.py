import csv
import io
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from food_ai.data import build_transforms
from food_ai.config import ROUTER_CLASSES
from local_web.server import (
    LocalEvaluationApp,
    LegacyOnnxRuntimeModel,
    OnnxRuntimeModel,
    REPORT_FIELDS,
    preprocess_pil_image,
    sample_metadata,
    summarize_speed,
)


class LocalWebTests(unittest.TestCase):
    def test_speed_summary_reports_batch_amortized_latency(self):
        summary = summarize_speed(
            [(0.2, 10), (0.3, 10)],
            artifact_bytes=5 * 1024 * 1024,
            batch_size=10,
        )

        self.assertEqual(summary["measured_images"], 20)
        self.assertEqual(summary["measured_batches"], 2)
        self.assertEqual(summary["mean_ms_per_image"], 25.0)
        self.assertEqual(summary["p50_ms_per_image"], 25.0)
        self.assertEqual(summary["p95_ms_per_image"], 30.0)
        self.assertEqual(summary["throughput_images_per_second"], 40.0)
        self.assertEqual(summary["artifact_size_mb"], 5.0)
        self.assertEqual(
            summary["latency_basis"],
            "batch_total_divided_by_image_count",
        )

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

    def test_report_rows_paginates_images_and_keeps_models_together(self):
        stream = io.StringIO()
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "image_path", "evaluation_model_id", "evaluation_model_type",
                "evaluation_available", "unified_joint_correct",
                "legacy_joint_correct",
            ],
        )
        writer.writeheader()
        writer.writerows([
            {"image_path": "a.jpg", "evaluation_model_id": "u1", "evaluation_model_type": "unified", "evaluation_available": True, "unified_joint_correct": True},
            {"image_path": "b.jpg", "evaluation_model_id": "u1", "evaluation_model_type": "unified", "evaluation_available": True, "unified_joint_correct": False},
            {"image_path": "a.jpg", "evaluation_model_id": "l1", "evaluation_model_type": "legacy", "evaluation_available": True, "legacy_joint_correct": False},
            {"image_path": "b.jpg", "evaluation_model_id": "l1", "evaluation_model_type": "legacy", "evaluation_available": True, "legacy_joint_correct": True},
        ])

        app = LocalEvaluationApp.__new__(LocalEvaluationApp)
        app.resolve_report_path = lambda filename: Path(filename)
        with patch("pathlib.Path.open", return_value=io.StringIO(stream.getvalue())):
            page = app.report_rows("evaluation.csv", offset=0, limit=1)

        self.assertEqual(page["total"], 2)
        self.assertEqual(page["row_total"], 4)
        self.assertEqual(page["image_count"], 1)
        self.assertTrue(page["has_next"])
        self.assertEqual(
            [row["evaluation_model_id"] for row in page["rows"]],
            ["u1", "l1"],
        )

        with patch("pathlib.Path.open", return_value=io.StringIO(stream.getvalue())):
            correct = app.report_rows(
                "evaluation.csv", offset=0, limit=10, correctness="correct"
            )
        self.assertEqual(correct["total"], 2)
        self.assertEqual(correct["row_total"], 2)
        self.assertEqual(
            [row["evaluation_model_id"] for row in correct["rows"]],
            ["u1", "l1"],
        )

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

    def test_legacy_router_rejects_negative_without_running_expert(self):
        model = LegacyOnnxRuntimeModel.__new__(LegacyOnnxRuntimeModel)
        model.error = None
        model.root = Path("legacy")
        model.router_path = model.root / "router.onnx"
        model.category_classes = list(ROUTER_CLASSES)
        model.sessions = {}
        model._session = lambda key, path: key
        logits = np.zeros((1, len(ROUTER_CLASSES)), dtype=np.float32)
        logits[0, ROUTER_CLASSES.index("NON_FOOD")] = 10.0
        model._run = lambda session, inputs: logits

        with patch(
            "local_web.server.preprocess_image",
            return_value=np.zeros((3, 224, 224), dtype=np.float32),
        ):
            result = model.predict_batch([Path("negative.jpg")], [""])[0]

        self.assertEqual(result["food_status"], "NON_FOOD")
        self.assertAlmostEqual(result["food_probabilities"]["NON_FOOD"], 0.9997, places=4)
        self.assertIsNone(result["pipeline"])
        self.assertIsNone(result["oracle"])

    def test_csv_report_contract_contains_detailed_predictions(self):
        self.assertEqual(len(REPORT_FIELDS), len(set(REPORT_FIELDS)))
        for field in (
            "image_path",
            "truth_freshness",
            "evaluation_model_type",
            "evaluation_model_id",
            "evaluation_available",
            "unified_category_confidence",
            "legacy_pipeline_freshness",
            "legacy_oracle_correct",
            "legacy_router_probability_cu",
        ):
            self.assertIn(field, REPORT_FIELDS)


if __name__ == "__main__":
    unittest.main()
