import unittest

from vnecheck.classification.evaluate import calculate_metrics


class ClassificationEvaluationTests(unittest.TestCase):
    def test_metrics_for_perfect_predictions(self) -> None:
        metrics = calculate_metrics([0, 0, 1, 1], [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["rotten_recall"], 1.0)
        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(metrics["confusion_matrix"], [[2, 0], [0, 2]])

    def test_metrics_keep_both_classes_when_one_is_never_predicted(self) -> None:
        metrics = calculate_metrics([0, 1], [0, 0], [0.1, 0.4])
        self.assertEqual(metrics["rotten_recall"], 0.0)
        self.assertEqual(metrics["rotten_f1"], 0.0)
        self.assertEqual(metrics["confusion_matrix"], [[1, 0], [1, 0]])


if __name__ == "__main__":
    unittest.main()
