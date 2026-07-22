import csv
import json
from datetime import datetime
from pathlib import Path


def create_run_directory(task, category, model_name, run_name=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = run_name or f"{task}_{category or 'all'}_{model_name}"
    safe_label = "".join(char if char.isalnum() or char in "-_" else "_" for char in label)
    run_dir = Path("runs") / f"{timestamp}_{safe_label}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path, payload):
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


class ExperimentTracker:
    FIELDNAMES = (
        "epoch",
        "train_loss",
        "train_accuracy",
        "valid_loss",
        "valid_accuracy",
        "learning_rate",
        "duration_seconds",
        "saved",
    )

    def __init__(self, run_dir, config, resume=False):
        self.run_dir = run_dir
        self.metrics_path = run_dir / "metrics.csv"
        if not resume or not self.metrics_path.exists():
            write_json(run_dir / "config.json", config)
            with self.metrics_path.open("w", newline="", encoding="utf-8") as file:
                csv.DictWriter(file, fieldnames=self.FIELDNAMES).writeheader()

    def log_epoch(self, row):
        with self.metrics_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.FIELDNAMES)
            writer.writerow({key: row[key] for key in self.FIELDNAMES})

    def save_summary(self, summary):
        write_json(self.run_dir / "summary.json", summary)


def classification_report(confusion, classes):
    rows = []
    total = int(confusion.sum().item())
    correct = int(confusion.diag().sum().item())
    for index, name in enumerate(classes):
        true_positive = int(confusion[index, index].item())
        false_positive = int(confusion[:, index].sum().item()) - true_positive
        false_negative = int(confusion[index, :].sum().item()) - true_positive
        support = int(confusion[index, :].sum().item())
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
            "class": name,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": support,
        })
    return {
        "accuracy": round(correct / total, 6) if total else 0.0,
        "macro_f1": round(sum(row["f1"] for row in rows) / len(rows), 6),
        "per_class": rows,
        "confusion_matrix": confusion.tolist(),
    }
