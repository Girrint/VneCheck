from __future__ import annotations

import argparse
import csv
import json
import time
from collections.abc import Iterable
from pathlib import Path


def calculate_metrics(targets: list[int], predictions: list[int], rotten_scores: list[float]) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=[0, 1], zero_division=0
    )
    matrix = confusion_matrix(targets, predictions, labels=[0, 1])
    try:
        roc_auc = float(roc_auc_score(targets, rotten_scores))
    except ValueError:
        roc_auc = None
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "fresh_precision": float(precision[0]),
        "fresh_recall": float(recall[0]),
        "fresh_f1": float(f1[0]),
        "fresh_support": int(support[0]),
        "rotten_precision": float(precision[1]),
        "rotten_recall": float(recall[1]),
        "rotten_f1": float(f1[1]),
        "rotten_support": int(support[1]),
        "roc_auc": roc_auc,
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def save_confusion_matrix(matrix: list[list[int]], output: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    values = np.asarray(matrix, dtype=int)
    figure, axis = plt.subplots(figsize=(5.2, 4.6))
    image = axis.imshow(values, cmap="Greens")
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Fresh", "Rotten"],
        yticklabels=["Fresh", "Rotten"],
        xlabel="Predicted",
        ylabel="True",
        title=title,
    )
    threshold = values.max() / 2 if values.size else 0
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                str(values[row, column]),
                ha="center",
                va="center",
                color="white" if values[row, column] > threshold else "black",
                fontsize=13,
            )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def evaluate_one(
    checkpoint_path: Path,
    dataset_root: Path,
    output_dir: Path,
    batch_size: int = 8,
    device_name: str = "cuda:0",
) -> dict:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder

    from .train import create_model, seed_everything
    from .transforms import build_transforms

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    architecture = str(checkpoint["architecture"])
    produce = str(checkpoint["produce"])
    config = checkpoint["config"]
    seed_everything(int(config["seed"]))

    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    test_root = dataset_root / produce / "test"
    test_set = ImageFolder(test_root, transform=build_transforms(config, False))
    if test_set.classes != ["fresh", "rotten"]:
        raise ValueError(f"Expected fresh/rotten folders for {produce}; got {test_set.classes}")
    loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    model = create_model(architecture)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")

    targets: list[int] = []
    predictions: list[int] = []
    rotten_scores: list[float] = []
    image_paths: list[str] = []
    total_loss = 0.0
    inference_seconds = 0.0
    warmed_up = False
    sample_offset = 0
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if not warmed_up:
                model(images)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                warmed_up = True
            start = time.perf_counter()
            logits = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds += time.perf_counter() - start
            total_loss += float(criterion(logits, labels).item())
            probabilities = logits.softmax(dim=1)
            batch_predictions = probabilities.argmax(dim=1)
            batch_count = labels.size(0)
            targets.extend(labels.cpu().tolist())
            predictions.extend(batch_predictions.cpu().tolist())
            rotten_scores.extend(probabilities[:, 1].cpu().tolist())
            image_paths.extend(
                str(test_set.samples[sample_offset + index][0]) for index in range(batch_count)
            )
            sample_offset += batch_count

    metrics = calculate_metrics(targets, predictions, rotten_scores)
    metrics.update(
        {
            "architecture": architecture,
            "produce": produce,
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "validation_macro_f1": float(checkpoint["val_metrics"]["macro_f1"]),
            "test_samples": len(test_set),
            "test_loss": total_loss / len(test_set),
            "inference_ms_per_image": inference_seconds * 1000 / len(test_set),
            "device": str(device),
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["image", "true_id", "true_class", "predicted_id", "predicted_class", "p_rotten"]
        )
        for path, target, prediction, score in zip(
            image_paths, targets, predictions, rotten_scores, strict=True
        ):
            writer.writerow(
                [path, target, test_set.classes[target], prediction, test_set.classes[prediction], score]
            )
    with (output_dir / "confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/predicted", "fresh", "rotten"])
        writer.writerow(["fresh", *metrics["confusion_matrix"][0]])
        writer.writerow(["rotten", *metrics["confusion_matrix"][1]])
    save_confusion_matrix(
        metrics["confusion_matrix"],
        output_dir / "confusion_matrix.png",
        f"{architecture} / {produce} / test",
    )
    return metrics


def write_aggregate_reports(rows: list[dict], output_root: Path) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    scalar_rows = sorted(
        ({key: value for key, value in row.items() if key != "confusion_matrix"} for row in rows),
        key=lambda row: (str(row["architecture"]), str(row["produce"])),
    )
    fieldnames = list(scalar_rows[0]) if scalar_rows else []
    with (output_root / "per_model.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scalar_rows)
    metric_columns = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "fresh_f1",
        "rotten_f1",
        "rotten_recall",
        "roc_auc",
        "test_loss",
        "inference_ms_per_image",
    ]
    grouped: dict[str, list[dict]] = {}
    for row in scalar_rows:
        grouped.setdefault(str(row["architecture"]), []).append(row)
    architecture_summary = []
    for architecture, architecture_rows in grouped.items():
        summary_row: dict[str, str | int | float | None] = {"architecture": architecture}
        for metric in metric_columns:
            values = [float(row[metric]) for row in architecture_rows if row.get(metric) is not None]
            summary_row[metric] = sum(values) / len(values) if values else None
        summary_row["test_samples"] = sum(int(row["test_samples"]) for row in architecture_rows)
        architecture_summary.append(summary_row)
    with (output_root / "architecture_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["architecture", *metric_columns, "test_samples"]
        )
        writer.writeheader()
        writer.writerows(architecture_summary)
    summary = {
        "models_evaluated": len(rows),
        "architectures": architecture_summary,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def evaluate_all(
    dataset_root: Path,
    runs_root: Path,
    output_root: Path,
    architectures: Iterable[str],
    produce_names: Iterable[str],
    batch_size: int = 8,
    device_name: str = "cuda:0",
) -> dict:
    rows = []
    combinations = [(architecture, produce) for architecture in architectures for produce in produce_names]
    for index, (architecture, produce) in enumerate(combinations, start=1):
        print(f"=== TEST {index}/{len(combinations)}: {architecture}/{produce} ===", flush=True)
        checkpoint = runs_root / architecture / produce / "best.pt"
        destination = output_root / architecture / produce
        rows.append(evaluate_one(checkpoint, dataset_root, destination, batch_size, device_name))
    return write_aggregate_reports(rows, output_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate freshness classifiers on the frozen test split")
    parser.add_argument("--data", type=Path, default=Path("datasets/classification"))
    parser.add_argument("--runs", type=Path, default=Path("runs/classification"))
    parser.add_argument("--output", type=Path, default=Path("reports/classification_test_e30"))
    parser.add_argument("--config", type=Path, default=Path("configs/classifiers.json"))
    parser.add_argument("--classes", type=Path, default=Path("configs/classes.json"))
    parser.add_argument("--architecture", default="all")
    parser.add_argument("--produce", default="all")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    classes = json.loads(args.classes.read_text(encoding="utf-8"))["detection_names"]
    architectures = config["architectures"] if args.architecture == "all" else [args.architecture]
    produce_names = classes if args.produce == "all" else [args.produce]
    summary = evaluate_all(
        args.data,
        args.runs,
        args.output,
        architectures,
        produce_names,
        args.batch_size,
        args.device,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
