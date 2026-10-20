import argparse
import csv
import json
import os
import random
import shutil
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from food_ai.config import (
    CATEGORIES,
    FOOD_STATUS_CLASSES,
    FRESHNESS_CLASSES,
    UNIFIED_MANIFEST,
    UNIFIED_MODEL,
)
from food_ai.data import preprocessing_metadata
from food_ai.experiment import classification_report, create_run_directory
from food_ai.unified_data import IGNORE_INDEX, create_unified_loaders
from food_ai.unified_loss import (
    FOOD_CLASS_BALANCE_MODES,
    MaskedMultiTaskLoss,
    compute_class_weights,
)
from food_ai.unified_model import UNIFIED_MODEL_NAMES, create_unified_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = DEVICE.type == "cuda"


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Train unified food model")
    parser.add_argument("--manifest", type=Path, default=UNIFIED_MANIFEST)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--workers", type=int, default=min(4, os.cpu_count() or 1)
    )
    parser.add_argument(
        "--model", choices=UNIFIED_MODEL_NAMES, default="mobilenet_v3_small"
    )
    parser.add_argument(
        "--method", choices=("fine_tune", "head_only"), default="fine_tune"
    )
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--food-loss-weight", type=float, default=1.0)
    parser.add_argument("--category-loss-weight", type=float, default=1.0)
    parser.add_argument("--freshness-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--food-class-balance",
        choices=FOOD_CLASS_BALANCE_MODES,
        default="sqrt_inverse",
        help="Cân bằng CrossEntropy của food head theo số mẫu train",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("food_macro_f1", "valid_loss"),
        default="food_macro_f1",
        help="Metric chọn best checkpoint và early stopping",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name")
    return parser.parse_args()


def validate_args(args):
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise SystemExit("Epoch/batch phải dương; workers không được âm")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise SystemExit("Learning rate phải dương; weight decay không âm")
    if not 0 <= args.label_smoothing < 1:
        raise SystemExit("Label smoothing phải nằm trong [0, 1)")
    weights = (
        args.food_loss_weight,
        args.category_loss_weight,
        args.freshness_loss_weight,
    )
    if any(weight < 0 for weight in weights) or sum(weights) == 0:
        raise SystemExit("Loss weight phải không âm và có ít nhất một weight > 0")


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def move_targets(targets):
    return {
        name: tensor.to(DEVICE, non_blocking=USE_CUDA)
        for name, tensor in targets.items()
    }


def update_confusion(confusion, targets, predictions, ignore=False):
    if ignore:
        valid = targets != IGNORE_INDEX
        targets = targets[valid]
        predictions = predictions[valid]
    if targets.numel() == 0:
        return
    size = confusion.shape[0]
    encoded = (targets * size + predictions).detach().cpu()
    confusion += torch.bincount(encoded, minlength=size**2).reshape(size, size)


def run_epoch(model, loader, criterion, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    if training and getattr(model, "training_method", None) == "head_only":
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
    loss_sum = 0.0
    sample_count = 0
    component_sums = {"food": 0.0, "category": 0.0, "freshness": 0.0}
    confusions = {
        "food": torch.zeros(
            len(FOOD_STATUS_CLASSES), len(FOOD_STATUS_CLASSES), dtype=torch.int64
        ),
        "category": torch.zeros(
            len(CATEGORIES), len(CATEGORIES), dtype=torch.int64
        ),
        "freshness": torch.zeros(
            len(FRESHNESS_CLASSES), len(FRESHNESS_CLASSES), dtype=torch.int64
        ),
    }
    context = nullcontext() if training else torch.inference_mode()

    with context:
        for images, targets in loader:
            images = images.to(DEVICE, non_blocking=USE_CUDA)
            targets = move_targets(targets)
            if training:
                optimizer.zero_grad(set_to_none=True)

            amp = (
                torch.autocast("cuda", dtype=torch.float16)
                if USE_CUDA
                else nullcontext()
            )
            with amp:
                outputs = model(images)
                loss, components = criterion(outputs, targets)

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            batch_size = images.shape[0]
            loss_sum += float(loss.detach()) * batch_size
            sample_count += batch_size
            for name, value in components.items():
                component_sums[name] += float(value.detach()) * batch_size

            food_logits, category_logits, freshness_logits = outputs
            update_confusion(
                confusions["food"],
                targets["food_status"],
                food_logits.argmax(1),
            )
            update_confusion(
                confusions["category"],
                targets["category"],
                category_logits.argmax(1),
                ignore=True,
            )
            update_confusion(
                confusions["freshness"],
                targets["freshness"],
                freshness_logits.argmax(1),
                ignore=True,
            )

    reports = {
        "food": classification_report(
            confusions["food"], FOOD_STATUS_CLASSES
        ),
        "category": classification_report(confusions["category"], CATEGORIES),
        "freshness": classification_report(
            confusions["freshness"], FRESHNESS_CLASSES
        ),
    }
    return {
        "loss": loss_sum / sample_count,
        "component_losses": {
            name: value / sample_count for name, value in component_sums.items()
        },
        "reports": reports,
    }


def food_gate_ready(datasets):
    required = set(FOOD_STATUS_CLASSES)
    return all(required.issubset(dataset.status_counts) for dataset in datasets.values())


def metric_columns(prefix, report):
    values = {f"{prefix}_macro_f1": report["macro_f1"]}
    for row in report["per_class"]:
        class_name = row["class"].lower()
        for metric in ("precision", "recall", "f1"):
            values[f"{prefix}_{class_name}_{metric}"] = row[metric]
    return values


def main():
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if USE_CUDA:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    try:
        loaders, datasets = create_unified_loaders(
            args.manifest, args.batch_size, args.workers
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Dữ liệu unified chưa sẵn sàng: {error}") from error

    gate_ready = food_gate_ready(datasets)
    if not gate_ready:
        print(
            "CẢNH BÁO: manifest chưa có đủ SUPPORTED/UNSUPPORTED/NON_FOOD "
            "trong cả ba split. Food gate sẽ được đánh dấu chưa sẵn sàng."
        )
    effective_food_weight = args.food_loss_weight if gate_ready else 0.0
    effective_selection_metric = (
        args.selection_metric if gate_ready else "valid_loss"
    )
    food_class_weights = None
    if gate_ready:
        food_class_weights = compute_class_weights(
            datasets["train"].status_counts,
            FOOD_STATUS_CLASSES,
            args.food_class_balance,
        )
    serialized_food_class_weights = (
        None
        if food_class_weights is None
        else {
            name: round(float(weight), 6)
            for name, weight in zip(FOOD_STATUS_CLASSES, food_class_weights)
        }
    )

    run_dir = create_run_directory("unified", None, args.model, args.run_name)
    config = {
        **vars(args),
        "manifest": str(args.manifest),
        "food_status_classes": FOOD_STATUS_CLASSES,
        "category_classes": CATEGORIES,
        "freshness_classes": FRESHNESS_CLASSES,
        "food_gate_ready": gate_ready,
        "effective_food_loss_weight": effective_food_weight,
        "effective_selection_metric": effective_selection_metric,
        "food_class_weights": serialized_food_class_weights,
        "split_sizes": {name: len(dataset) for name, dataset in datasets.items()},
        "status_counts": {
            name: dict(dataset.status_counts) for name, dataset in datasets.items()
        },
        "preprocessing": preprocessing_metadata(),
        "device": torch.cuda.get_device_name(0) if USE_CUDA else "CPU",
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    model = create_unified_model(
        args.model, pretrained=True, method=args.method
    ).to(DEVICE)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    criterion = MaskedMultiTaskLoss(
        food_weight=effective_food_weight,
        category_weight=args.category_loss_weight,
        freshness_weight=args.freshness_loss_weight,
        label_smoothing=args.label_smoothing,
        food_class_weights=food_class_weights,
    ).to(DEVICE)
    optimizer = optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )
    scaler = torch.amp.GradScaler("cuda", enabled=USE_CUDA)

    metrics_path = run_dir / "metrics.csv"
    food_metric_fields = tuple(
        f"valid_food_{status.lower()}_{metric}"
        for status in FOOD_STATUS_CLASSES
        for metric in ("precision", "recall", "f1")
    )
    fieldnames = (
        "epoch",
        "train_loss",
        "valid_loss",
        "valid_food_accuracy",
        "valid_food_macro_f1",
        *food_metric_fields,
        "valid_category_accuracy",
        "valid_freshness_accuracy",
        "learning_rate",
        "duration_seconds",
        "saved",
    )
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=fieldnames).writeheader()

    best_score = float("-inf")
    best_valid_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    best_path = run_dir / "best_model.pth"
    last_path = run_dir / "last_checkpoint.pth"

    def payload(epoch):
        return {
            "state_dict": model.state_dict(),
            "model_name": args.model,
            "training_method": args.method,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "best_valid_loss": best_valid_loss,
            "selection_metric": effective_selection_metric,
            "food_gate_ready": gate_ready,
            "food_status_classes": FOOD_STATUS_CLASSES,
            "category_classes": CATEGORIES,
            "freshness_classes": FRESHNESS_CLASSES,
            "preprocessing": preprocessing_metadata(),
            "config": config,
        }

    print(f"Run: {run_dir}")
    print(f"Model: {args.model} | device: {config['device']}")
    print(f"Split sizes: {config['split_sizes']}")
    print(f"Food class weights: {serialized_food_class_weights}")
    print(f"Best checkpoint metric: {effective_selection_metric}")

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_result = run_epoch(
            model, loaders["train"], criterion, optimizer, scaler
        )
        valid_result = run_epoch(model, loaders["valid"], criterion)
        valid_loss = valid_result["loss"]
        food_report = valid_result["reports"]["food"]
        score = (
            food_report["macro_f1"]
            if effective_selection_metric == "food_macro_f1"
            else -valid_loss
        )
        improved = score > best_score or (
            score == best_score and valid_loss < best_valid_loss
        )
        if improved:
            best_score = score
            best_valid_loss = valid_loss
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1

        duration = time.perf_counter() - started
        row = {
            "epoch": epoch,
            "train_loss": round(train_result["loss"], 6),
            "valid_loss": round(valid_loss, 6),
            "valid_food_accuracy": food_report["accuracy"],
            **metric_columns("valid_food", food_report),
            "valid_category_accuracy": valid_result["reports"]["category"]["accuracy"],
            "valid_freshness_accuracy": valid_result["reports"]["freshness"]["accuracy"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "duration_seconds": round(duration, 3),
            "saved": improved,
        }
        with metrics_path.open("a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=fieldnames).writerow(row)

        scheduler.step(valid_loss)
        atomic_save(payload(epoch), last_path)
        if improved:
            atomic_save(payload(epoch), best_path)
            UNIFIED_MODEL.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_path, UNIFIED_MODEL)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train={train_result['loss']:.4f} | valid={valid_loss:.4f} | "
            f"food={100 * row['valid_food_accuracy']:.2f}%/"
            f"F1={100 * row['valid_food_macro_f1']:.2f}% | "
            f"cat={100 * row['valid_category_accuracy']:.2f}% | "
            f"fresh={100 * row['valid_freshness_accuracy']:.2f}%"
            + (" | saved" if improved else ""),
            flush=True,
        )
        if stale_epochs >= 3:
            print("Early stopping: 3 epoch không cải thiện.")
            break

    checkpoint = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    test_result = run_epoch(model, loaders["test"], criterion)
    summary = {
        "best_epoch": best_epoch,
        "selection_metric": effective_selection_metric,
        "best_score": round(best_score, 6),
        "best_valid_loss": round(best_valid_loss, 6),
        "food_gate_ready": gate_ready,
        "test_loss": round(test_result["loss"], 6),
        "test": test_result,
        "deployed_checkpoint": str(UNIFIED_MODEL),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Đã lưu unified checkpoint: {UNIFIED_MODEL}")
    print(f"Kết quả: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
