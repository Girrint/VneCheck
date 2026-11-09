from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_model(architecture: str):
    from torch import nn
    from torchvision import models

    if architecture == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
    elif architecture == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    return model


def metrics_from_predictions(targets: list[int], predictions: list[int]) -> dict:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, labels=[0, 1], zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "fresh_f1": float(f1[0]),
        "rotten_f1": float(f1[1]),
    }


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    scaler=None,
    amp_enabled: bool = True,
    gradient_clip_norm: float = 5.0,
) -> tuple[float, dict]:
    import torch

    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    targets: list[int] = []
    predictions: list[int] = []
    scaler_enabled = amp_enabled and device.type == "cuda"
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.amp.autocast("cuda", enabled=scaler_enabled):
            logits = model(images)
            loss = criterion(logits, labels)
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        total_loss += float(loss.item()) * labels.size(0)
        targets.extend(labels.detach().cpu().tolist())
        predictions.extend(logits.argmax(1).detach().cpu().tolist())
    return total_loss / len(loader.dataset), metrics_from_predictions(targets, predictions)


def train_one(dataset_root: Path, produce: str, architecture: str, config: dict, output_root: Path) -> Path:
    import torch
    from torch import nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder

    from .transforms import build_transforms

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for classifier training")
    seed_everything(config["seed"])
    train_set = ImageFolder(dataset_root / produce / "train", transform=build_transforms(config, True))
    val_set = ImageFolder(dataset_root / produce / "valid", transform=build_transforms(config, False))
    if train_set.classes != ["fresh", "rotten"] or val_set.classes != ["fresh", "rotten"]:
        raise ValueError(f"Expected fresh/rotten folders for {produce}; got {train_set.classes}")
    loader_args = {
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
        "pin_memory": True,
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    device = torch.device("cuda:0")
    model = create_model(architecture).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=config["scheduler_factor"], patience=config["scheduler_patience"]
    )
    run_dir = output_root / architecture / produce
    run_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    best_loss = float("inf")
    stale = 0
    history = []
    scaler = torch.amp.GradScaler("cuda", enabled=config["amp"])
    for epoch in range(1, config["max_epochs"] + 1):
        train_loss, train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            scaler,
            config["amp"],
            config["gradient_clip_norm"],
        )
        val_loss, val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            amp_enabled=config["amp"],
            gradient_clip_norm=config["gradient_clip_norm"],
        )
        scheduler.step(val_metrics["macro_f1"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
        )
        improved = val_metrics["macro_f1"] > best_f1 + 1e-4 or (
            abs(val_metrics["macro_f1"] - best_f1) <= 1e-4 and val_loss < best_loss
        )
        if improved:
            best_f1 = val_metrics["macro_f1"]
            best_loss = val_loss
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "architecture": architecture,
                    "produce": produce,
                    "classes": train_set.classes,
                    "config": config,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                run_dir / "best.pt",
            )
        else:
            stale += 1
        print(
            f"{architecture}/{produce} epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}"
        )
        if stale >= config["early_stopping_patience"]:
            break
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train scratch freshness classifiers")
    parser.add_argument("--data", type=Path, default=Path("datasets/classification"))
    parser.add_argument("--config", type=Path, default=Path("configs/classifiers.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/classification"))
    parser.add_argument("--produce", default="all")
    parser.add_argument("--architecture", default="all")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.max_epochs is not None:
        if args.max_epochs < 1:
            parser.error("--max-epochs must be at least 1")
        config["max_epochs"] = args.max_epochs
    if args.batch_size is not None:
        if args.batch_size < 1:
            parser.error("--batch-size must be at least 1")
        config["batch_size"] = args.batch_size
    if args.disable_amp:
        config["amp"] = False
    produce_names = sorted(path.name for path in args.data.iterdir() if path.is_dir())
    selected_produce = produce_names if args.produce == "all" else [args.produce]
    selected_architectures = (
        config["architectures"] if args.architecture == "all" else [args.architecture]
    )
    for architecture in selected_architectures:
        for produce in selected_produce:
            run_dir = train_one(args.data, produce, architecture, config, args.output)
            if not args.no_export:
                from .export import export_classifier

                exported = export_classifier(run_dir / "best.pt", run_dir / f"{produce}_{architecture}.onnx")
                print(f"Exported and verified ONNX: {exported}")


if __name__ == "__main__":
    main()
