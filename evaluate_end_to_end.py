import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from food_ai.config import (
    CATEGORIES,
    DATA_DIR,
    FRESHNESS_CLASSES,
    NEGATIVE_DATA_DIR,
    ROUTER_CLASSES,
    ROUTER_MODEL,
    freshness_model,
)
from food_ai.data import build_transforms, create_router_dataset
from food_ai.model import load_checkpoint

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def update_confusion(confusion, targets, predictions, classes):
    for target, prediction in zip(targets, predictions):
        confusion[classes[target]][classes[prediction]] += 1


def report(confusion, classes, macro_classes=None):
    rows = []
    for name in classes:
        true_positive = confusion[name][name]
        predicted = sum(confusion[target][name] for target in classes)
        actual = sum(confusion[name].values())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"class": name, "precision": precision, "recall": recall, "f1": f1, "support": actual})
    total = sum(row["support"] for row in rows)
    correct = sum(confusion[name][name] for name in classes)
    return {
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(
            row["f1"] for row in rows
            if row["class"] in set(macro_classes or classes)
        ) / len(macro_classes or classes),
        "per_class": rows,
        "confusion_matrix": [[confusion[target][prediction] for prediction in classes] for target in classes],
    }


def predict(model, images):
    with torch.inference_mode():
        return model(images).argmax(dim=1)


def main():
    parser = argparse.ArgumentParser(description="Đánh giá pipeline router -> freshness expert")
    parser.add_argument("--root", type=Path, default=DATA_DIR / "test")
    parser.add_argument(
        "--negative-root", type=Path, default=NEGATIVE_DATA_DIR / "test"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    _, eval_transform = build_transforms()
    router, router_classes = load_checkpoint(ROUTER_MODEL, DEVICE)
    router_dataset = create_router_dataset(
        args.root, args.negative_root, eval_transform, ROUTER_CLASSES
    )
    if list(router_dataset.classes) != list(router_classes):
        raise ValueError(f"Nhãn dataset/router không khớp: {router_dataset.classes} != {router_classes}")

    router_loader = DataLoader(router_dataset, batch_size=args.batch_size, shuffle=False)
    router_confusion = defaultdict(lambda: defaultdict(int))
    for images, labels in router_loader:
        images = images.to(DEVICE)
        predictions = predict(router, images).cpu().tolist()
        labels = labels.tolist()
        update_confusion(router_confusion, labels, predictions, router_classes)

    experts = {}
    for category in CATEGORIES:
        checkpoint = freshness_model(category)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Thiếu expert cho '{category}': {checkpoint}")
        experts[category] = load_checkpoint(checkpoint, DEVICE)

    oracle_confusion = defaultdict(lambda: defaultdict(int))
    end_to_end_confusion = defaultdict(lambda: defaultdict(int))
    pipeline_freshness_classes = FRESHNESS_CLASSES + ("REJECTED",)
    for category in CATEGORIES:
        dataset = ImageFolder(args.root / category, transform=eval_transform)
        expert, classes = experts[category]
        if dataset.classes != classes:
            raise ValueError(f"Nhãn freshness không khớp tại {category}: {dataset.classes} != {classes}")
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.tolist()
            oracle_predictions = predict(expert, images).cpu().tolist()
            target_indices = [
                FRESHNESS_CLASSES.index(dataset.classes[index]) for index in labels
            ]
            oracle_indices = [
                FRESHNESS_CLASSES.index(classes[index])
                for index in oracle_predictions
            ]
            update_confusion(
                oracle_confusion,
                target_indices,
                oracle_indices,
                FRESHNESS_CLASSES,
            )

            router_predictions = predict(router, images).cpu().tolist()
            for image, target, router_prediction in zip(
                images, target_indices, router_predictions
            ):
                predicted_category = router_classes[router_prediction]
                if predicted_category not in CATEGORIES:
                    update_confusion(
                        end_to_end_confusion,
                        [pipeline_freshness_classes.index(FRESHNESS_CLASSES[target])],
                        [pipeline_freshness_classes.index("REJECTED")],
                        pipeline_freshness_classes,
                    )
                    continue
                selected_expert, selected_classes = experts[predicted_category]
                prediction = int(predict(selected_expert, image.unsqueeze(0))[0].cpu())
                predicted_freshness = FRESHNESS_CLASSES.index(
                    selected_classes[prediction]
                )
                update_confusion(
                    end_to_end_confusion,
                    [pipeline_freshness_classes.index(FRESHNESS_CLASSES[target])],
                    [pipeline_freshness_classes.index(FRESHNESS_CLASSES[predicted_freshness])],
                    pipeline_freshness_classes,
                )

    result = {
        "root": str(args.root),
        "negative_root": str(args.negative_root),
        "device": str(DEVICE),
        "router": report(router_confusion, router_classes),
        "freshness_oracle_category": report(
            oracle_confusion, FRESHNESS_CLASSES
        ),
        "freshness_end_to_end": report(
            end_to_end_confusion,
            pipeline_freshness_classes,
            macro_classes=FRESHNESS_CLASSES,
        ),
        "note": "Router 7 lớp từ chối UNSUPPORTED/NON_FOOD; freshness_end_to_end tính REJECTED là sai trên ảnh SUPPORTED.",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
