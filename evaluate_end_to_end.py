import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from food_ai.config import DATA_DIR, ROUTER_MODEL, freshness_model
from food_ai.data import build_transforms
from food_ai.model import load_checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def update_confusion(confusion, targets, predictions, classes):
    for target, prediction in zip(targets, predictions):
        confusion[classes[target]][classes[prediction]] += 1


def report(confusion, classes):
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
        "macro_f1": sum(row["f1"] for row in rows) / len(rows),
        "per_class": rows,
        "confusion_matrix": [[confusion[target][prediction] for prediction in classes] for target in classes],
    }


def predict(model, images):
    with torch.inference_mode():
        return model(images).argmax(dim=1)


def main():
    parser = argparse.ArgumentParser(description="Đánh giá pipeline router -> freshness expert")
    parser.add_argument("--root", type=Path, default=DATA_DIR / "valid")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    _, eval_transform = build_transforms()
    router, router_classes = load_checkpoint(ROUTER_MODEL, DEVICE)
    router_dataset = ImageFolder(args.root, transform=eval_transform)
    if router_dataset.classes != router_classes:
        raise ValueError(f"Nhãn valid/router không khớp: {router_dataset.classes} != {router_classes}")

    router_loader = DataLoader(router_dataset, batch_size=args.batch_size, shuffle=False)
    router_confusion = defaultdict(lambda: defaultdict(int))
    for images, labels in router_loader:
        images = images.to(DEVICE)
        predictions = predict(router, images).cpu().tolist()
        labels = labels.tolist()
        update_confusion(router_confusion, labels, predictions, router_classes)

    experts = {}
    for category in router_classes:
        checkpoint = freshness_model(category)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Thiếu expert cho '{category}': {checkpoint}")
        experts[category] = load_checkpoint(checkpoint, DEVICE)

    freshness_classes = None
    oracle_confusion = defaultdict(lambda: defaultdict(int))
    end_to_end_confusion = defaultdict(lambda: defaultdict(int))
    for category in router_classes:
        dataset = ImageFolder(args.root / category, transform=eval_transform)
        expert, classes = experts[category]
        if freshness_classes is None:
            freshness_classes = classes
        if classes != freshness_classes or dataset.classes != classes:
            raise ValueError(f"Nhãn freshness không khớp tại {category}: {dataset.classes} != {classes}")
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.tolist()
            oracle_predictions = predict(expert, images).cpu().tolist()
            update_confusion(oracle_confusion, labels, oracle_predictions, classes)

            router_predictions = predict(router, images).cpu().tolist()
            for image, target, router_prediction in zip(images, labels, router_predictions):
                predicted_category = router_classes[router_prediction]
                selected_expert, selected_classes = experts[predicted_category]
                prediction = int(predict(selected_expert, image.unsqueeze(0))[0].cpu())
                update_confusion(
                    end_to_end_confusion,
                    [target],
                    [prediction],
                    selected_classes,
                )

    result = {
        "root": str(args.root),
        "device": str(DEVICE),
        "router": report(router_confusion, router_classes),
        "freshness_oracle_category": report(oracle_confusion, freshness_classes),
        "freshness_end_to_end": report(end_to_end_confusion, freshness_classes),
        "note": "freshness_end_to_end chọn expert bằng dự đoán router; metric này phản ánh pipeline thật.",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
