import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from food_ai.config import DATA_DIR, ROUTER_MODEL
from food_ai.data import build_transforms
from food_ai.model import load_checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, loader, threshold):
    total = correct = accepted = accepted_correct = 0
    confidence_sum = entropy_sum = 0.0
    confidence_bins = {"0.0-0.5": [0, 0], "0.5-0.8": [0, 0], "0.8-0.9": [0, 0], "0.9-1.0": [0, 0]}

    with torch.inference_mode():
        for images, labels in loader:
            probabilities = torch.softmax(model(images.to(DEVICE)), dim=1)
            confidence, predictions = probabilities.max(dim=1)
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
            correct_mask = predictions.cpu() == labels
            accepted_mask = confidence.cpu() >= threshold

            batch_size = labels.size(0)
            total += batch_size
            correct += int(correct_mask.sum())
            accepted += int(accepted_mask.sum())
            accepted_correct += int((accepted_mask & correct_mask).sum())
            confidence_sum += float(confidence.sum())
            entropy_sum += float(entropy.sum())

            for value, is_correct in zip(confidence.cpu().tolist(), correct_mask.tolist()):
                if value < 0.5:
                    bucket = "0.0-0.5"
                elif value < 0.8:
                    bucket = "0.5-0.8"
                elif value < 0.9:
                    bucket = "0.8-0.9"
                else:
                    bucket = "0.9-1.0"
                confidence_bins[bucket][0] += int(is_correct)
                confidence_bins[bucket][1] += 1

    rejected = total - accepted
    return {
        "threshold": threshold,
        "samples": total,
        "accuracy": correct / total if total else 0.0,
        "mean_confidence": confidence_sum / total if total else 0.0,
        "mean_entropy": entropy_sum / total if total else 0.0,
        "accepted_samples": accepted,
        "coverage": accepted / total if total else 0.0,
        "accepted_accuracy": accepted_correct / accepted if accepted else 0.0,
        "rejected_samples": rejected,
        "confidence_bins": {
            name: {
                "correct": values[0],
                "samples": values[1],
                "accuracy": values[0] / values[1] if values[1] else 0.0,
            }
            for name, values in confidence_bins.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Đánh giá confidence và entropy của router")
    parser.add_argument("--root", type=Path, default=DATA_DIR / "valid")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        raise SystemExit("--threshold phải nằm trong [0, 1]")

    _, eval_transform = build_transforms()
    model, classes = load_checkpoint(ROUTER_MODEL, DEVICE)
    dataset = ImageFolder(args.root, transform=eval_transform)
    if dataset.classes != classes:
        raise ValueError(f"Nhãn valid/router không khớp: {dataset.classes} != {classes}")

    result = evaluate(model, DataLoader(dataset, batch_size=args.batch_size, shuffle=False), args.threshold)
    result.update({"root": str(args.root), "device": str(DEVICE), "classes": classes})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
