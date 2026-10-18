import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

from food_ai.config import (
    CATEGORY_LABELS,
    FRESHNESS_ANCHORS,
    FRESHNESS_LABELS,
    ROUTER_MODEL,
    freshness_model,
)
from food_ai.data import build_transforms
from food_ai.model import load_checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def predict_probabilities(model, classes, tensor):
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    return {name: probabilities[index].item() for index, name in enumerate(classes)}


def predict(image_path):
    if not ROUTER_MODEL.exists():
        raise FileNotFoundError(f"Thiếu model router: {ROUTER_MODEL}")

    _, eval_transform = build_transforms()
    with Image.open(image_path) as image:
        tensor = eval_transform(image).unsqueeze(0).to(DEVICE)

    router, category_classes = load_checkpoint(ROUTER_MODEL, DEVICE)
    category_probs = predict_probabilities(router, category_classes, tensor)
    category = max(category_probs, key=category_probs.get)

    expert_path = freshness_model(category)
    if not expert_path.exists():
        raise FileNotFoundError(
            f"Thiếu expert cho '{category}': {expert_path}"
        )

    expert, freshness_classes = load_checkpoint(expert_path, DEVICE)
    freshness_probs = predict_probabilities(expert, freshness_classes, tensor)
    freshness = max(freshness_probs, key=freshness_probs.get)
    score = sum(
        freshness_probs[name] * FRESHNESS_ANCHORS[name]
        for name in freshness_classes
    )

    return {
        "category": category,
        "category_label": CATEGORY_LABELS[category],
        "category_confidence": round(category_probs[category], 4),
        "freshness": freshness,
        "freshness_label": FRESHNESS_LABELS[freshness],
        "freshness_confidence": round(freshness_probs[freshness], 4),
        "estimated_freshness_score": round(score, 2),
        "scale": 10,
        "warning": "Chỉ số ước tính từ hình ảnh, không thay thế kiểm nghiệm an toàn.",
    }


def main():
    parser = argparse.ArgumentParser(description="Dự đoán độ tươi nhiều tầng")
    parser.add_argument("image", type=Path)
    args = parser.parse_args()
    print(json.dumps(predict(args.image), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
