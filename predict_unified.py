import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

from food_ai.config import (
    ALLOWED_FRESHNESS_BY_CATEGORY,
    CATEGORIES,
    CATEGORY_LABELS,
    FOOD_STATUS_CLASSES,
    FRESHNESS_ANCHORS,
    FRESHNESS_CLASSES,
    FRESHNESS_LABELS,
    UNIFIED_MODEL,
)
from food_ai.data import build_transforms
from food_ai.unified_model import load_unified_checkpoint


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def probabilities(logits, classes):
    values = torch.softmax(logits, dim=1)[0]
    return {name: float(values[index]) for index, name in enumerate(classes)}


def masked_freshness_probabilities(logits, category):
    allowed = set(ALLOWED_FRESHNESS_BY_CATEGORY[category])
    masked = logits.clone()
    for index, name in enumerate(FRESHNESS_CLASSES):
        if name not in allowed:
            masked[:, index] = -torch.inf
    return probabilities(masked, FRESHNESS_CLASSES)


def predict_unified(
    image_path,
    checkpoint_path=UNIFIED_MODEL,
    food_threshold=0.70,
    category_threshold=0.50,
    freshness_threshold=0.50,
):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Thiếu unified checkpoint: {checkpoint_path}. "
            "Chạy train_unified.py trước."
        )

    model, checkpoint = load_unified_checkpoint(checkpoint_path, DEVICE)
    _, transform = build_transforms()
    with Image.open(image_path) as image:
        tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        food_logits, category_logits, freshness_logits = model(tensor)

    food_probs = probabilities(food_logits, FOOD_STATUS_CLASSES)
    gate_ready = bool(checkpoint.get("food_gate_ready", False))
    food_status = max(food_probs, key=food_probs.get)
    food_confidence = food_probs[food_status]

    if gate_ready and (
        food_status != "SUPPORTED" or food_confidence < food_threshold
    ):
        return {
            "accepted": False,
            "food_gate_ready": True,
            "food_status": food_status,
            "food_confidence": round(food_confidence, 4),
            "reason": "Ảnh không thuộc phạm vi thực phẩm được hỗ trợ.",
        }

    category_probs = probabilities(category_logits, CATEGORIES)
    category = max(category_probs, key=category_probs.get)
    category_confidence = category_probs[category]
    freshness_probs = masked_freshness_probabilities(
        freshness_logits, category
    )
    freshness = max(freshness_probs, key=freshness_probs.get)
    freshness_confidence = freshness_probs[freshness]
    accepted = (
        category_confidence >= category_threshold
        and freshness_confidence >= freshness_threshold
    )
    score = sum(
        freshness_probs[name] * FRESHNESS_ANCHORS[name]
        for name in ALLOWED_FRESHNESS_BY_CATEGORY[category]
    )
    return {
        "accepted": accepted,
        "food_gate_ready": gate_ready,
        "food_status": food_status if gate_ready else "NOT_TRAINED",
        "food_confidence": round(food_confidence, 4) if gate_ready else None,
        "category": category,
        "category_label": CATEGORY_LABELS[category],
        "category_confidence": round(category_confidence, 4),
        "freshness": freshness,
        "freshness_label": FRESHNESS_LABELS[freshness],
        "freshness_confidence": round(freshness_confidence, 4),
        "estimated_freshness_score": round(score, 2),
        "scale": 10,
        "warning": (
            "Food gate chưa được train bằng negative data. "
            if not gate_ready
            else ""
        ) + "Chỉ số không thay thế kiểm nghiệm an toàn thực phẩm.",
    }


def main():
    parser = argparse.ArgumentParser(description="Unified food inference")
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=UNIFIED_MODEL)
    parser.add_argument("--food-threshold", type=float, default=0.70)
    parser.add_argument("--category-threshold", type=float, default=0.50)
    parser.add_argument("--freshness-threshold", type=float, default=0.50)
    args = parser.parse_args()
    result = predict_unified(
        args.image,
        args.checkpoint,
        args.food_threshold,
        args.category_threshold,
        args.freshness_threshold,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
