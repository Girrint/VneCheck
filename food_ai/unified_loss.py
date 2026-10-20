import torch
import torch.nn as nn

from food_ai.unified_data import IGNORE_INDEX


FOOD_CLASS_BALANCE_MODES = ("none", "sqrt_inverse", "inverse")


def compute_class_weights(class_counts, classes, mode="sqrt_inverse"):
    """Tạo class weight theo đúng class order và chuẩn hóa mean bằng 1."""
    if mode not in FOOD_CLASS_BALANCE_MODES:
        raise ValueError(
            f"Class balance không hợp lệ: {mode}; chọn {FOOD_CLASS_BALANCE_MODES}"
        )
    if mode == "none":
        return None

    missing = [name for name in classes if class_counts.get(name, 0) <= 0]
    if missing:
        raise ValueError(f"Không thể cân bằng class không có mẫu: {missing}")

    counts = torch.tensor(
        [class_counts[name] for name in classes], dtype=torch.float32
    )
    exponent = -0.5 if mode == "sqrt_inverse" else -1.0
    weights = counts.pow(exponent)
    return weights / weights.mean()


class MaskedMultiTaskLoss(nn.Module):
    def __init__(
        self,
        food_weight=1.0,
        category_weight=1.0,
        freshness_weight=1.0,
        label_smoothing=0.05,
        food_class_weights=None,
    ):
        super().__init__()
        self.weights = {
            "food": food_weight,
            "category": category_weight,
            "freshness": freshness_weight,
        }
        if food_class_weights is not None:
            food_class_weights = torch.as_tensor(
                food_class_weights, dtype=torch.float32
            )
        self.food_criterion = nn.CrossEntropyLoss(
            weight=food_class_weights,
            label_smoothing=label_smoothing,
        )
        self.masked_criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
        )

    def masked_loss(self, logits, targets):
        valid = targets != IGNORE_INDEX
        if not bool(valid.any()):
            return logits.sum() * 0.0
        return self.masked_criterion(logits[valid], targets[valid])

    def forward(self, outputs, targets):
        food_logits, category_logits, freshness_logits = outputs
        losses = {
            "food": self.food_criterion(food_logits, targets["food_status"]),
            "category": self.masked_loss(
                category_logits, targets["category"]
            ),
            "freshness": self.masked_loss(
                freshness_logits, targets["freshness"]
            ),
        }
        total = sum(self.weights[name] * loss for name, loss in losses.items())
        return total, losses
