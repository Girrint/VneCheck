import torch
import torch.nn as nn

from food_ai.unified_data import IGNORE_INDEX


class MaskedMultiTaskLoss(nn.Module):
    def __init__(
        self,
        food_weight=1.0,
        category_weight=1.0,
        freshness_weight=1.0,
        label_smoothing=0.05,
    ):
        super().__init__()
        self.weights = {
            "food": food_weight,
            "category": category_weight,
            "freshness": freshness_weight,
        }
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
        )

    def masked_loss(self, logits, targets):
        valid = targets != IGNORE_INDEX
        if not bool(valid.any()):
            return logits.sum() * 0.0
        return self.criterion(logits[valid], targets[valid])

    def forward(self, outputs, targets):
        food_logits, category_logits, freshness_logits = outputs
        losses = {
            "food": self.criterion(food_logits, targets["food_status"]),
            "category": self.masked_loss(
                category_logits, targets["category"]
            ),
            "freshness": self.masked_loss(
                freshness_logits, targets["freshness"]
            ),
        }
        total = sum(self.weights[name] * loss for name, loss in losses.items())
        return total, losses
