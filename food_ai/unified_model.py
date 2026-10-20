import torch
import torch.nn as nn
from torchvision import models

from food_ai.config import CATEGORIES, FOOD_STATUS_CLASSES, FRESHNESS_CLASSES


UNIFIED_MODEL_NAMES = (
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "resnet18",
)


class UnifiedFoodModel(nn.Module):
    """Một backbone với ba head: food status, category và freshness."""

    def __init__(self, model_name="mobilenet_v3_small", pretrained=False):
        super().__init__()
        if model_name == "mobilenet_v3_small":
            weights = (
                models.MobileNet_V3_Small_Weights.DEFAULT
                if pretrained
                else None
            )
            base = models.mobilenet_v3_small(weights=weights)
        elif model_name == "mobilenet_v3_large":
            weights = (
                models.MobileNet_V3_Large_Weights.DEFAULT
                if pretrained
                else None
            )
            base = models.mobilenet_v3_large(weights=weights)
        elif model_name == "efficientnet_b0":
            weights = (
                models.EfficientNet_B0_Weights.DEFAULT
                if pretrained
                else None
            )
            base = models.efficientnet_b0(weights=weights)
        elif model_name == "resnet18":
            weights = (
                models.ResNet18_Weights.DEFAULT
                if pretrained
                else None
            )
            base = models.resnet18(weights=weights)
        else:
            raise ValueError(f"Unified model không hỗ trợ: {model_name}")

        self.model_name = model_name
        if model_name == "resnet18":
            self.features = nn.Sequential(*list(base.children())[:-2])
            self.avgpool = base.avgpool
            self.shared_classifier = nn.Identity()
            feature_size = base.fc.in_features
        else:
            self.features = base.features
            self.avgpool = base.avgpool
            self.shared_classifier = base.classifier[:-1]
            feature_size = base.classifier[-1].in_features
        self.food_head = nn.Linear(feature_size, len(FOOD_STATUS_CLASSES))
        self.category_head = nn.Linear(feature_size, len(CATEGORIES))
        self.freshness_head = nn.Linear(feature_size, len(FRESHNESS_CLASSES))

    def forward(self, images):
        features = self.features(images)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        features = self.shared_classifier(features)
        return (
            self.food_head(features),
            self.category_head(features),
            self.freshness_head(features),
        )

    def head_parameters(self):
        yield from self.food_head.parameters()
        yield from self.category_head.parameters()
        yield from self.freshness_head.parameters()


def create_unified_model(
    model_name="mobilenet_v3_small", pretrained=False, method="fine_tune"
):
    model = UnifiedFoodModel(model_name, pretrained=pretrained)
    if method == "head_only":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.head_parameters():
            parameter.requires_grad = True
    elif method != "fine_tune":
        raise ValueError(f"Training method không hợp lệ: {method}")
    model.training_method = method
    return model


def load_unified_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = create_unified_model(
        checkpoint.get("model_name", "mobilenet_v3_small"),
        pretrained=False,
        method=checkpoint.get("training_method", "fine_tune"),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint
