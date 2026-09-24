import torch
import torch.nn as nn
from torchvision import models

MODEL_NAMES = (
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "resnet18",
)
TRAINING_METHODS = ("fine_tune", "head_only")


def create_model(
    num_classes: int,
    model_name: str = "mobilenet_v3_large",
    pretrained: bool = False,
) -> nn.Module:
    if model_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif model_name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Model không hỗ trợ: {model_name}")
    return model


def classifier_parameters(model, model_name):
    if model_name.startswith("mobilenet") or model_name.startswith("efficientnet"):
        return model.classifier.parameters()
    return model.fc.parameters()


def configure_training_method(model, model_name, method):
    if method not in TRAINING_METHODS:
        raise ValueError(f"Phương pháp không hỗ trợ: {method}")
    if method == "head_only":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in classifier_parameters(model, model_name):
            parameter.requires_grad = True
    model.training_method = method
    return model


def trainable_parameters(model):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def load_checkpoint(path, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    classes = checkpoint["classes"]
    model_name = checkpoint.get("model_name", "mobilenet_v3_large")
    model = create_model(len(classes), model_name=model_name)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, classes

