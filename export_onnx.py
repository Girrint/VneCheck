from pathlib import Path

import torch
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from food_ai.config import CATEGORIES, MODEL_DIR, ROUTER_MODEL, freshness_model
from food_ai.model import create_model


def export_checkpoint(source: Path, destination: Path) -> None:
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    classes = checkpoint["classes"]
    model = create_model(len(classes), checkpoint.get("model_name", "mobilenet_v3_large"))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch.randn(1, 3, 224, 224),
        destination,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )
    print(f"{destination}: {classes}")


def main() -> None:
    export_checkpoint(ROUTER_MODEL, MODEL_DIR / "router.onnx")

    for category in CATEGORIES:
        source = freshness_model(category)

        if source.exists():
            export_checkpoint(
                source,
                MODEL_DIR / "freshness" / f"{category}.onnx",
            )
        else:
            print(f"skip missing: {source}")


if __name__ == "__main__":
    main()
