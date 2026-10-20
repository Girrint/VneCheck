import argparse
from pathlib import Path

import torch
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from food_ai.config import (
    CATEGORIES,
    IMAGE_SIZE,
    MODEL_DIR,
)
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
        torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE),
        destination,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )
    print(f"{destination}: {classes}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export router và 5 freshness expert legacy sang ONNX"
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=MODEL_DIR,
        help="Thư mục chứa router.pth và freshness/*.pth (mặc định: models)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=MODEL_DIR,
        help="Thư mục ghi router.onnx và freshness/*.onnx (mặc định: models)",
    )
    args = parser.parse_args()

    router_checkpoint = args.checkpoint_root / "router.pth"
    if not router_checkpoint.is_file():
        raise FileNotFoundError(f"Thiếu checkpoint router: {router_checkpoint}")
    export_checkpoint(router_checkpoint, args.output_root / "router.onnx")

    for category in CATEGORIES:
        source = args.checkpoint_root / "freshness" / f"{category}.pth"

        if source.exists():
            export_checkpoint(
                source,
                args.output_root / "freshness" / f"{category}.onnx",
            )
        else:
            print(f"skip missing: {source}")


if __name__ == "__main__":
    main()
