import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import torch

from food_ai.config import (
    CATEGORIES,
    FOOD_STATUS_CLASSES,
    FRESHNESS_CLASSES,
    IMAGE_SIZE,
    UNIFIED_MODEL,
    UNIFIED_ONNX_MODEL,
)
from food_ai.data import preprocessing_metadata
from food_ai.unified_model import load_unified_checkpoint


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_artifact_name(value):
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._")
    if not name:
        raise ValueError("Tên artifact không hợp lệ")
    return name


def main():
    parser = argparse.ArgumentParser(description="Export unified model sang ONNX")
    parser.add_argument("--checkpoint", type=Path, default=UNIFIED_MODEL)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--name",
        help="Tên artifact; tự ghi vào models/unified/exports/<name>.onnx",
    )
    parser.add_argument("--display-name", help="Tên hiển thị trong web đánh giá")
    args = parser.parse_args()

    if args.output and args.name:
        raise SystemExit("Chỉ dùng một trong --output hoặc --name")
    try:
        artifact_name = safe_artifact_name(
            args.name or (args.output.stem if args.output else UNIFIED_ONNX_MODEL.stem)
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    args.output = (
        Path("models/unified/exports") / f"{artifact_name}.onnx"
        if args.name
        else (args.output or UNIFIED_ONNX_MODEL)
    )

    model, checkpoint = load_unified_checkpoint(args.checkpoint, "cpu")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE),
        args.output,
        input_names=["image"],
        output_names=["food_logits", "category_logits", "freshness_logits"],
        dynamic_axes={
            "image": {0: "batch"},
            "food_logits": {0: "batch"},
            "category_logits": {0: "batch"},
            "freshness_logits": {0: "batch"},
        },
        opset_version=18,
        dynamo=False,
    )
    config = checkpoint.get("config", {})
    display_name = (
        args.display_name
        or config.get("run_name")
        or f"{checkpoint.get('model_name', 'unified')} · {artifact_name}"
    )
    manifest = {
        "manifest_version": 2,
        "artifact_id": artifact_name,
        "display_name": display_name,
        "architecture": checkpoint.get("model_name", "mobilenet_v3_small"),
        "training_method": checkpoint.get("training_method"),
        "run_name": config.get("run_name"),
        "epoch": checkpoint.get("epoch"),
        "best_epoch": checkpoint.get("best_epoch"),
        "selection_metric": checkpoint.get("selection_metric"),
        "food_gate_ready": bool(checkpoint.get("food_gate_ready", False)),
        "input": {"shape": [1, 3, IMAGE_SIZE, IMAGE_SIZE], "dtype": "float32"},
        "outputs": {
            "food_logits": FOOD_STATUS_CLASSES,
            "category_logits": CATEGORIES,
            "freshness_logits": FRESHNESS_CLASSES,
        },
        "preprocessing": preprocessing_metadata(),
        "checkpoint": {
            "path": args.checkpoint.as_posix(),
            "sha256": sha256(args.checkpoint),
        },
        "onnx": {"path": args.output.as_posix(), "sha256": sha256(args.output)},
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.output.resolve() == UNIFIED_ONNX_MODEL.resolve():
        legacy_manifest_path = args.output.parent / "manifest.json"
        legacy_manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(f"Đã export: {args.output}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
