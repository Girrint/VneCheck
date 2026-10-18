import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch

from food_ai.config import (
    CATEGORIES,
    IMAGE_SIZE,
    MODEL_DIR,
    ROUTER_MODEL,
    freshness_model,
)
from food_ai.data import preprocessing_metadata


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_metadata(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "model_name": checkpoint.get("model_name", "mobilenet_v3_large"),
        "training_method": checkpoint.get("training_method", "unknown"),
        "classes": checkpoint["classes"],
    }


def add_artifact(manifest, kind, name, checkpoint, onnx):
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    item = {
        "kind": kind,
        "name": name,
        "checkpoint": {
            "path": checkpoint.as_posix(),
            "sha256": sha256(checkpoint),
        },
        "onnx": None,
        **checkpoint_metadata(checkpoint),
    }
    if onnx.exists():
        item["onnx"] = {
            "path": onnx.as_posix(),
            "sha256": sha256(onnx),
        }
        sidecar = Path(f"{onnx}.data")
        if sidecar.exists():
            item["onnx_data"] = {
                "path": sidecar.as_posix(),
                "sha256": sha256(sidecar),
            }
    manifest["models"].append(item)


def copy_assets(manifest, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    for item in manifest["models"]:
        for artifact in (
            item["checkpoint"],
            item["onnx"],
            item.get("onnx_data"),
        ):
            if artifact is None:
                continue
            source = Path(artifact["path"])
            target = destination / source.relative_to(MODEL_DIR)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def main():
    parser = argparse.ArgumentParser(description="Đóng gói model và tạo manifest checksum")
    parser.add_argument("--output", type=Path, default=MODEL_DIR / "manifest.json")
    parser.add_argument("--assets-destination", type=Path)
    parser.add_argument("--require-onnx", action="store_true")
    args = parser.parse_args()

    manifest = {
        "manifest_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "shape": [1, 3, IMAGE_SIZE, IMAGE_SIZE],
            "dtype": "float32",
        },
        "preprocessing": preprocessing_metadata(),
        "models": [],
    }
    add_artifact(manifest, "router", "router", ROUTER_MODEL, MODEL_DIR / "router.onnx")
    for category in CATEGORIES:
        add_artifact(
            manifest,
            "freshness",
            category,
            freshness_model(category),
            MODEL_DIR / "freshness" / f"{category}.onnx",
        )

    missing_onnx = [item["name"] for item in manifest["models"] if item["onnx"] is None]
    if args.require_onnx and missing_onnx:
        raise SystemExit(f"Thiếu ONNX: {', '.join(missing_onnx)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.assets_destination:
        copy_assets(manifest, args.assets_destination)
    print(f"Đã tạo manifest: {args.output}")
    if missing_onnx:
        print(f"ONNX chưa có: {', '.join(missing_onnx)}")


if __name__ == "__main__":
    main()
