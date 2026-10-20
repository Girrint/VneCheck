import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

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
from food_ai.data import preprocessing_metadata


def safe_bundle_name(value: str) -> str:
    name = "_".join(value.strip().replace("\\", " ").replace("/", " ").split())
    name = "".join(character for character in name if character.isalnum() or character in "-_.")
    if not name or name in {".", ".."}:
        raise ValueError("Tên legacy bundle không hợp lệ")
    return name


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_checkpoint(source: Path, destination: Path) -> dict:
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
    return {
        "classes": classes,
        "model_name": checkpoint.get("model_name", "mobilenet_v3_large"),
        "training_method": checkpoint.get("training_method", "unknown"),
        "run_name": checkpoint.get("run_name"),
        "epoch": checkpoint.get("epoch"),
    }


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
        default=None,
        help="Thư mục ghi router.onnx và freshness/*.onnx",
    )
    parser.add_argument(
        "--bundle-name",
        help="Tên bundle; output mặc định models/legacy/<bundle-name>",
    )
    parser.add_argument(
        "--display-name",
        help="Tên hiển thị trên web (mặc định lấy từ --bundle-name)",
    )
    args = parser.parse_args()

    if args.bundle_name and args.output_root:
        parser.error("Chỉ dùng một trong --bundle-name hoặc --output-root")
    bundle_id = safe_bundle_name(args.bundle_name) if args.bundle_name else None
    output_root = (
        MODEL_DIR / "legacy" / bundle_id
        if bundle_id
        else (args.output_root or MODEL_DIR)
    )

    sources = {"router": args.checkpoint_root / "router.pth"}
    sources.update({
        f"freshness/{category}": args.checkpoint_root / "freshness" / f"{category}.pth"
        for category in CATEGORIES
    })
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Thiếu checkpoint legacy: " + ", ".join(missing))

    checkpoints = {}
    artifacts = []
    for logical_name, source in sources.items():
        relative = Path(f"{logical_name}.onnx")
        destination = output_root / relative
        checkpoints[logical_name] = export_checkpoint(source, destination)
        artifacts.append({
            "name": logical_name,
            "path": relative.as_posix(),
            "size_bytes": destination.stat().st_size,
            "sha256": sha256(destination),
        })

    if bundle_id:
        manifest = {
            "bundle_version": 1,
            "bundle_id": bundle_id,
            "name": bundle_id,
            "display_name": args.display_name or args.bundle_name,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "preprocessing": preprocessing_metadata(),
            "checkpoints": checkpoints,
            "artifacts": artifacts,
        }
        manifest_path = output_root / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Legacy bundle: {manifest_path}")


if __name__ == "__main__":
    main()
