"""Đóng băng 6 model legacy vào thư mục version riêng để so sánh/rollback."""

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from food_ai.config import CATEGORIES, MODEL_DIR, ROUTER_MODEL, freshness_model
from food_ai.data import preprocessing_metadata


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_pairs():
    yield "router", ROUTER_MODEL
    yield "router", MODEL_DIR / "router.onnx"
    for category in CATEGORIES:
        yield f"freshness/{category}", freshness_model(category)
        yield f"freshness/{category}", MODEL_DIR / "freshness" / f"{category}.onnx"


def snapshot_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Snapshot đã tồn tại: {destination}")
    shutil.copy2(source, destination)
    return "copy"


def main():
    parser = argparse.ArgumentParser(description="Snapshot model legacy")
    parser.add_argument("--name", default="original")
    args = parser.parse_args()
    destination_root = MODEL_DIR / "legacy" / args.name
    if destination_root.exists() and any(destination_root.iterdir()):
        raise SystemExit(f"Snapshot không rỗng: {destination_root}")

    entries = []
    checkpoint_metadata = {}
    for logical_name, source in artifact_pairs():
        if not source.is_file():
            raise SystemExit(f"Thiếu legacy artifact: {source}")
        relative = source.relative_to(MODEL_DIR)
        destination = destination_root / relative
        mode = snapshot_file(source, destination)
        entry = {
            "name": logical_name,
            "source": source.as_posix(),
            "snapshot": destination.as_posix(),
            "size_bytes": destination.stat().st_size,
            "sha256": sha256(destination),
            "mode": mode,
        }
        entries.append(entry)
        if source.suffix == ".pth":
            checkpoint = torch.load(source, map_location="cpu", weights_only=False)
            checkpoint_metadata[logical_name] = {
                "classes": checkpoint.get("classes"),
                "model_name": checkpoint.get("model_name", "mobilenet_v3_large"),
                "training_method": checkpoint.get("training_method", "unknown"),
            }

    manifest = {
        "snapshot_version": 1,
        "name": args.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Artifacts có trước unified pipeline và trước clean-split retraining.",
        "preprocessing": preprocessing_metadata(),
        "checkpoints": checkpoint_metadata,
        "artifacts": entries,
    }
    manifest_path = destination_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Đã snapshot {len(entries)} artifacts tại {destination_root}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
