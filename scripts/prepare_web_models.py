from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCE = [
    "apple",
    "banana",
    "bellpepper",
    "carrot",
    "cucumber",
    "mango",
    "orange",
    "potato",
    "strawberry",
    "tomato",
]
ARCHITECTURES = {
    "mobilenet_v3_large": "MobileNetV3-Large",
    "resnet18": "ResNet-18",
    "efficientnet_b0": "EfficientNet-B0",
}


def is_complete_classifier(model_path: Path, expected_epochs: int) -> bool:
    metadata_path = model_path.with_suffix(".json")
    if not model_path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return int(metadata.get("training_max_epochs", -1)) == expected_epochs


def transfer(source: Path, destination: Path, mode: str, force: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        source_stat = source.stat()
        destination_stat = destination.stat()
        unchanged = (
            destination_stat.st_size == source_stat.st_size
            and destination_stat.st_mtime_ns == source_stat.st_mtime_ns
        )
        if unchanged and not force:
            return
        destination.unlink()
    if mode == "copy":
        shutil.copy2(source, destination)
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def sync_pair(source_onnx: Path, destination_onnx: Path, mode: str, force: bool) -> bool:
    if not source_onnx.is_file():
        return False
    transfer(source_onnx, destination_onnx, mode, force)
    source_metadata = source_onnx.with_suffix(".json")
    if source_metadata.is_file():
        transfer(source_metadata, destination_onnx.with_suffix(".json"), "copy", force)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expose completed VneCheck ONNX artifacts to the local benchmark web app"
    )
    parser.add_argument(
        "--detector-run",
        default="yolo11l_640_e30_b4",
        help="Directory name below <runs-root>/detection",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=PROJECT_ROOT / "runs",
        help="Root containing detection and classification run directories",
    )
    parser.add_argument(
        "--mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="Hardlink avoids duplicating large ONNX files on the same drive",
    )
    parser.add_argument("--force", action="store_true", help="Replace existing web model files")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail unless all 31 ONNX models are present",
    )
    args = parser.parse_args()

    classifier_config = json.loads(
        (PROJECT_ROOT / "configs" / "classifiers.json").read_text(encoding="utf-8")
    )
    expected_epochs = int(classifier_config["max_epochs"])

    web_models = PROJECT_ROOT / "web" / "models"
    detector_source_dir = (
        args.runs_root / "detection" / args.detector_run / "export"
    )
    detector_source = detector_source_dir / "yolo11l_final.onnx"
    detector_destination = web_models / "detector" / "yolo11l_final.onnx"
    detector_available = sync_pair(
        detector_source, detector_destination, args.mode, args.force
    )
    detector_metadata = detector_source_dir / "export_info.json"
    if detector_available and detector_metadata.is_file():
        transfer(
            detector_metadata,
            web_models / "detector" / "export_info.json",
            "copy",
            args.force,
        )

    classifiers: dict[str, dict] = {}
    classifier_count = 0
    for architecture, display_name in ARCHITECTURES.items():
        available: list[str] = []
        for produce in PRODUCE:
            filename = f"{produce}_{architecture}.onnx"
            source = (
                args.runs_root
                / "classification"
                / architecture
                / produce
                / filename
            )
            destination = web_models / "classifiers" / architecture / produce / filename
            if is_complete_classifier(source, expected_epochs) and sync_pair(
                source, destination, args.mode, args.force
            ):
                available.append(produce)
                classifier_count += 1
        classifiers[architecture] = {
            "display_name": display_name,
            "available": available,
        }

    manifest = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "detector": {
            "available": detector_available,
            "path": "models/detector/yolo11l_final.onnx",
            "metadata": "models/detector/export_info.json",
            "input_size": 640,
            "classes": PRODUCE,
        },
        "classifiers": classifiers,
    }
    manifest_path = web_models / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(manifest_path)

    print(f"Detector: {int(detector_available)}/1")
    print(f"Classifiers: {classifier_count}/30")
    print(f"Manifest: {manifest_path}")
    if args.require_complete and (not detector_available or classifier_count != 30):
        raise SystemExit("Model set is incomplete; expected 1 detector and 30 classifiers")


if __name__ == "__main__":
    main()
