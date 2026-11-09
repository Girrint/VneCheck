from __future__ import annotations

import argparse
import json
from pathlib import Path

from vnecheck.classification.evaluate import evaluate_all as evaluate_classifiers
from vnecheck.classification.export import export_classifier
from vnecheck.classification.train import train_one
from vnecheck.detection.evaluate import evaluate as evaluate_detector
from vnecheck.detection.train import train as train_detector


def validate_classification_dataset(root: Path, produce_names: list[str]) -> None:
    missing = []
    for produce in produce_names:
        for split in ("train", "valid", "test"):
            for freshness in ("fresh", "rotten"):
                directory = root / produce / split / freshness
                if not directory.is_dir() or not any(directory.iterdir()):
                    missing.append(str(directory))
    if missing:
        raise FileNotFoundError("Classification dataset is incomplete:\n" + "\n".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train YOLO11l and all 30 freshness classifiers, exporting every model to ONNX"
    )
    parser.add_argument("--skip-detector", action="store_true")
    parser.add_argument("--skip-classifiers", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--detector-name", default="yolo11l_640_e30_b4")
    parser.add_argument("--detection-data", type=Path, default=Path("datasets/detection/data.yaml"))
    parser.add_argument("--classification-data", type=Path, default=Path("datasets/classification"))
    parser.add_argument("--detector-config", type=Path, default=Path("configs/detector_yolo11l.json"))
    parser.add_argument("--classifier-config", type=Path, default=Path("configs/classifiers.json"))
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--evaluation-batch-size", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="Retrain models even when complete ONNX files exist")
    args = parser.parse_args()
    if args.evaluation_batch_size < 1:
        parser.error("--evaluation-batch-size must be at least 1")

    classifier_config = json.loads(args.classifier_config.read_text(encoding="utf-8"))
    produce_names = json.loads(Path("configs/classes.json").read_text(encoding="utf-8"))[
        "detection_names"
    ]
    validate_classification_dataset(args.classification_data, produce_names)

    if not args.skip_detector:
        detector_onnx = args.runs / "detection" / args.detector_name / "export" / "yolo11l_final.onnx"
        if detector_onnx.is_file() and not args.force:
            print(f"=== STAGE 1 ALREADY COMPLETE: {detector_onnx} ===", flush=True)
        else:
            print("=== STAGE 1: TRAIN AND EXPORT YOLO11L ===", flush=True)
            train_detector(
                args.detection_data,
                args.detector_config,
                args.runs / "detection",
                args.detector_name,
            )

    if not args.skip_classifiers:
        print("=== STAGE 2: TRAIN AND EXPORT 30 CLASSIFIERS ===", flush=True)
        total = len(classifier_config["architectures"]) * len(produce_names)
        completed = 0
        for architecture in classifier_config["architectures"]:
            for produce in produce_names:
                completed += 1
                print(
                    f"=== CLASSIFIER {completed}/{total}: {architecture}/{produce} ===",
                    flush=True,
                )
                run_dir = args.runs / "classification" / architecture / produce
                onnx_path = run_dir / f"{produce}_{architecture}.onnx"
                metadata_path = onnx_path.with_suffix(".json")
                complete = False
                if onnx_path.is_file() and metadata_path.is_file() and not args.force:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    complete = int(metadata.get("training_max_epochs", -1)) == int(
                        classifier_config["max_epochs"]
                    )
                if complete:
                    print(f"Already complete, skipping: {onnx_path}", flush=True)
                    continue
                run_dir = train_one(
                    args.classification_data,
                    produce,
                    architecture,
                    classifier_config,
                    args.runs / "classification",
                )
                export_classifier(run_dir / "best.pt", onnx_path)
                print(f"Exported and verified ONNX: {onnx_path}", flush=True)

    if not args.skip_evaluation:
        print("=== STAGE 3: FINAL TEST EVALUATION ===", flush=True)
        detector_weights = args.runs / "detection" / args.detector_name / "weights" / "best.pt"
        detector_output = args.reports / "detection_test_e30"
        detector_report = detector_output / "test_metrics.json"
        if args.force or not detector_report.is_file():
            if not detector_weights.is_file():
                raise FileNotFoundError(detector_weights)
            evaluate_detector(
                detector_weights,
                args.detection_data,
                detector_output,
                imgsz=640,
                batch=min(4, args.evaluation_batch_size),
            )
        else:
            print(f"Detector test report already exists, skipping: {detector_report}", flush=True)

        classifier_output = args.reports / "classification_test_e30"
        classifier_report = classifier_output / "summary.json"
        classifier_complete = False
        if classifier_report.is_file() and not args.force:
            summary = json.loads(classifier_report.read_text(encoding="utf-8"))
            classifier_complete = int(summary.get("models_evaluated", -1)) == (
                len(classifier_config["architectures"]) * len(produce_names)
            )
        if classifier_complete:
            print(f"Classifier test report already exists, skipping: {classifier_report}", flush=True)
        else:
            evaluate_classifiers(
                args.classification_data,
                args.runs / "classification",
                classifier_output,
                classifier_config["architectures"],
                produce_names,
                args.evaluation_batch_size,
                "cuda:0",
            )

    print("=== ALL REQUESTED TRAINING, EXPORTS, AND EVALUATION COMPLETED ===", flush=True)


if __name__ == "__main__":
    main()
