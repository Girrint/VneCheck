from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit and inference benchmark for the VneCheck E2E set.")
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/e2e")
    parser.add_argument(
        "--detector",
        type=Path,
        default=ROOT / "runs" / "detection" / "yolo11l_640_e30_b4" / "weights" / "best.pt",
    )
    parser.add_argument("--classification-root", type=Path, default=ROOT / "runs" / "classification")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "e2e_test_e30")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--padding", type=float, default=0.03)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def read_labels(path: Path) -> tuple[list[dict], list[str]]:
    boxes: list[dict] = []
    errors: list[str] = []
    if not path.is_file():
        return boxes, ["missing_label"]
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"line_{line_number}_field_count_{len(parts)}")
            continue
        try:
            class_id = int(parts[0])
            x, y, width, height = map(float, parts[1:])
        except ValueError:
            errors.append(f"line_{line_number}_parse_error")
            continue
        if not all(math.isfinite(value) for value in (x, y, width, height)):
            errors.append(f"line_{line_number}_non_finite")
            continue
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
            errors.append(f"line_{line_number}_out_of_range")
        boxes.append({"class_id": class_id, "xywhn": [x, y, width, height]})
    return boxes, errors


def xywhn_to_xyxy(values: list[float], image_width: int, image_height: int) -> list[float]:
    x, y, width, height = values
    half_width = width * image_width / 2
    half_height = height * image_height / 2
    center_x = x * image_width
    center_y = y * image_height
    return [center_x - half_width, center_y - half_height, center_x + half_width, center_y + half_height]


def box_iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def greedy_match(predictions: list[list[float]], targets: list[list[float]], threshold: float) -> list[tuple[int, int, float]]:
    candidates = sorted(
        (
            (box_iou(prediction, target), prediction_index, target_index)
            for prediction_index, prediction in enumerate(predictions)
            for target_index, target in enumerate(targets)
        ),
        reverse=True,
    )
    used_predictions: set[int] = set()
    used_targets: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, prediction_index, target_index in candidates:
        if iou < threshold:
            break
        if prediction_index in used_predictions or target_index in used_targets:
            continue
        used_predictions.add(prediction_index)
        used_targets.add(target_index)
        matches.append((prediction_index, target_index, iou))
    return matches


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    import torch
    from PIL import Image, ImageOps
    from ultralytics import YOLO

    from vnecheck.classification.train import create_model
    from vnecheck.classification.transforms import build_transforms

    dataset = args.dataset.resolve()
    image_dir = dataset / "images"
    label_dir = dataset / "labels"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    label_paths = {path.stem: path for path in label_dir.glob("*.txt")}
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    if not args.detector.is_file():
        raise FileNotFoundError(args.detector)

    audit_rows: list[dict] = []
    ground_truth: dict[str, list[dict]] = {}
    class_counts: Counter[int] = Counter()
    widths: list[int] = []
    heights: list[int] = []
    corrupt_images: list[str] = []
    total_label_errors = 0
    for image_path in image_paths:
        try:
            with Image.open(image_path) as image:
                image = ImageOps.exif_transpose(image)
                width, height = image.size
        except Exception as exc:  # pragma: no cover - real data guard
            corrupt_images.append(f"{image_path.name}: {exc}")
            continue
        labels, errors = read_labels(label_paths.get(image_path.stem, label_dir / f"{image_path.stem}.txt"))
        for label in labels:
            label["xyxy"] = xywhn_to_xyxy(label["xywhn"], width, height)
            class_counts[label["class_id"]] += 1
        ground_truth[image_path.name] = labels
        widths.append(width)
        heights.append(height)
        total_label_errors += len(errors)
        audit_rows.append(
            {
                "image": image_path.name,
                "width": width,
                "height": height,
                "aspect_ratio": width / height,
                "gt_boxes": len(labels),
                "gt_class_ids": " ".join(map(str, sorted({label['class_id'] for label in labels}))),
                "label_errors": "|".join(errors),
            }
        )

    model = YOLO(str(args.detector.resolve()))
    model_names = {int(key): str(value) for key, value in model.names.items()}
    if sorted(model_names) != list(range(10)):
        raise ValueError(f"Expected 10 detector classes, got {model_names}")

    detector_start = time.perf_counter()
    detector_results = model.predict(
        source=str(image_dir),
        imgsz=640,
        batch=1,
        conf=args.confidence,
        iou=args.nms_iou,
        device=args.device,
        half=args.device.startswith("cuda") and torch.cuda.is_available(),
        verbose=False,
        stream=True,
    )
    prediction_rows: list[dict] = []
    detector_speed = defaultdict(float)
    detector_images = 0
    total_matches = 0
    total_gt = 0
    total_predictions = 0
    matched_ious: list[float] = []
    matched_images = 0
    predictions_by_produce: defaultdict[str, list[int]] = defaultdict(list)
    for result in detector_results:
        image_path = Path(result.path)
        detector_images += 1
        for key, value in result.speed.items():
            detector_speed[key] += float(value)
        gt = ground_truth[image_path.name]
        gt_boxes = [item["xyxy"] for item in gt]
        boxes = result.boxes
        predicted_boxes = boxes.xyxy.detach().cpu().tolist() if boxes is not None else []
        confidences = boxes.conf.detach().cpu().tolist() if boxes is not None else []
        class_ids = [int(value) for value in boxes.cls.detach().cpu().tolist()] if boxes is not None else []
        matches = greedy_match(predicted_boxes, gt_boxes, args.match_iou)
        match_by_prediction = {prediction_index: (target_index, iou) for prediction_index, target_index, iou in matches}
        total_matches += len(matches)
        total_gt += len(gt_boxes)
        total_predictions += len(predicted_boxes)
        matched_ious.extend(iou for _, _, iou in matches)
        matched_images += int(bool(matches))
        for prediction_index, (box, confidence, class_id) in enumerate(zip(predicted_boxes, confidences, class_ids, strict=True)):
            produce = model_names[class_id]
            matched = match_by_prediction.get(prediction_index)
            row = {
                "image": image_path.name,
                "object_index": prediction_index + 1,
                "detector_class_id": class_id,
                "produce": produce,
                "detector_confidence": confidence,
                "x1": box[0],
                "y1": box[1],
                "x2": box[2],
                "y2": box[3],
                "matched_gt_index": matched[0] + 1 if matched else "",
                "matched_gt_class_id": gt[matched[0]]["class_id"] if matched else "",
                "matched_iou": matched[1] if matched else "",
            }
            prediction_rows.append(row)
            predictions_by_produce[produce].append(len(prediction_rows) - 1)
    detector_wall_seconds = time.perf_counter() - detector_start

    architectures = ["mobilenet_v3_large", "resnet18", "efficientnet_b0"]
    architecture_inference_seconds = Counter()
    architecture_preprocess_seconds = Counter()
    architecture_model_load_seconds = Counter()
    architecture_roi_counts = Counter()
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    for produce, record_indices in sorted(predictions_by_produce.items()):
        reference_checkpoint = args.classification_root / architectures[0] / produce / "best.pt"
        checkpoint = torch.load(reference_checkpoint, map_location="cpu", weights_only=False)
        transform = build_transforms(checkpoint["config"], False)
        preprocess_start = time.perf_counter()
        tensors = []
        for record_index in record_indices:
            record = prediction_rows[record_index]
            with Image.open(image_dir / record["image"]) as source_image:
                source_image = ImageOps.exif_transpose(source_image).convert("RGB")
                width = record["x2"] - record["x1"]
                height = record["y2"] - record["y1"]
                crop_box = (
                    max(0, math.floor(record["x1"] - width * args.padding)),
                    max(0, math.floor(record["y1"] - height * args.padding)),
                    min(source_image.width, math.ceil(record["x2"] + width * args.padding)),
                    min(source_image.height, math.ceil(record["y2"] + height * args.padding)),
                )
                tensors.append(transform(source_image.crop(crop_box)))
        preprocess_seconds = time.perf_counter() - preprocess_start
        for architecture in architectures:
            checkpoint_path = args.classification_root / architecture / produce / "best.pt"
            load_start = time.perf_counter()
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            classifier = create_model(architecture)
            classifier.load_state_dict(checkpoint["model"])
            classifier.to(device).eval()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            architecture_model_load_seconds[architecture] += time.perf_counter() - load_start
            probabilities: list[list[float]] = []
            inference_start = time.perf_counter()
            with torch.inference_mode():
                for offset in range(0, len(tensors), args.batch_size):
                    batch = torch.stack(tensors[offset : offset + args.batch_size]).to(device, non_blocking=True)
                    batch_probabilities = classifier(batch).softmax(dim=1)
                    probabilities.extend(batch_probabilities.detach().cpu().tolist())
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds = time.perf_counter() - inference_start
            architecture_inference_seconds[architecture] += inference_seconds
            architecture_preprocess_seconds[architecture] += preprocess_seconds
            architecture_roi_counts[architecture] += len(tensors)
            for record_index, probability in zip(record_indices, probabilities, strict=True):
                predicted_id = int(probability[1] >= probability[0])
                prediction_rows[record_index][f"{architecture}_freshness"] = "rotten" if predicted_id else "fresh"
                prediction_rows[record_index][f"{architecture}_p_fresh"] = probability[0]
                prediction_rows[record_index][f"{architecture}_p_rotten"] = probability[1]
            del classifier
            if device.type == "cuda":
                torch.cuda.empty_cache()

    agreement_count = 0
    for row in prediction_rows:
        states = [row[f"{architecture}_freshness"] for architecture in architectures]
        row["classifier_agreement"] = len(set(states)) == 1
        row["majority_freshness"] = Counter(states).most_common(1)[0][0]
        agreement_count += int(row["classifier_agreement"])

    detector_precision = total_matches / total_predictions if total_predictions else 0.0
    detector_recall = total_matches / total_gt if total_gt else 0.0
    detector_f1 = (
        2 * detector_precision * detector_recall / (detector_precision + detector_recall)
        if detector_precision + detector_recall
        else 0.0
    )
    class_distribution = [
        {
            "label_class_id": class_id,
            "boxes": count,
            "compatible_with_model_id_range": 0 <= class_id < len(model_names),
            "project_model_name_if_assumed": model_names.get(class_id, ""),
        }
        for class_id, count in sorted(class_counts.items())
    ]
    predicted_class_counts = Counter(row["produce"] for row in prediction_rows)
    architecture_summary = {}
    for architecture in architectures:
        count = architecture_roi_counts[architecture]
        freshness_counts = Counter(row[f"{architecture}_freshness"] for row in prediction_rows)
        architecture_summary[architecture] = {
            "roi_count": count,
            "fresh_predictions": freshness_counts["fresh"],
            "rotten_predictions": freshness_counts["rotten"],
            "fresh_rate": freshness_counts["fresh"] / count if count else None,
            "rotten_rate": freshness_counts["rotten"] / count if count else None,
            "model_load_seconds": architecture_model_load_seconds[architecture],
            "preprocess_ms_per_roi": architecture_preprocess_seconds[architecture] * 1000 / count if count else None,
            "inference_ms_per_roi": architecture_inference_seconds[architecture] * 1000 / count if count else None,
        }

    summary = {
        "status": "inference_complete_ground_truth_schema_incomplete",
        "limitations": [
            "datasets/e2e has label IDs 0-14 but no data.yaml/class-name mapping; the e30 detector has IDs 0-9.",
            "Visual spot checks show the test label IDs do not follow configs/classes.json, so class-aware metrics are intentionally not computed.",
            "Labels contain bounding boxes and class IDs only; Fresh/Rotten ground truth is absent, so freshness accuracy and full E2E accuracy cannot be computed.",
        ],
        "settings": {
            "detector": str(args.detector.resolve()),
            "classification_root": str(args.classification_root.resolve()),
            "device": str(device),
            "confidence_threshold": args.confidence,
            "nms_iou_threshold": args.nms_iou,
            "matching_iou_threshold": args.match_iou,
            "roi_padding_fraction": args.padding,
            "image_size": 640,
            "classifier_input_size": 224,
        },
        "dataset": {
            "root": str(dataset),
            "images": len(image_paths),
            "readable_images": len(ground_truth),
            "labels": len(label_paths),
            "ground_truth_boxes": total_gt,
            "label_id_min": min(class_counts) if class_counts else None,
            "label_id_max": max(class_counts) if class_counts else None,
            "boxes_with_ids_0_to_9": sum(count for class_id, count in class_counts.items() if 0 <= class_id < 10),
            "boxes_with_ids_outside_model_range": sum(count for class_id, count in class_counts.items() if class_id < 0 or class_id >= 10),
            "images_with_label_errors": sum(bool(row["label_errors"]) for row in audit_rows),
            "label_error_count": total_label_errors,
            "corrupt_images": corrupt_images,
            "width_min": min(widths) if widths else None,
            "width_max": max(widths) if widths else None,
            "height_min": min(heights) if heights else None,
            "height_max": max(heights) if heights else None,
            "boxes_per_image_mean": total_gt / len(ground_truth) if ground_truth else None,
            "boxes_per_image_median": statistics.median(row["gt_boxes"] for row in audit_rows) if audit_rows else None,
        },
        "detector_class_agnostic_iou_0_5": {
            "predicted_boxes": total_predictions,
            "matched_boxes": total_matches,
            "false_positive_boxes": total_predictions - total_matches,
            "missed_ground_truth_boxes": total_gt - total_matches,
            "precision": detector_precision,
            "recall": detector_recall,
            "f1": detector_f1,
            "mean_matched_iou": statistics.mean(matched_ious) if matched_ious else None,
            "median_matched_iou": statistics.median(matched_ious) if matched_ious else None,
            "images_with_at_least_one_match": matched_images,
            "predicted_class_counts": dict(sorted(predicted_class_counts.items())),
            "wall_seconds": detector_wall_seconds,
            "speed_ms_per_image": {
                key: value / detector_images for key, value in detector_speed.items()
            },
        },
        "classifiers_inference_only": architecture_summary,
        "classifier_three_model_agreement": {
            "agreed_rois": agreement_count,
            "total_rois": len(prediction_rows),
            "agreement_rate": agreement_count / len(prediction_rows) if prediction_rows else None,
            "majority_fresh": sum(row["majority_freshness"] == "fresh" for row in prediction_rows),
            "majority_rotten": sum(row["majority_freshness"] == "rotten" for row in prediction_rows),
        },
    }

    columns = [
        "image", "object_index", "detector_class_id", "produce", "detector_confidence",
        "x1", "y1", "x2", "y2", "matched_gt_index", "matched_gt_class_id", "matched_iou",
    ]
    for architecture in architectures:
        columns.extend([f"{architecture}_freshness", f"{architecture}_p_fresh", f"{architecture}_p_rotten"])
    columns.extend(["classifier_agreement", "majority_freshness"])
    write_csv(output / "predictions.csv", prediction_rows, columns)
    write_csv(output / "image_audit.csv", audit_rows, list(audit_rows[0]))
    write_csv(output / "label_class_distribution.csv", class_distribution, list(class_distribution[0]))
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
