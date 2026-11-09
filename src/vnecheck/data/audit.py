from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from .common import SPLITS, list_images, load_class_config, parse_filename_identity, read_yolo_label, source_group_id


def audit_dataset(data_root: Path) -> dict:
    config = load_class_config()
    raw_to_detection = {int(key): int(value) for key, value in config["raw_to_detection"].items()}
    raw_freshness = {int(key): value for key, value in config["raw_default_freshness"].items()}

    report: dict = {
        "data_root": str(data_root.resolve()),
        "splits": {},
        "raw_class_counts": {},
        "detection_class_counts": {},
        "freshness_counts": {},
        "errors": [],
        "cross_split_groups": {},
    }
    raw_counts: Counter[int] = Counter()
    detection_counts: Counter[int] = Counter()
    freshness_counts: Counter[str] = Counter()
    groups: defaultdict[str, set[str]] = defaultdict(set)

    for split in SPLITS:
        image_dir = data_root / split / "images"
        label_dir = data_root / split / "labels"
        images = list_images(image_dir) if image_dir.is_dir() else []
        labels = sorted(label_dir.rglob("*.txt")) if label_dir.is_dir() else []
        image_keys = {image.relative_to(image_dir).with_suffix(""): image for image in images}
        label_keys = {label.relative_to(label_dir).with_suffix(""): label for label in labels}
        missing = sorted(str(key) for key in image_keys.keys() - label_keys.keys())
        orphan = sorted(str(key) for key in label_keys.keys() - image_keys.keys())
        box_count = 0
        empty_labels = 0
        unknown_freshness = 0

        for key in sorted(image_keys.keys() & label_keys.keys(), key=str):
            image = image_keys[key]
            label = label_keys[key]
            groups[source_group_id(image.name)].add(split)
            try:
                boxes = read_yolo_label(label)
            except ValueError as exc:
                report["errors"].append(str(exc))
                continue
            if not boxes:
                empty_labels += 1
            explicit_freshness, explicit_produce = parse_filename_identity(image.name)
            unique_raw_ids = {box.class_id for box in boxes}
            for box in boxes:
                box_count += 1
                raw_counts[box.class_id] += 1
                if box.class_id not in raw_to_detection:
                    report["errors"].append(f"{label}: unmapped raw class {box.class_id}")
                    continue
                detection_id = raw_to_detection[box.class_id]
                detection_counts[detection_id] += 1
                expected_produce = config["detection_names"][detection_id]
                freshness = raw_freshness.get(box.class_id)
                if (
                    explicit_freshness
                    and explicit_produce == expected_produce
                    and len(unique_raw_ids) == 1
                ):
                    freshness = explicit_freshness
                if freshness is None:
                    unknown_freshness += 1
                    freshness_counts["unknown"] += 1
                else:
                    freshness_counts[freshness] += 1

        report["splits"][split] = {
            "images": len(images),
            "labels": len(labels),
            "boxes": box_count,
            "empty_labels": empty_labels,
            "missing_labels": len(missing),
            "orphan_labels": len(orphan),
            "unknown_freshness_boxes": unknown_freshness,
            "missing_label_examples": missing[:20],
            "orphan_label_examples": orphan[:20],
        }

    report["raw_class_counts"] = {str(key): raw_counts[key] for key in sorted(raw_counts)}
    report["detection_class_counts"] = {
        config["detection_names"][key]: detection_counts[key] for key in sorted(detection_counts)
    }
    report["freshness_counts"] = dict(sorted(freshness_counts.items()))
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        overlaps = sorted(group for group, splits in groups.items() if {left, right} <= splits)
        report["cross_split_groups"][f"{left}-{right}"] = {
            "count": len(overlaps),
            "examples": overlaps[:20],
        }
    return report


def write_report(report: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "class_counts.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "boxes"])
        writer.writerows(report["detection_class_counts"].items())


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the raw VneCheck YOLO dataset")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("reports/data_audit"))
    args = parser.parse_args()
    report = audit_dataset(args.data)
    write_report(report, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
