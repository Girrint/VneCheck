from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path

from .common import (
    SPLITS,
    deterministic_split,
    list_images,
    load_class_config,
    parse_filename_identity,
    read_yolo_label,
    source_group_id,
)


def materialize(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def prepare(source_root: Path, output_root: Path) -> dict:
    config = load_class_config()
    names = config["detection_names"]
    raw_to_detection = {int(key): int(value) for key, value in config["raw_to_detection"].items()}
    raw_freshness = {int(key): value for key, value in config["raw_default_freshness"].items()}

    if output_root.resolve() == source_root.resolve() or output_root.resolve().is_relative_to(source_root.resolve()):
        raise ValueError("Output dataset must not be inside the immutable source dataset")

    manifest_rows: list[dict] = []
    review_rows: list[dict] = []
    split_counts: Counter[str] = Counter()

    for original_split in SPLITS:
        image_dir = source_root / original_split / "images"
        label_dir = source_root / original_split / "labels"
        for image in list_images(image_dir):
            relative = image.relative_to(image_dir)
            label = label_dir / relative.with_suffix(".txt")
            boxes = read_yolo_label(label)
            group_id = source_group_id(image.name)
            target_split = deterministic_split(group_id)
            target_image = output_root / target_split / "images" / relative
            target_label = output_root / target_split / "labels" / relative.with_suffix(".txt")
            if target_image.exists() or target_label.exists():
                raise FileExistsError(f"Output collision for {relative}")
            materialize(image, target_image)
            target_label.parent.mkdir(parents=True, exist_ok=True)
            target_label.write_text(
                "\n".join(box.serialize(raw_to_detection[box.class_id]) for box in boxes)
                + ("\n" if boxes else ""),
                encoding="utf-8",
            )

            explicit_freshness, explicit_produce = parse_filename_identity(image.name)
            unique_raw_ids = {box.class_id for box in boxes}
            for box_index, box in enumerate(boxes):
                detection_id = raw_to_detection[box.class_id]
                produce = names[detection_id]
                freshness = raw_freshness.get(box.class_id)
                if explicit_freshness and explicit_produce == produce and len(unique_raw_ids) == 1:
                    freshness = explicit_freshness
                row = {
                    "split": target_split,
                    "original_split": original_split,
                    "group_id": group_id,
                    "image": relative.as_posix(),
                    "box_index": box_index,
                    "raw_class_id": box.class_id,
                    "detection_class_id": detection_id,
                    "produce": produce,
                    "freshness": freshness or "",
                    "x_center": box.x_center,
                    "y_center": box.y_center,
                    "width": box.width,
                    "height": box.height,
                }
                manifest_rows.append(row)
                if freshness is None:
                    review_rows.append(row)
            split_counts[target_split] += 1

    output_root.mkdir(parents=True, exist_ok=True)
    yaml_lines = [
        f"path: {output_root.resolve().as_posix()}",
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        f"nc: {len(names)}",
        "names:",
        *[f"  {index}: {name}" for index, name in enumerate(names)],
        "",
    ]
    (output_root / "data.yaml").write_text("\n".join(yaml_lines), encoding="utf-8")

    fields = list(manifest_rows[0])
    with (output_root / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)
    with (output_root / "review_freshness.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(review_rows)

    summary = {
        "source": str(source_root.resolve()),
        "output": str(output_root.resolve()),
        "images_per_split": dict(split_counts),
        "boxes": len(manifest_rows),
        "freshness_review_required": len(review_rows),
        "model_classes": names,
    }
    (output_root / "prepare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the leakage-safe 10-class YOLO dataset")
    parser.add_argument("--source", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("datasets/detection"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
