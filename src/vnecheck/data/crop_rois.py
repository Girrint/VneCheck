from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def load_reviews(path: Path | None) -> dict[tuple[str, str, int], str]:
    if path is None or not path.is_file():
        return {}
    reviews = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            value = row.get("freshness", "").strip().casefold()
            if not value:
                continue
            if value not in {"fresh", "rotten"}:
                raise ValueError(f"Invalid freshness in {path}: {row}")
            reviews[(row["split"], row["image"], int(row["box_index"]))] = value
    return reviews


def crop_rois(
    detection_root: Path,
    output_root: Path,
    review_path: Path | None,
    padding: float = 0.03,
    min_size: int = 24,
) -> dict:
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError("Pillow is required to crop classification ROIs") from exc

    manifest_path = detection_root / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    reviews = load_reviews(review_path)
    records: list[dict] = []
    unresolved: list[dict] = []
    counts: Counter[tuple[str, str, str]] = Counter()
    skipped_small = 0

    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        split = row["split"]
        image_rel = Path(row["image"])
        box_index = int(row["box_index"])
        freshness = row["freshness"].strip().casefold()
        if not freshness:
            freshness = reviews.get((split, row["image"], box_index), "")
        if freshness not in {"fresh", "rotten"}:
            unresolved.append(row)
            continue

        image_path = detection_root / split / "images" / image_rel
        with Image.open(image_path) as source_image:
            image = ImageOps.exif_transpose(source_image).convert("RGB")
        width, height = image.size
        x_center = float(row["x_center"])
        y_center = float(row["y_center"])
        box_width = float(row["width"])
        box_height = float(row["height"])
        x1 = max(0, int((x_center - (0.5 + padding) * box_width) * width))
        y1 = max(0, int((y_center - (0.5 + padding) * box_height) * height))
        x2 = min(width, int((x_center + (0.5 + padding) * box_width) * width + 0.9999))
        y2 = min(height, int((y_center + (0.5 + padding) * box_height) * height + 0.9999))
        if x2 - x1 < min_size or y2 - y1 < min_size:
            skipped_small += 1
            continue

        produce = row["produce"]
        digest = hashlib.sha256(
            f"{split}|{row['image']}|{box_index}".encode("utf-8")
        ).hexdigest()[:16]
        crop_rel = Path(produce) / split / freshness / f"{produce}_{digest}.jpg"
        crop_path = output_root / crop_rel
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        image.crop((x1, y1, x2, y2)).save(crop_path, quality=95)
        output_row = dict(row)
        output_row.update(
            {
                "freshness": freshness,
                "crop": crop_rel.as_posix(),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )
        records.append(output_row)
        counts[(produce, split, freshness)] += 1

    output_root.mkdir(parents=True, exist_ok=True)
    if records:
        with (output_root / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    if unresolved:
        with (output_root / "unresolved.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(unresolved[0]))
            writer.writeheader()
            writer.writerows(unresolved)

    summary = {
        "crops": len(records),
        "unresolved": len(unresolved),
        "skipped_small": skipped_small,
        "counts": {"/".join(key): value for key, value in sorted(counts.items())},
    }
    (output_root / "crop_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop ground-truth ROIs for freshness classifiers")
    parser.add_argument("--detection", type=Path, default=Path("datasets/detection"))
    parser.add_argument("--output", type=Path, default=Path("datasets/classification"))
    parser.add_argument(
        "--review", type=Path, default=Path("datasets/detection/review_freshness.csv")
    )
    parser.add_argument("--padding", type=float, default=0.03)
    parser.add_argument("--min-size", type=int, default=24)
    args = parser.parse_args()
    result = crop_rois(args.detection, args.output, args.review, args.padding, args.min_size)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["unresolved"]:
        print("Fill the freshness column in the review CSV, then rerun to include all eligible ROIs.")


if __name__ == "__main__":
    main()
