"""Tạo manifest cho unified model mà không mở nội dung ảnh."""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

from torchvision.datasets.folder import IMG_EXTENSIONS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from food_ai.config import (
    CATEGORIES,
    DATA_DIR,
    FOOD_STATUS_CLASSES,
    FRESHNESS_CLASSES,
    UNIFIED_MANIFEST,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


SPLITS = ("train", "valid", "test")
NEGATIVE_STATUSES = tuple(
    status for status in FOOD_STATUS_CLASSES if status != "SUPPORTED"
)


def portable_path(path):
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(resolved)


def images_under(root):
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS
    )


def positive_rows(root):
    rows = []
    for split in SPLITS:
        split_root = root / split
        for path in images_under(split_root):
            parts = path.relative_to(split_root).parts
            if len(parts) < 4:
                raise ValueError(f"Đường dẫn positive không hợp lệ: {path}")
            category, freshness = parts[:2]
            if category not in CATEGORIES or freshness not in FRESHNESS_CLASSES:
                raise ValueError(f"Nhãn không hợp lệ trong đường dẫn: {path}")
            rows.append({
                "path": portable_path(path),
                "split": split,
                "food_status": "SUPPORTED",
                "category": category,
                "freshness": freshness,
            })
    return rows


def negative_rows(root):
    rows = []
    if root is None:
        return rows
    for split in SPLITS:
        for status in NEGATIVE_STATUSES:
            for path in images_under(root / split / status):
                rows.append({
                    "path": portable_path(path),
                    "split": split,
                    "food_status": status,
                    "category": "",
                    "freshness": "",
                })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Tạo manifest unified model")
    parser.add_argument("--positive-root", type=Path, default=DATA_DIR)
    parser.add_argument("--negative-root", type=Path)
    parser.add_argument("--output", type=Path, default=UNIFIED_MANIFEST)
    args = parser.parse_args()

    rows = positive_rows(args.positive_root)
    rows.extend(negative_rows(args.negative_root))
    rows.sort(key=lambda row: (row["split"], row["food_status"], row["path"]))
    if not rows:
        raise SystemExit("Không tìm thấy ảnh để tạo manifest")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "path",
                "split",
                "food_status",
                "category",
                "freshness",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter((row["split"], row["food_status"]) for row in rows)
    print(f"Đã tạo: {args.output}")
    for key, count in sorted(counts.items()):
        print(f"  {key[0]:<5} {key[1]:<11}: {count}")
    if not any(row["food_status"] != "SUPPORTED" for row in rows):
        print(
            "CẢNH BÁO: chưa có negative data; food head chưa thể dùng để reject."
        )


if __name__ == "__main__":
    main()
