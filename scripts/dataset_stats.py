"""Thống kê số lượng ảnh theo cấu trúc split/category/freshness/product.

Script chỉ duyệt đường dẫn và phần mở rộng, không mở hoặc giải mã ảnh.

Ví dụ:
    python scripts/dataset_stats.py
    python scripts/dataset_stats.py --root data
    python scripts/dataset_stats.py --csv reports/dataset_stats.csv
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}
EXPECTED_SPLITS = ("train", "valid", "test")


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def image_paths(root: Path):
    return (
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def collect_stats(root: Path):
    rows = Counter()
    extensions = Counter()
    malformed = []

    for path in image_paths(root):
        relative = path.relative_to(root)
        parts = relative.parts
        if len(parts) < 4:
            malformed.append(relative.as_posix())
            continue

        split, category, freshness = parts[:3]
        product = "/".join(parts[3:-1]) or "(không có product)"
        rows[(split, category, freshness, product)] += 1
        extensions[path.suffix.lower()] += 1

    return rows, extensions, malformed


def aggregate(rows, prefix_length):
    result = Counter()
    for key, count in rows.items():
        result[key[:prefix_length]] += count
    return result


def print_table(title, headers, values):
    print(f"\n{title}")
    print("-" * 88)
    if not values:
        print("  Không có dữ liệu.")
        return

    widths = [len(header) for header in headers]
    for row in values:
        for index, value in enumerate(row[:-1]):
            widths[index] = max(widths[index], len(str(value)))
    widths[-1] = max(widths[-1], len("Số ảnh"))

    header = "  ".join(
        f"{name:<{widths[index]}}" if index < len(headers) - 1
        else f"{name:>{widths[index]}}"
        for index, name in enumerate(headers)
    )
    print(header)
    print("  ".join("-" * width for width in widths))
    for row in values:
        cells = [
            f"{value:<{widths[index]}}" if index < len(row) - 1
            else f"{value:>{widths[index]}}"
            for index, value in enumerate(row)
        ]
        print("  ".join(cells))


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(("split", "category", "freshness", "product", "count"))
        for key, count in sorted(rows.items()):
            writer.writerow((*key, count))


def main():
    parser = argparse.ArgumentParser(
        description="Thống kê số lượng ảnh mà không mở nội dung ảnh"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data_split"),
        help="Dataset root có các thư mục split (mặc định: data_split)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Xuất thống kê chi tiết theo product ra CSV",
    )
    parser.add_argument(
        "--products",
        action="store_true",
        help="In thêm bảng chi tiết tới cấp product",
    )
    args = parser.parse_args()

    if not args.root.is_dir():
        raise SystemExit(f"Không tìm thấy dataset root: {args.root}")

    rows, extensions, malformed = collect_stats(args.root)
    total = sum(rows.values())

    print("=" * 88)
    print("THỐNG KÊ DATASET")
    print(f"Root     : {args.root.resolve()}")
    print(f"Tổng ảnh : {total}")
    print("=" * 88)

    split_counts = aggregate(rows, 1)
    split_order = {
        name: index for index, name in enumerate(EXPECTED_SPLITS)
    }
    def row_order(item):
        key = item[0]
        return (split_order.get(key[0], 999), *key[1:])

    split_values = [
        (split, count)
        for (split,), count in sorted(
            split_counts.items(),
            key=lambda item: (split_order.get(item[0][0], 999), item[0][0]),
        )
    ]
    print_table("THEO SPLIT", ("Split", "Số ảnh"), split_values)

    class_counts = aggregate(rows, 3)
    class_values = [
        (*key, count)
        for key, count in sorted(class_counts.items(), key=row_order)
    ]
    print_table(
        "THEO SPLIT / CATEGORY / FRESHNESS",
        ("Split", "Category", "Freshness", "Số ảnh"),
        class_values,
    )

    if args.products:
        product_values = [
            (*key, count) for key, count in sorted(rows.items(), key=row_order)
        ]
        print_table(
            "THEO PRODUCT",
            ("Split", "Category", "Freshness", "Product", "Số ảnh"),
            product_values,
        )

    if extensions:
        extension_text = ", ".join(
            f"{extension}: {count}" for extension, count in sorted(extensions.items())
        )
        print(f"\nĐịnh dạng: {extension_text}")

    if malformed:
        print(
            f"Cảnh báo: bỏ qua {len(malformed)} ảnh không đúng cấu trúc "
            "split/category/freshness/product."
        )
        for path in malformed[:10]:
            print(f"  {path}")

    if args.csv:
        write_csv(args.csv, rows)
        print(f"\nĐã lưu CSV: {args.csv}")


if __name__ == "__main__":
    main()
