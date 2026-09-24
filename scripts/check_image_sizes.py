"""
Kiểm tra kích thước ảnh trong dataset.

Usage:
    # Thống kê kích thước
    python scripts/check_image_sizes.py

    # Chỉ kiểm tra 1 split
    python scripts/check_image_sizes.py --root data/valid

    # Liệt kê ảnh nhỏ hơn 512x512
    python scripts/check_image_sizes.py --min-size 512 --list-small

    # Xoá ảnh nhỏ hơn 512x512
    python scripts/check_image_sizes.py --min-size 512 --delete-small

    # Resize toàn bộ về 512x512
    python scripts/check_image_sizes.py --resize 512
"""
import argparse
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image


IMG_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')


def iter_images(root):
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                yield os.path.join(dirpath, f)


def analyze(root):
    """Thống kê kích thước ảnh."""
    size_counter = Counter()
    per_class = defaultdict(lambda: {
        "total": 0, "small": 0,
        "min_w": 10**9, "min_h": 10**9,
        "max_w": 0, "max_h": 0,
        "sizes": Counter(),
    })
    total = 0
    broken = []

    for path in iter_images(root):
        try:
            with Image.open(path) as im:
                w, h = im.size
        except Exception as e:
            broken.append((path, str(e)))
            continue

        total += 1
        size_counter[(w, h)] += 1

        # Class key: thư mục ngay trên ảnh
        rel = os.path.relpath(path, root)
        parts = rel.split(os.sep)
        if len(parts) >= 3:
            cls = os.sep.join(parts[:-1])  # group/level/food
        else:
            cls = os.path.dirname(rel)

        c = per_class[cls]
        c["total"] += 1
        c["min_w"] = min(c["min_w"], w)
        c["min_h"] = min(c["min_h"], h)
        c["max_w"] = max(c["max_w"], w)
        c["max_h"] = max(c["max_h"], h)
        c["sizes"][(w, h)] += 1

    return {
        "total": total,
        "broken": broken,
        "size_counter": size_counter,
        "per_class": per_class,
    }


def print_summary(result, min_size):
    total = result["total"]
    print(f"\n{'=' * 70}")
    print(f"TỔNG: {total} ảnh")
    print(f"{'=' * 70}")

    if result["broken"]:
        print(f"\n⚠️  {len(result['broken'])} ảnh LỖI (không mở được):")
        for p, e in result["broken"][:10]:
            print(f"   {p}  → {e}")

    # Top kích thước phổ biến
    print(f"\n--- Top 15 kích thước phổ biến ---")
    print(f"{'Kích thước':<20s} {'Số lượng':>10s} {'%':>7s}")
    print("-" * 40)
    for (w, h), n in result["size_counter"].most_common(15):
        print(f"{f'{w}x{h}':<20s} {n:>10d} {n/total:>6.1%}")

    # Đếm ảnh nhỏ
    small = sum(n for (w, h), n in result["size_counter"].items()
                if w < min_size or h < min_size)
    print(f"\n--- Ảnh nhỏ hơn {min_size}x{min_size} ---")
    print(f"Số lượng: {small}/{total} ({small/total:.1%})")

    # Top class có ảnh nhỏ
    print(f"\n--- Class có nhiều ảnh nhỏ nhất ---")
    rows = []
    for cls, c in result["per_class"].items():
        small_n = sum(n for (w, h), n in c["sizes"].items()
                      if w < min_size or h < min_size)
        if small_n > 0:
            rows.append((cls, small_n, c["total"], c["min_w"], c["min_h"]))
    rows.sort(key=lambda x: -x[1])

    print(f"{'Class':<42s} {'small':>7s} {'total':>7s} {'min WxH':>12s}")
    print("-" * 70)
    for cls, small_n, tot, mw, mh in rows[:25]:
        print(f"{cls:<42s} {small_n:>7d} {tot:>7d} {f'{mw}x{mh}':>12s}")

    return small


def list_small(root, min_size, limit=None):
    n = 0
    for path in iter_images(root):
        try:
            with Image.open(path) as im:
                w, h = im.size
        except Exception:
            continue
        if w < min_size or h < min_size:
            print(f"  {w}x{h}  {path}")
            n += 1
            if limit and n >= limit:
                print(f"  ... (dừng ở {limit})")
                return n
    return n


def delete_small(root, min_size):
    print(f"\n⚠️  Sẽ XOÁ ảnh nhỏ hơn {min_size}x{min_size} trong {root}")
    print("Gõ 'DELETE' để xác nhận: ", end="")
    if input().strip() != "DELETE":
        print("❌ Huỷ.")
        return
    n = 0
    for path in iter_images(root):
        try:
            with Image.open(path) as im:
                w, h = im.size
        except Exception:
            continue
        if w < min_size or h < min_size:
            os.remove(path)
            n += 1
    print(f"✅ Đã xoá {n} ảnh")


def resize_all(root, target):
    """Resize toàn bộ ảnh về target x target (giữ aspect ratio? ở đây ép vuông)."""
    print(f"\n⚠️  Sẽ RESIZE toàn bộ ảnh trong {root} về {target}x{target}")
    print("Ghi đè file gốc! Gõ 'RESIZE' để xác nhận: ", end="")
    if input().strip() != "RESIZE":
        print("❌ Huỷ.")
        return
    n = 0
    for path in iter_images(root):
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                im = im.resize((target, target), Image.LANCZOS)
                # Lưu đè, giữ format gốc
                fmt = "JPEG" if path.lower().endswith((".jpg", ".jpeg")) else None
                im.save(path, format=fmt, quality=92)
            n += 1
        except Exception as e:
            print(f"⚠️  Lỗi {path}: {e}")
    print(f"✅ Đã resize {n} ảnh")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data", help="Thư mục gốc")
    ap.add_argument("--min-size", type=int, default=224,
                    help="Ngưỡng kích thước tối thiểu (mặc định 224)")
    ap.add_argument("--list-small", action="store_true",
                    help="Liệt kê ảnh nhỏ hơn ngưỡng")
    ap.add_argument("--delete-small", action="store_true",
                    help="Xoá ảnh nhỏ hơn ngưỡng")
    ap.add_argument("--resize", type=int, metavar="N",
                    help="Resize toàn bộ về NxN")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        print(f"❌ Không tìm thấy {args.root}")
        sys.exit(1)

    print(f"Đang quét {args.root}...")
    result = analyze(args.root)
    print_summary(result, args.min_size)

    if args.list_small:
        print(f"\n--- Danh sách ảnh < {args.min_size}x{args.min_size} ---")
        n = list_small(args.root, args.min_size)
        print(f"Tổng: {n} ảnh")

    if args.delete_small:
        delete_small(args.root, args.min_size)

    if args.resize:
        resize_all(args.root, args.resize)


if __name__ == "__main__":
    main()