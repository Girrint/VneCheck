"""
Phát hiện near-duplicate bằng imagehash (pHash).
Nhẹ, ổn định, không phụ thuộc cấu trúc thư mục.

Usage:
    pip install imagehash pillow
    python scripts/check_near_duplicates_v2.py --threshold 5
    python scripts/check_near_duplicates_v2.py --threshold 5 --save-csv reports/near_dupes.csv
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import imagehash
from PIL import Image

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff')


def index_folder(root, hash_size=8):
    """Trả về {path: imagehash}."""
    hashes = {}
    n_ok, n_err = 0, 0
    for dirpath, _, files in os.walk(root):
        for f in files:
            if not f.lower().endswith(IMG_EXTS):
                continue
            p = os.path.join(dirpath, f)
            try:
                with Image.open(p) as im:
                    h = imagehash.phash(im.convert("RGB"), hash_size=hash_size)
                hashes[p] = h
                n_ok += 1
                if n_ok % 10000 == 0:
                    print(f"    ... đã encode {n_ok} ảnh")
            except Exception as e:
                n_err += 1
                if n_err <= 5:
                    print(f"  ⚠️  Lỗi {p}: {e}")
    print(f"  Đã encode {n_ok} ảnh ({n_err} lỗi) trong {root}")
    return hashes


def ensure_parent_dir(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
        print(f"  📁 Đã tạo thư mục: {parent}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train")
    ap.add_argument("--valid", default="data/valid")
    ap.add_argument("--threshold", type=int, default=5,
                    help="Hamming distance tối đa (0-64)")
    ap.add_argument("--hash-size", type=int, default=8,
                    help="Kích thước hash (8 -> 64 bit)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--save-csv")
    args = ap.parse_args()

    print("=" * 70)
    print("KIỂM TRA NEAR-DUPLICATE (pHash + imagehash)")
    print(f"Threshold : {args.threshold}")
    print(f"Hash size : {args.hash_size} ({args.hash_size**2} bit)")
    print("=" * 70)

    for d in (args.train, args.valid):
        if not os.path.isdir(d):
            print(f"❌ Không tìm thấy: {d}")
            sys.exit(1)

    print("\n[1/3] Encoding train...")
    train_h = index_folder(args.train, hash_size=args.hash_size)

    print("\n[2/3] Encoding valid...")
    valid_h = index_folder(args.valid, hash_size=args.hash_size)

    if not train_h or not valid_h:
        print("\n❌ Một trong hai tập rỗng.")
        sys.exit(1)

    print("\n[3/3] So khớp...")
    print(f"  Train: {len(train_h)} ảnh, Valid: {len(valid_h)} ảnh")
    print(f"  Tổng phép so sánh: {len(train_h) * len(valid_h):,}")

    train_items = list(train_h.items())
    valid_items = list(valid_h.items())

    pairs = []
    stats = defaultdict(int)
    valid_flagged = set()

    for vi, (vpath, vhash) in enumerate(valid_items):
        if vi % 500 == 0 and vi > 0:
            print(f"  ... {vi}/{len(valid_items)} ảnh valid")

        best_dist = 999
        best_tpath = None
        for tpath, thash in train_items:
            d = vhash - thash  # imagehash hỗ trợ trừ -> int
            if d < best_dist:
                best_dist = d
                best_tpath = tpath
                if d == 0:
                    break

        if best_dist <= args.threshold:
            pairs.append((vpath, best_tpath, best_dist))
            valid_flagged.add(vpath)
            v_key = os.path.relpath(os.path.dirname(vpath), args.valid)
            stats[v_key] += 1

    print("\n" + "=" * 70)
    print("KẾT QUẢ")
    print("=" * 70)
    print(f"Tổng ảnh train             : {len(train_items)}")
    print(f"Tổng ảnh valid             : {len(valid_items)}")
    print(f"Ảnh valid gần giống train  : {len(valid_flagged)} "
          f"({len(valid_flagged)/len(valid_items):.1%})")
    print(f"Số cặp                     : {len(pairs)}")

    if valid_flagged:
        print(f"\n🚨 {len(valid_flagged)} ảnh valid gần giống train!")

    if stats:
        print("\n--- Top 20 class bị ảnh hưởng ---")
        for k, v in sorted(stats.items(), key=lambda x: -x[1])[:20]:
            print(f"  {k:<45s} {v:>6d}")

    if args.verbose:
        print("\n--- Chi tiết 100 cặp đầu ---")
        for vp, tp, d in pairs[:100]:
            print(f"  [dist={d}] {vp}")
            print(f"           ≈ {tp}")

    if args.save_csv:
        ensure_parent_dir(args.save_csv)
        with open(args.save_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["valid_path", "train_path", "hamming_distance"])
            w.writerows(pairs)
        print(f"\n💾 Đã lưu {len(pairs)} cặp vào {args.save_csv}")

    sys.exit(1 if valid_flagged else 0)


if __name__ == "__main__":
    main()