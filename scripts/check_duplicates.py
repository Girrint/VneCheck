"""
Phát hiện ảnh trùng byte giữa train và valid bằng SHA-256.
Output dạng TÓM TẮT để dễ đọc với dataset lớn.

Usage:
    python scripts/check_duplicates.py                    # tóm tắt
    python scripts/check_duplicates.py --verbose          # in từng cặp
    python scripts/check_duplicates.py --save-csv dupes.csv  # lưu danh sách
    python scripts/check_duplicates.py --delete-valid     # xoá ảnh valid trùng
"""
import argparse
import csv
import hashlib
import os
import sys
from collections import defaultdict
from pathlib import Path


def file_hash(path: str, chunk_size: int = 1 << 20) -> str:
    """Tính SHA-256 của file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def index_folder(root: str) -> dict:
    """Trả về {hash: [danh sách đường dẫn]}."""
    hash2paths = defaultdict(list)
    exts = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
    n = 0
    for dirpath, _, files in os.walk(root):
        for fname in files:
            if not fname.lower().endswith(exts):
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                h = file_hash(fpath)
                hash2paths[h].append(fpath)
                n += 1
            except Exception as e:
                print(f"⚠️  Lỗi đọc {fpath}: {e}", file=sys.stderr)
    print(f"  Đã index {n} ảnh trong {root}")
    return hash2paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train")
    ap.add_argument("--valid", default="data/valid")
    ap.add_argument("--verbose", action="store_true",
                    help="In chi tiết từng cặp trùng")
    ap.add_argument("--save-csv", help="Lưu danh sách trùng vào CSV")
    ap.add_argument("--delete-valid", action="store_true",
                    help="Xoá ảnh valid trùng với train (cần xác nhận)")
    args = ap.parse_args()

    print("=" * 70)
    print("KIỂM TRA TRÙNG SHA-256 GIỮA TRAIN VÀ VALID")
    print("=" * 70)

    print("\n[1/3] Index train...")
    train_hashes = index_folder(args.train)

    print("\n[2/3] Index valid...")
    valid_hashes = index_folder(args.valid)

    print("\n[3/3] So khớp...")
    common = set(train_hashes.keys()) & set(valid_hashes.keys())

    # Thống kê theo thư mục
    stats = defaultdict(lambda: {"train": 0, "valid": 0})
    pairs = []

    for h in common:
        t_paths = train_hashes[h]
        v_paths = valid_hashes[h]
        # Lấy thư mục cha làm key (level/food)
        t_key = os.path.relpath(os.path.dirname(t_paths[0]), args.train)
        v_key = os.path.relpath(os.path.dirname(v_paths[0]), args.valid)
        stats[t_key]["train"] += len(t_paths)
        stats[v_key]["valid"] += len(v_paths)
        for tp in t_paths:
            for vp in v_paths:
                pairs.append((vp, tp, h))

    # Tổng kết
    n_train_dup = sum(1 for h in common for _ in train_hashes[h])
    n_valid_dup = sum(1 for h in common for _ in valid_hashes[h])
    total_train = sum(len(v) for v in train_hashes.values())
    total_valid = sum(len(v) for v in valid_hashes.values())

    print("\n" + "=" * 70)
    print("KẾT QUẢ")
    print("=" * 70)
    print(f"Tổng ảnh train             : {total_train}")
    print(f"Tổng ảnh valid             : {total_valid}")
    print(f"Số hash trùng              : {len(common)}")
    print(f"Ảnh train trùng valid      : {n_train_dup}  ({n_train_dup/total_train:.1%})")
    print(f"Ảnh valid trùng train      : {n_valid_dup}  ({n_valid_dup/total_valid:.1%})")
    print(f"Số cặp (valid, train)      : {len(pairs)}")

    if n_valid_dup > 0:
        print(f"\n🚨 CẢNH BÁO: {n_valid_dup} ảnh trong valid đã xuất hiện trong train!")
        print("   → Metric validation KHÔNG đáng tin.")
        print("   → Phải xoá khỏi valid trước khi đánh giá.")

    # Thống kê theo class
    if stats:
        print("\n--- Trùng theo class ---")
        print(f"{'Class':<45s} {'#dup':>6s}")
        print("-" * 55)
        for k in sorted(stats.keys()):
            n = max(stats[k]["train"], stats[k]["valid"])
            print(f"{k:<45s} {n:>6d}")

    # In chi tiết nếu --verbose
    if args.verbose:
        print("\n--- Chi tiết từng cặp ---")
        for vp, tp, h in pairs:
            print(f"  {vp}")
            print(f"    == {tp}  [{h[:12]}...]")

    # Lưu CSV
    if args.save_csv:
        with open(args.save_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["valid_path", "train_path", "sha256"])
            w.writerows(pairs)
        print(f"\n💾 Đã lưu {len(pairs)} cặp vào {args.save_csv}")

    # Xoá ảnh valid trùng
    if args.delete_valid and n_valid_dup > 0:
        print(f"\n⚠️  Sẽ xoá {n_valid_dup} ảnh valid. Gõ 'DELETE' để xác nhận: ", end="")
        if input().strip() == "DELETE":
            deleted = 0
            for h in common:
                for vp in valid_hashes[h]:
                    os.remove(vp)
                    deleted += 1
            print(f"✅ Đã xoá {deleted} ảnh")
        else:
            print("❌ Huỷ.")

    # Exit code
    sys.exit(1 if n_valid_dup > 0 else 0)


if __name__ == "__main__":
    main()