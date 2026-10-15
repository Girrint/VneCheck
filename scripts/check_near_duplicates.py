"""
Phát hiện NEAR-duplicate (ảnh gần giống) giữa train và valid.
Dùng perceptual hash (pHash) — phát hiện được ảnh đã resize/crop/nén.

Usage:
    pip install imagededup pillow
    python scripts/check_near_duplicates.py                    # tóm tắt
    python scripts/check_near_duplicates.py --threshold 8      # nới lỏng
    python scripts/check_near_duplicates.py --workers 4        # đa luồng (nếu ổn)
    python scripts/check_near_duplicates.py --save-csv reports/near_dupes.csv
    python scripts/check_near_duplicates.py --delete-valid     # xoá ảnh valid gần giống
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

from imagededup.methods import PHash


def index_folder(phasher, root, workers=0):
    """Encode tất cả ảnh trong folder.

    workers=0  -> single-thread (an toàn trên Windows + Python 3.12)
    workers>0  -> đa luồng (nhanh hơn nhưng có thể lỗi im lặng trên Windows)
    """
    print(f"  Encoding {root}...")
    encodings = phasher.encode_images(image_dir=root, num_enc_workers=workers)
    print(f"  Đã encode {len(encodings)} ảnh")
    return encodings


def hamming_distance(h1: int, h2: int) -> int:
    """Đếm số bit khác nhau giữa 2 hash (đã ở dạng int)."""
    return (h1 ^ h2).bit_count()  # Python 3.10+


def ensure_parent_dir(path: str):
    """Tạo thư mục cha nếu chưa tồn tại."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
        print(f"  📁 Đã tạo thư mục: {parent}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train")
    ap.add_argument("--valid", default="data/valid")
    ap.add_argument("--threshold", type=int, default=5,
                    help="Hamming distance tối đa (nhỏ hơn = giống hơn, 0-64)")
    ap.add_argument("--workers", type=int, default=0,
                    help="Số worker encode (0 = single-thread, an toàn Windows)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--save-csv", help="Lưu danh sách cặp gần giống")
    ap.add_argument("--delete-valid", action="store_true")
    args = ap.parse_args()

    print("=" * 70)
    print("KIỂM TRA NEAR-DUPLICATE (pHash) GIỮA TRAIN VÀ VALID")
    print(f"Threshold (Hamming distance): {args.threshold}")
    print(f"Workers                     : {args.workers} "
          f"({'single-thread' if args.workers == 0 else 'đa luồng'})")
    print("=" * 70)

    # Kiểm tra thư mục tồn tại
    for d in (args.train, args.valid):
        if not os.path.isdir(d):
            print(f"❌ Không tìm thấy thư mục: {d}")
            sys.exit(1)

    phasher = PHash()

    print("\n[1/3] Encoding train...")
    train_enc = index_folder(phasher, args.train, workers=args.workers)

    print("\n[2/3] Encoding valid...")
    valid_enc = index_folder(phasher, args.valid, workers=args.workers)

    if not train_enc or not valid_enc:
        print("\n❌ Một trong hai tập rỗng — không thể so khớp.")
        print("   → Kiểm tra lại đường dẫn hoặc định dạng ảnh.")
        sys.exit(1)

    print("\n[3/3] So khớp...")
    print("  (có thể mất vài phút với dataset lớn)")

    # Chuyển hash hex sang int để so nhanh (bit_count nhanh hơn bin().count)
    train_items = [(p, int(h, 16)) for p, h in train_enc.items()]
    valid_items = [(p, int(h, 16)) for p, h in valid_enc.items()]

    # Tham chiếu nhanh đến thư mục valid tuyệt đối
    valid_root_abs = os.path.abspath(args.valid)

    pairs = []
    stats = defaultdict(int)
    valid_flagged = set()

    for vi, (vpath, vhash) in enumerate(valid_items):
        if vi % 500 == 0 and vi > 0:
            print(f"  ... đã xử lý {vi}/{len(valid_items)} ảnh valid")

        for tpath, thash in train_items:
            if hamming_distance(vhash, thash) <= args.threshold:
                pairs.append((vpath, tpath, hamming_distance(vhash, thash)))
                valid_flagged.add(vpath)
                v_key = os.path.relpath(os.path.dirname(vpath), args.valid)
                stats[v_key] += 1
                break  # chỉ cần 1 match là đủ

    print("\n" + "=" * 70)
    print("KẾT QUẢ")
    print("=" * 70)
    print(f"Tổng ảnh train                    : {len(train_items)}")
    print(f"Tổng ảnh valid                    : {len(valid_items)}")
    print(f"Ảnh valid GẦN GIỐNG train         : {len(valid_flagged)}")
    if valid_items:
        print(f"Tỉ lệ                             : "
              f"{len(valid_flagged) / len(valid_items):.1%}")
    print(f"Số cặp                             : {len(pairs)}")

    if valid_flagged:
        print(f"\n🚨 {len(valid_flagged)} ảnh valid gần giống với train!")
        print("   → Cần xoá trước khi đánh giá, nếu không metric vẫn sai.")

    # Top class bị ảnh hưởng
    if stats:
        print("\n--- Top 20 class có ảnh valid gần giống train ---")
        print(f"{'Class':<45s} {'#valid_dup':>10s}")
        print("-" * 57)
        for k, v in sorted(stats.items(), key=lambda x: -x[1])[:20]:
            print(f"{k:<45s} {v:>10d}")

    if args.verbose:
        print("\n--- Chi tiết từng cặp (100 đầu) ---")
        for vp, tp, d in pairs[:100]:
            print(f"  [dist={d}] {vp}")
            print(f"           ≈ {tp}")
        if len(pairs) > 100:
            print(f"  ... và {len(pairs) - 100} cặp nữa")

    # Lưu CSV — tự tạo thư mục cha
    if args.save_csv:
        ensure_parent_dir(args.save_csv)
        with open(args.save_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["valid_path", "train_path", "hamming_distance"])
            w.writerows(pairs)
        print(f"\n💾 Đã lưu {len(pairs)} cặp vào {args.save_csv}")

    # Xoá ảnh valid gần giống
    if args.delete_valid and valid_flagged:
        print(f"\n⚠️  Sẽ xoá {len(valid_flagged)} ảnh valid gần giống train.")
        print("Gõ 'DELETE' để xác nhận: ", end="")
        if input().strip() == "DELETE":
            deleted = 0
            for vp in valid_flagged:
                # vp có thể là đường dẫn tương đối hoặc tuyệt đối tùy imagededup
                full = vp if os.path.isabs(vp) else os.path.join(args.valid, vp)
                if os.path.exists(full):
                    os.remove(full)
                    deleted += 1
                else:
                    print(f"  ⚠️  Không tìm thấy: {full}")
            print(f"✅ Đã xoá {deleted} ảnh")
        else:
            print("❌ Huỷ.")

    sys.exit(1 if valid_flagged else 0)


if __name__ == "__main__":
    main()