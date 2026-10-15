"""
Bước 2: Di chuyển ảnh từ train sang valid để bù số lượng.

CHIẾN LƯỢC (Cách 3 - Chia lại tỷ lệ):
- Với mỗi class, target valid = 20% tổng (train + valid).
- Nhưng không vượt quá --max-target (mặc định 500).
- Không thấp hơn --min-target (mặc định 50).
- Luôn giữ tối thiểu --keep-train-ratio (mặc định 0.8) tổng train.

AN TOÀN:
- Check dry_run trước mọi thao tác move.
- Ghi log CSV.
- Bọc shutil.move trong try/except.
- Kiểm tra đích tồn tại trước khi move.

Usage:
    python scripts/step2_move_train_to_valid.py --dry-run
    python scripts/step2_move_train_to_valid.py
    python scripts/step2_move_train_to_valid.py --max-target 600 --min-target 100
"""
import argparse
import csv
import os
import random
import shutil
import sys
from collections import defaultdict

import imagehash
from PIL import Image

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff')


def list_images(root):
    paths = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                paths.append(os.path.join(dirpath, f))
    return paths


def class_key(path, root):
    rel = os.path.relpath(path, root).replace(os.sep, "/")
    parts = rel.split("/")
    if len(parts) >= 4:
        return "/".join(parts[-4:-1])
    return "/".join(parts[:-1])


def phash_of(path, hash_size=8):
    try:
        with Image.open(path) as im:
            return imagehash.phash(im.convert("RGB"), hash_size=hash_size)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train")
    ap.add_argument("--valid", default="data/valid")
    ap.add_argument("--max-target", type=int, default=500,
                    help="Target valid tối đa mỗi class")
    ap.add_argument("--min-target", type=int, default=50,
                    help="Target valid tối thiểu mỗi class")
    ap.add_argument("--valid-ratio", type=float, default=0.2,
                    help="Tỷ lệ valid/tổng (mặc định 0.2 = 20%)")
    ap.add_argument("--keep-train-ratio", type=float, default=0.8,
                    help="Giữ tối thiểu tỷ lệ này của train (mặc định 0.8)")
    ap.add_argument("--threshold", type=int, default=5,
                    help="Ngưỡng pHash coi là trùng")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log", default="reports/moved_train_to_valid.csv")
    args = ap.parse_args()

    random.seed(args.seed)

    print("=" * 70)
    print("BƯỚC 2: BÙ ẢNH TỪ TRAIN SANG VALID (Cách 3 - Tỷ lệ)")
    print(f"  max-target      : {args.max_target}")
    print(f"  min-target      : {args.min_target}")
    print(f"  valid-ratio     : {args.valid_ratio}")
    print(f"  keep-train-ratio: {args.keep_train_ratio}")
    print("=" * 70)

    print("\n[1/5] Liệt kê ảnh...")
    valid_paths = list_images(args.valid)
    train_paths = list_images(args.train)
    print(f"  Valid: {len(valid_paths)} ảnh")
    print(f"  Train: {len(train_paths)} ảnh")

    print("\n[2/5] Đếm theo class...")
    valid_by_class = defaultdict(int)
    for p in valid_paths:
        valid_by_class[class_key(p, args.valid)] += 1
    train_by_class = defaultdict(list)
    for p in train_paths:
        train_by_class[class_key(p, args.train)].append(p)

    all_classes = sorted(set(valid_by_class) | set(train_by_class))
    print(f"  Tổng số class: {len(all_classes)}")

    print("\n[3/5] Lập kế hoạch...")
    plan = {}
    for cls in all_classes:
        n_valid = valid_by_class.get(cls, 0)
        n_train = len(train_by_class.get(cls, []))
        total = n_valid + n_train

        # Target = 20% tổng, kẹp trong [min_target, max_target]
        target = int(total * args.valid_ratio)
        target = max(args.min_target, min(args.max_target, target))

        need = max(0, target - n_valid)
        # Giới hạn: chỉ lấy tối đa (1 - keep_train_ratio) của train
        max_move = int(n_train * (1 - args.keep_train_ratio))
        actual = min(need, max_move)

        if actual > 0:
            plan[cls] = {
                "n_valid": n_valid,
                "n_train": n_train,
                "total": total,
                "target": target,
                "need": need,
                "max_move": max_move,
                "actual": actual,
            }

    if not plan:
        print("\n✅ Không cần bù ảnh.")
        return

    print(f"\n--- Kế hoạch bù ({len(plan)} class) ---")
    print(f"{'Class':<45s} {'train':>7s} {'valid':>6s} {'total':>7s} "
          f"{'target':>6s} {'cần':>5s} {'max':>5s} {'chuyển':>7s}")
    print("-" * 100)
    total_move = 0
    for cls in sorted(plan.keys()):
        p = plan[cls]
        print(f"{cls:<45s} {p['n_train']:>7d} {p['n_valid']:>6d} {p['total']:>7d} "
              f"{p['target']:>6d} {p['need']:>5d} {p['max_move']:>5d} {p['actual']:>7d}")
        total_move += p["actual"]
    print(f"\n📋 Tổng sẽ chuyển: {total_move} ảnh")

    if args.dry_run:
        print("\n🔍 DRY-RUN — không chuyển gì.")
        return

    # Xác nhận
    print(f"\n⚠️  Sẽ chuyển {total_move} ảnh từ train sang valid.")
    print("Gõ 'MOVE' để xác nhận: ", end="")
    if input().strip() != "MOVE":
        print("❌ Huỷ.")
        return

    print("\n[4/5] Encode pHash valid...")
    valid_hashes = defaultdict(list)
    for p in valid_paths:
        h = phash_of(p)
        if h is not None:
            valid_hashes[class_key(p, args.valid)].append(h)
    print(f"  Đã encode {sum(len(v) for v in valid_hashes.values())} ảnh valid")

    print("\n[5/5] Chuyển ảnh...")
    os.makedirs(os.path.dirname(args.log) if os.path.dirname(args.log) else ".", exist_ok=True)

    moved = 0
    skipped_dup = 0
    skipped_other = 0
    with open(args.log, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["src", "dst", "class", "status"])

        for cls, info in plan.items():
            need = info["actual"]
            candidates = list(train_by_class.get(cls, []))
            if not candidates:
                continue
            random.shuffle(candidates)
            cls_hashes = list(valid_hashes.get(cls, []))
            cls_moved = 0

            for tp in candidates:
                if cls_moved >= need:
                    break

                th = phash_of(tp)
                if th is None:
                    skipped_other += 1
                    continue

                # Kiểm tra trùng với valid hiện có
                if any((th - vh) <= args.threshold for vh in cls_hashes):
                    skipped_dup += 1
                    continue

                rel = os.path.relpath(tp, args.train)
                dest = os.path.join(args.valid, rel)

                if os.path.exists(dest):
                    w.writerow([tp, dest, cls, "dest_exists"])
                    skipped_other += 1
                    continue

                os.makedirs(os.path.dirname(dest), exist_ok=True)
                try:
                    shutil.move(tp, dest)
                    cls_hashes.append(th)
                    cls_moved += 1
                    moved += 1
                    w.writerow([tp, dest, cls, "moved"])
                except Exception as e:
                    w.writerow([tp, dest, cls, f"error: {e}"])
                    skipped_other += 1

            print(f"  {cls:<45s} chuyển {cls_moved}/{need}")

    print(f"\n✅ Đã chuyển {moved} ảnh")
    print(f"⚠️  Bỏ qua do trùng pHash: {skipped_dup}")
    print(f"⚠️  Bỏ qua do lỗi/khác: {skipped_other}")
    print(f"📄 Log: {args.log}")


if __name__ == "__main__":
    main()