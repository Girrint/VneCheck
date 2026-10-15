"""
Chia lại dataset từ 1 thư mục duy nhất (data/valid/) thành train/valid/test.

CHIẾN LƯỢC:
- Chia theo class (group/level/food) — mỗi class chia riêng.
- Tỷ lệ mặc định: 70% train, 15% valid, 15% test.
- Shuffle ngẫu nhiên với seed cố định (reproducible).
- Ghi log đầy đủ.

Usage:
    # Xem kế hoạch trước
    python scripts/resplit_dataset.py --dry-run

    # Chia thật (70/15/15)
    python scripts/resplit_dataset.py

    # Chia 80/20 (không test)
    python scripts/resplit_dataset.py --train-ratio 0.8 --valid-ratio 0.2 --test-ratio 0.0
"""
import argparse
import os
import random
import shutil
import sys
from collections import defaultdict


IMG_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')


def list_images_by_class(root):
    """Trả về {class: [paths]}."""
    by_class = defaultdict(list)
    for dirpath, _, files in os.walk(root):
        for f in files:
            if not f.lower().endswith(IMG_EXTS):
                continue
            p = os.path.join(dirpath, f)
            # Class = đường dẫn tương đối từ root
            rel = os.path.relpath(dirpath, root).replace(os.sep, "/")
            by_class[rel].append(p)
    return by_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="data/valid",
                    help="Thư mục chứa toàn bộ ảnh hiện tại")
    ap.add_argument("--output", default="data_split",
                    help="Thư mục output (sẽ tạo train/valid/test)")
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--valid-ratio", type=float, default=0.15)
    ap.add_argument("--test-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # Kiểm tra tỷ lệ
    total_ratio = args.train_ratio + args.valid_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        print(f"❌ Tổng tỷ lệ phải = 1.0, hiện tại = {total_ratio}")
        sys.exit(1)

    random.seed(args.seed)

    print("=" * 70)
    print("CHIA LẠI DATASET")
    print(f"  Source      : {args.source}")
    print(f"  Output      : {args.output}")
    print(f"  Train ratio : {args.train_ratio}")
    print(f"  Valid ratio : {args.valid_ratio}")
    print(f"  Test ratio  : {args.test_ratio}")
    print(f"  Seed        : {args.seed}")
    print("=" * 70)

    print("\n[1/3] Liệt kê ảnh theo class...")
    by_class = list_images_by_class(args.source)
    total = sum(len(v) for v in by_class.values())
    print(f"  Tổng: {total} ảnh, {len(by_class)} class")

    print("\n[2/3] Lập kế hoạch chia...")
    plan = {}
    for cls, paths in by_class.items():
        random.shuffle(paths)
        n = len(paths)
        n_train = int(n * args.train_ratio)
        n_valid = int(n * args.valid_ratio)
        n_test = n - n_train - n_valid

        # Nếu không muốn test, dồn phần dư vào train
        if args.test_ratio == 0:
            n_train += n_test
            n_test = 0

        plan[cls] = {
            "total": n,
            "train": paths[:n_train],
            "valid": paths[n_train:n_train + n_valid],
            "test": paths[n_train + n_valid:],
        }

    # In kế hoạch
    print(f"\n{'Class':<45s} {'total':>7s} {'train':>7s} {'valid':>7s} {'test':>6s}")
    print("-" * 80)
    sum_train = sum_valid = sum_test = 0
    for cls in sorted(plan.keys()):
        p = plan[cls]
        print(f"{cls:<45s} {p['total']:>7d} {len(p['train']):>7d} "
              f"{len(p['valid']):>7d} {len(p['test']):>6d}")
        sum_train += len(p["train"])
        sum_valid += len(p["valid"])
        sum_test += len(p["test"])
    print("-" * 80)
    print(f"{'TỔNG':<45s} {total:>7d} {sum_train:>7d} {sum_valid:>7d} {sum_test:>6d}")

    if args.dry_run:
        print("\n🔍 DRY-RUN — không copy gì cả.")
        return

    # Xác nhận
    print(f"\n⚠️  Sẽ COPY {total} ảnh từ {args.source} sang {args.output}/")
    print("Gõ 'SPLIT' để xác nhận: ", end="")
    if input().strip() != "SPLIT":
        print("❌ Huỷ.")
        return

    print("\n[3/3] Copy ảnh...")
    for split in ("train", "valid", "test"):
        os.makedirs(os.path.join(args.output, split), exist_ok=True)

    copied = 0
    for cls, p in plan.items():
        for split in ("train", "valid", "test"):
            for src in p[split]:
                rel = os.path.relpath(src, args.source)
                dest = os.path.join(args.output, split, rel)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy2(src, dest)
                copied += 1
                if copied % 10000 == 0:
                    print(f"  ... đã copy {copied}/{total}")
        print(f"  {cls:<45s} done")

    print(f"\n✅ Đã copy {copied} ảnh")
    print(f"📁 Output: {args.output}/train, {args.output}/valid, {args.output}/test")


if __name__ == "__main__":
    main()