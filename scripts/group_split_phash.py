"""
Chia dataset theo pHash clustering.

Nguyên tắc:
- Encode pHash cho tất cả ảnh trong mỗi class.
- Nhóm ảnh có Hamming distance <= threshold thành cluster.
- Chia cluster vào train/valid (không tách cluster).
- LƯU CACHE clustering -> chạy lại không cần clustering lại.

Usage:
    # Lần đầu (clustering + chia)
    python scripts/group_split_phash.py --source data/train

    # Lần sau (dùng cache, không clustering lại)
    python scripts/group_split_phash.py --source data/train --use-cache

    # Chỉ xem kế hoạch
    python scripts/group_split_phash.py --source data/train --dry-run --use-cache
"""
import argparse
import json
import os
import pickle
import random
import shutil
import sys
import time
from collections import defaultdict

import imagehash
from PIL import Image


IMG_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
CACHE_FILE = "reports/phash_clusters.pkl"


def list_images_by_class(root):
    by_class = defaultdict(list)
    for dirpath, _, files in os.walk(root):
        for f in files:
            if not f.lower().endswith(IMG_EXTS):
                continue
            p = os.path.join(dirpath, f)
            rel = os.path.relpath(dirpath, root).replace(os.sep, "/")
            by_class[rel].append(p)
    return by_class


def phash_of(path, hash_size=8):
    try:
        with Image.open(path) as im:
            return imagehash.phash(im.convert("RGB"), hash_size=hash_size)
    except Exception:
        return None


def cluster_by_phash(paths, threshold=5):
    """Nhóm ảnh theo pHash dùng Union-Find."""
    print(f"    Encoding {len(paths)} ảnh...")
    t0 = time.time()
    items = []
    for p in paths:
        h = phash_of(p)
        if h is not None:
            items.append((p, h))
    print(f"    Encode xong trong {time.time()-t0:.1f}s")

    n = len(items)
    if n == 0:
        return []

    hashes = [h for _, h in items]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    print(f"    Clustering {n} ảnh...")
    t0 = time.time()

    BATCH = 2000
    for start in range(0, n, BATCH):
        end = min(start + BATCH, n)
        for i in range(start, end):
            hi = hashes[i]
            for j in range(i + 1, end):
                if (hi - hashes[j]) <= threshold:
                    union(i, j)
            for j in range(0, start):
                if (hi - hashes[j]) <= threshold:
                    union(i, j)
        print(f"    Batch {start//BATCH + 1}/{(n + BATCH - 1)//BATCH} "
              f"({time.time()-t0:.0f}s)")

    clusters = defaultdict(list)
    for i, (p, h) in enumerate(items):
        clusters[find(i)].append(p)

    return list(clusters.values())


def split_clusters(clusters, train_ratio, rng):
    """Chia clusters vào train/valid sao cho cân bằng nhất.

    Chiến lược:
    1. Sort clusters theo kích thước giảm dần.
    2. Greedy: thêm vào train nếu chưa đạt target.
    3. Bù từ valid nếu train thiếu, hoặc ngược lại.
    """
    rng.shuffle(clusters)  # shuffle trước để không bias
    n_total = sum(len(c) for c in clusters)
    target_train = int(n_total * train_ratio)

    # Sort theo kích thước giảm dần
    clusters_sorted = sorted(clusters, key=len, reverse=True)

    train, valid = [], []
    for c in clusters_sorted:
        if len(train) + len(c) <= target_train:
            train.extend(c)
        else:
            valid.extend(c)

    # Nếu train thiếu nhiều (> 5% target), bù từ valid
    gap = target_train - len(train)
    if gap > target_train * 0.05:
        # Tìm cluster trong valid nhỏ nhất để chuyển
        valid_clusters = [c for c in clusters_sorted if c[0] in valid]
        valid_clusters.sort(key=len)
        for c in valid_clusters:
            if gap <= 0:
                break
            if len(c) <= gap + target_train * 0.05:  # chấp nhận lệch nhẹ
                train.extend(c)
                for p in c:
                    valid.remove(p)
                gap -= len(c)

    return train, valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="data/train")
    ap.add_argument("--output", default="data_group_split")
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--valid-ratio", type=float, default=0.2)
    ap.add_argument("--threshold", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--use-cache", action="store_true",
                    help="Dùng cache clusters đã lưu (bỏ qua bước clustering)")
    ap.add_argument("--cache-file", default=CACHE_FILE)
    args = ap.parse_args()

    random.seed(args.seed)
    rng = random.Random(args.seed)

    print("=" * 70)
    print("CHIA THEO PHASH CLUSTERING")
    print(f"  Source   : {args.source}")
    print(f"  Output   : {args.output}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Tỷ lệ    : {args.train_ratio}/{args.valid_ratio}")
    print(f"  Cache    : {'DÙNG' if args.use_cache else 'KHÔNG (sẽ clustering)'}")
    print("=" * 70)

    print("\n[1/3] Liệt kê ảnh theo class...")
    by_class = list_images_by_class(args.source)
    total = sum(len(v) for v in by_class.values())
    print(f"  Tổng: {total} ảnh, {len(by_class)} class")

    # Clustering hoặc load cache
    clusters_by_class = {}
    if args.use_cache and os.path.isfile(args.cache_file):
        print(f"\n[2/3] Load cache từ {args.cache_file}...")
        with open(args.cache_file, "rb") as f:
            clusters_by_class = pickle.load(f)
        print(f"  Đã load {len(clusters_by_class)} class từ cache")
    else:
        print("\n[2/3] Clustering theo pHash...")
        t_start = time.time()
        for cls in sorted(by_class.keys()):
            paths = by_class[cls]
            print(f"\n  Class: {cls} ({len(paths)} ảnh)")
            clusters = cluster_by_phash(paths, threshold=args.threshold)
            n_clusters = len(clusters)
            avg_size = len(paths) / n_clusters if n_clusters > 0 else 0
            print(f"    -> {n_clusters} clusters (trung bình {avg_size:.1f} ảnh/cluster)")
            clusters_by_class[cls] = clusters

        # Lưu cache
        os.makedirs(os.path.dirname(args.cache_file), exist_ok=True)
        with open(args.cache_file, "wb") as f:
            pickle.dump(clusters_by_class, f)
        print(f"\n[Clustering xong trong {(time.time()-t_start)/60:.1f} phút]")
        print(f"💾 Đã lưu cache: {args.cache_file}")

    # Chia
    print("\n[3/3] Chia clusters vào train/valid...")
    plan = {}
    for cls, clusters in clusters_by_class.items():
        train_paths, valid_paths = split_clusters(
            clusters, args.train_ratio, rng
        )
        plan[cls] = {
            "total": len(train_paths) + len(valid_paths),
            "n_clusters": len(clusters),
            "train": train_paths,
            "valid": valid_paths,
        }

    # In kế hoạch
    print(f"\n{'Class':<45s} {'total':>7s} {'clusters':>9s} {'train':>7s} {'valid':>6s} {'%val':>6s}")
    print("-" * 95)
    sum_train = sum_valid = 0
    for cls in sorted(plan.keys()):
        p = plan[cls]
        pct_val = len(p['valid']) / p['total'] * 100 if p['total'] else 0
        flag = " ⚠️" if pct_val < 10 or pct_val > 40 else ""
        print(f"{cls:<45s} {p['total']:>7d} {p['n_clusters']:>9d} "
              f"{len(p['train']):>7d} {len(p['valid']):>6d} {pct_val:>5.1f}%{flag}")
        sum_train += len(p["train"])
        sum_valid += len(p["valid"])
    print("-" * 95)
    print(f"{'TỔNG':<45s} {total:>7d} {'':>9s} {sum_train:>7d} {sum_valid:>6d} "
          f"{sum_valid/total*100:>5.1f}%")

    if args.dry_run:
        print("\n🔍 DRY-RUN — không copy gì.")
        return

    print(f"\n⚠️  Sẽ COPY {total} ảnh sang {args.output}/")
    print("Gõ 'SPLIT' để xác nhận: ", end="")
    if input().strip() != "SPLIT":
        print("❌ Huỷ.")
        return

    print("\nCopy ảnh...")
    for split in ("train", "valid"):
        os.makedirs(os.path.join(args.output, split), exist_ok=True)

    copied = 0
    for cls, p in plan.items():
        for split in ("train", "valid"):
            for src in p[split]:
                rel = os.path.relpath(src, args.source)
                dest = os.path.join(args.output, split, rel)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy2(src, dest)
                copied += 1
        print(f"  {cls:<45s} done")

    print(f"\n✅ Đã copy {copied} ảnh")
    print(f"📁 Output: {args.output}/train, {args.output}/valid")


if __name__ == "__main__":
    main()