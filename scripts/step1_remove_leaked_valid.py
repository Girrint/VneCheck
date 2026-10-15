"""
Bước 1: Xoá ảnh valid bị leak (đã phát hiện bởi check_near_duplicates_v2.py).

CHIẾN LƯỢC AN TOÀN:
- Chỉ xoá ảnh có hamming_distance <= --max-distance (mặc định 0 = trùng hệt).
- Giữ tối thiểu --keep-per-class ảnh mỗi class (không xoá sạch class).
- Backup trước khi xoá (copy, không move).
- Ghi log đầy đủ.

Usage:
    python scripts/step1_remove_leaked_valid.py --dry-run
    python scripts/step1_remove_leaked_valid.py --csv reports/dupes_exact.csv --dry-run
    python scripts/step1_remove_leaked_valid.py --max-distance 2
    python scripts/step1_remove_leaked_valid.py --max-distance 5 --keep-per-class 200
"""
import argparse
import csv
import os
import shutil
import sys
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="reports/near_dupes.csv",
                    help="File CSV chứa danh sách cặp")
    ap.add_argument("--backup", default="_backup_valid_leaked",
                    help="Thư mục backup ảnh trước khi xoá")
    ap.add_argument("--log", default="reports/deleted_valid.csv",
                    help="File log ghi lại ảnh đã xoá / giữ")
    ap.add_argument("--max-distance", type=int, default=0,
                    help="Chỉ xoá ảnh có hamming_distance <= N (mặc định 0)")
    ap.add_argument("--keep-per-class", type=int, default=100,
                    help="Giữ tối thiểu N ảnh mỗi class (mặc định 100)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Chỉ in kế hoạch, KHÔNG xoá gì")
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        print(f"❌ Không tìm thấy {args.csv}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Đọc CSV: {valid_path: min_hamming_distance}
    #    Tự động phát hiện cột distance (hamming_distance hoặc sha256)
    # ------------------------------------------------------------------
    valid_to_distance = {}
    with open(args.csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        # Tự động chọn cột distance
        if "hamming_distance" in r.fieldnames:
            dist_col = "hamming_distance"
        elif "sha256" in r.fieldnames:
            dist_col = "sha256"
        else:
            print(f"❌ CSV không có cột distance.")
            print(f"   Các cột hiện có: {r.fieldnames}")
            sys.exit(1)
        print(f"📄 Đọc cột distance: '{dist_col}'")
        for row in r:
            vp = row["valid_path"]
            if dist_col == "sha256":
                d = 0  # SHA-256 trùng hệt -> distance = 0
            else:
                d = int(row[dist_col])
            if vp not in valid_to_distance or d < valid_to_distance[vp]:
                valid_to_distance[vp] = d

    print(f"📋 Tổng số ảnh valid bị flag là leak: {len(valid_to_distance)}")

    # Phân bố theo distance
    dist_counter = Counter(valid_to_distance.values())
    print("\n--- Phân bố theo hamming_distance ---")
    for d in sorted(dist_counter.keys()):
        n = dist_counter[d]
        marker = "  ← sẽ xoá" if d <= args.max_distance else ""
        print(f"  distance={d}: {n:>6d} ảnh{marker}")

    # ------------------------------------------------------------------
    # 2. Lọc theo max-distance
    # ------------------------------------------------------------------
    candidates = {p: d for p, d in valid_to_distance.items() if d <= args.max_distance}
    print(f"\n📋 Sẽ xem xét xoá {len(candidates)} ảnh (distance <= {args.max_distance})")

    if not candidates:
        print("\n✅ Không có ảnh nào cần xoá theo ngưỡng hiện tại.")
        return

    # ------------------------------------------------------------------
    # 3. Tính class của từng ảnh candidate
    # ------------------------------------------------------------------
    class_of = {}
    for p in candidates:
        parts = p.replace("\\", "/").split("/")
        if len(parts) >= 4:
            class_of[p] = "/".join(parts[-4:-1])
        else:
            class_of[p] = "/".join(parts[:-1])

    class_counts = Counter(class_of.values())
    print("\n--- Số ảnh sẽ xoá theo class (trước khi áp dụng keep-per-class) ---")
    for k, v in class_counts.most_common(30):
        print(f"  {k:<45s} {v:>6d}")

    # ------------------------------------------------------------------
    # 4. Đếm tổng valid mỗi class, CHUẨN HOÁ dấu phân cách
    # ------------------------------------------------------------------
    print("\n[Đang đếm tổng valid mỗi class...]")
    total_valid_per_class = Counter()
    valid_root = "data/valid"
    for dirpath, _, files in os.walk(valid_root):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
                rel = os.path.relpath(dirpath, valid_root).replace(os.sep, "/")
                total_valid_per_class[rel] += 1

    # ------------------------------------------------------------------
    # 5. Quyết định ảnh nào xoá (áp dụng keep-per-class)
    # ------------------------------------------------------------------
    to_delete = []
    delete_per_class = Counter()
    for cls in class_counts:
        total = total_valid_per_class.get(cls, 0)
        max_delete = max(0, total - args.keep_per_class)
        cls_candidates = sorted(
            [(p, d) for p, d in candidates.items() if class_of[p] == cls],
            key=lambda x: x[1]
        )
        for p, d in cls_candidates[:max_delete]:
            to_delete.append((p, d))
            delete_per_class[cls] += 1

    print(f"\n📋 Sẽ XOÁ {len(to_delete)} ảnh (sau keep-per-class={args.keep_per_class})")
    print(f"📋 Sẽ GIỮ {len(candidates) - len(to_delete)} ảnh bị flag")

    print("\n--- Chi tiết xoá theo class ---")
    print(f"{'Class':<45s} {'total_valid':>11s} {'flagged':>8s} {'delete':>7s} {'keep':>6s}")
    print("-" * 85)
    for cls in sorted(class_counts.keys()):
        total = total_valid_per_class.get(cls, 0)
        flagged = class_counts[cls]
        dele = delete_per_class.get(cls, 0)
        print(f"{cls:<45s} {total:>11d} {flagged:>8d} {dele:>7d} {total - dele:>6d}")

    # ------------------------------------------------------------------
    # 6. Dry-run
    # ------------------------------------------------------------------
    if args.dry_run:
        print("\n🔍 DRY-RUN — không xoá gì cả.")
        print(f"  Nếu chạy thật, {len(to_delete)} ảnh sẽ được backup vào {args.backup} rồi xoá.")
        print("\n20 ảnh đầu tiên sẽ xoá:")
        for p, d in to_delete[:20]:
            print(f"  [dist={d}] {p}")
        return

    # ------------------------------------------------------------------
    # 7. Xác nhận + thực hiện
    # ------------------------------------------------------------------
    print(f"\n⚠️  Sẽ BACKUP {len(to_delete)} ảnh vào {args.backup}")
    print("    rồi XOÁ khỏi data/valid/")
    print("Gõ 'DELETE' để xác nhận: ", end="")
    if input().strip() != "DELETE":
        print("❌ Huỷ.")
        return

    os.makedirs(args.backup, exist_ok=True)
    os.makedirs(os.path.dirname(args.log) if os.path.dirname(args.log) else ".", exist_ok=True)

    deleted = 0
    missing = 0
    with open(args.log, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "hamming_distance", "backup_path", "status"])
        for p, d in to_delete:
            if not os.path.exists(p):
                w.writerow([p, d, "", "not_found"])
                missing += 1
                continue
            rel = os.path.relpath(p, valid_root) if p.startswith(valid_root) else os.path.basename(p)
            backup_path = os.path.join(args.backup, rel)
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            shutil.copy2(p, backup_path)
            os.remove(p)
            w.writerow([p, d, backup_path, "deleted"])
            deleted += 1

    print(f"\n✅ Đã backup + xoá {deleted} ảnh")
    if missing:
        print(f"⚠️  {missing} ảnh không tìm thấy")
    print(f"📁 Backup: {args.backup}")
    print(f"📄 Log: {args.log}")


if __name__ == "__main__":
    main()