"""Khử trùng lặp và chia UNSUPPORTED/NON_FOOD theo SHA-256 + pHash."""

import argparse
import csv
import os
import pickle
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.resplit_dataset import (
    IMG_EXTS,
    cluster_records,
    encode_records,
    split_clusters,
    write_audit_reports,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


CACHE_VERSION = 1
STATUSES = ("UNSUPPORTED", "NON_FOOD")
SPLITS = ("train", "valid", "test")


def image_records(root, class_name):
    if not root.is_dir():
        raise FileNotFoundError(f"Không tìm thấy source: {root}")
    records = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMG_EXTS:
            continue
        resolved = path.resolve()
        stat = resolved.stat()
        records.append({
            "path": str(resolved),
            "class": class_name,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        })
    return sorted(records, key=lambda row: os.path.normcase(row["path"]))


def load_sha_cache(path):
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        if payload.get("version") != CACHE_VERSION:
            return {}
        return payload.get("entries", {})
    except (OSError, EOFError, ValueError, TypeError, pickle.UnpicklingError):
        return {}


def sha256_worker(path):
    import hashlib

    try:
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return path, digest.hexdigest(), None
    except Exception as error:
        return path, None, f"{type(error).__name__}: {error}"


def add_sha256(records, workers, cache_path):
    cache = load_sha_cache(cache_path)
    pending = []
    for record in records:
        previous = cache.get(record["path"])
        if (
            previous
            and previous.get("size") == record["size"]
            and previous.get("mtime_ns") == record["mtime_ns"]
            and previous.get("sha256")
        ):
            record["sha256"] = previous["sha256"]
        else:
            pending.append(record)

    print(f"  Cache SHA-256 hợp lệ: {len(records) - len(pending)} ảnh")
    print(f"  Cần tính SHA-256: {len(pending)} ảnh")
    paths = [record["path"] for record in pending]
    executor = None
    if not paths:
        results = ()
    elif workers:
        executor = ProcessPoolExecutor(max_workers=workers)
        results = executor.map(sha256_worker, paths, chunksize=32)
    else:
        results = map(sha256_worker, paths)

    values = {}
    errors = []
    try:
        for index, (path, digest, error) in enumerate(results, start=1):
            if error:
                errors.append((path, error))
            else:
                values[path] = digest
            if index % 5000 == 0:
                print(f"    ... đã hash {index}/{len(pending)} ảnh mới")
    finally:
        if executor is not None:
            executor.shutdown()

    entries = {}
    hashed = []
    for record in records:
        digest = record.get("sha256") or values.get(record["path"])
        if digest is None:
            continue
        record["sha256"] = digest
        entries[record["path"]] = {
            "size": record["size"],
            "mtime_ns": record["mtime_ns"],
            "sha256": digest,
        }
        hashed.append(record)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(f".{cache_path.name}.tmp")
    with temporary.open("wb") as stream:
        pickle.dump(
            {"version": CACHE_VERSION, "entries": entries},
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    os.replace(temporary, cache_path)
    return hashed, errors


def deduplicate(negative_records, supported_records):
    supported_sha = {record["sha256"] for record in supported_records}
    by_sha = defaultdict(list)
    for record in negative_records:
        by_sha[record["sha256"]].append(record)

    kept = []
    excluded = []
    for digest in sorted(by_sha):
        group = sorted(by_sha[digest], key=lambda row: os.path.normcase(row["path"]))
        labels = sorted({record["class"] for record in group})
        if digest in supported_sha:
            reason = "conflict_with_supported"
        elif len(labels) > 1:
            reason = "negative_label_conflict"
        else:
            reason = None

        if reason:
            for record in group:
                excluded.append({
                    "reason": reason,
                    "class": record["class"],
                    "sha256": digest,
                    "path": record["path"],
                    "kept_path": "",
                })
            continue

        representative = group[0]
        kept.append(representative)
        for duplicate in group[1:]:
            excluded.append({
                "reason": "exact_duplicate",
                "class": duplicate["class"],
                "sha256": digest,
                "path": duplicate["path"],
                "kept_path": representative["path"],
            })
    return kept, excluded


def write_excluded(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("reason", "class", "sha256", "path", "kept_path"),
        )
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["reason"], row["path"])))


def ensure_empty_splits(output):
    nonempty = []
    for split in SPLITS:
        split_root = output / split
        if split_root.exists() and any(path.is_file() for path in split_root.rglob("*")):
            nonempty.append(split_root)
    if nonempty:
        raise FileExistsError(
            "Output split không rỗng: " + ", ".join(map(str, nonempty))
        )


def copy_plan(plan, output):
    used = set()
    copied = 0
    for split, records in plan.items():
        for record in records:
            source = Path(record["path"])
            base = output / split / record["class"] / source.name
            destination = base
            normalized = os.path.normcase(str(destination))
            if normalized in used or destination.exists():
                destination = base.with_name(
                    f"{base.stem}_{record['sha256'][:12]}{base.suffix.lower()}"
                )
                normalized = os.path.normcase(str(destination))
            if normalized in used or destination.exists():
                raise FileExistsError(f"Trùng destination: {destination}")
            used.add(normalized)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied += 1
            if copied % 1000 == 0:
                print(f"    ... đã copy {copied}/{sum(map(len, plan.values()))} ảnh")


def print_plan(plan):
    counts = {
        split: Counter(record["class"] for record in records)
        for split, records in plan.items()
    }
    print(f"\n{'Class':<16} {'total':>8} {'train':>8} {'valid':>8} {'test':>8}")
    print("-" * 52)
    for status in STATUSES:
        values = [counts[split][status] for split in SPLITS]
        print(f"{status:<16} {sum(values):>8} {values[0]:>8} {values[1]:>8} {values[2]:>8}")
    totals = [sum(counts[split].values()) for split in SPLITS]
    print("-" * 52)
    print(f"{'TỔNG':<16} {sum(totals):>8} {totals[0]:>8} {totals[1]:>8} {totals[2]:>8}")


def main():
    parser = argparse.ArgumentParser(
        description="Khử trùng lặp và chia negative dataset 70/15/15"
    )
    parser.add_argument(
        "--negative-root", type=Path, default=Path("data_negative/_staging")
    )
    parser.add_argument("--supported-root", type=Path, default=Path("data_split"))
    parser.add_argument("--output", type=Path, default=Path("data_negative"))
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--threshold", type=int, default=0)
    parser.add_argument("--hash-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument(
        "--sha-cache", type=Path, default=Path("reports/negative_sha256_cache.pkl")
    )
    parser.add_argument(
        "--phash-cache", type=Path, default=Path("reports/negative_phash_cache.pkl")
    )
    parser.add_argument(
        "--excluded-report",
        type=Path,
        default=Path("reports/negative_excluded_files.csv"),
    )
    parser.add_argument(
        "--audit-summary",
        type=Path,
        default=Path("reports/negative_cluster_summary.csv"),
    )
    parser.add_argument(
        "--audit-members",
        type=Path,
        default=Path("reports/negative_cluster_members.csv"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ratios = (args.train_ratio, args.valid_ratio, args.test_ratio)
    if any(value < 0 for value in ratios) or abs(sum(ratios) - 1) > 1e-9:
        raise SystemExit("Tỷ lệ train/valid/test phải có tổng bằng 1")
    if not 0 <= args.threshold <= args.hash_size**2:
        raise SystemExit("--threshold nằm ngoài số bit pHash")
    if args.workers < 0:
        raise SystemExit("--workers không được âm")

    print("=" * 72)
    print("CHUẨN BỊ NEGATIVE DATASET THEO SHA-256 + PHASH")
    print(f"  Negative  : {args.negative_root}")
    print(f"  Supported : {args.supported_root}")
    print(f"  Output    : {args.output}")
    print(f"  Ratios    : {args.train_ratio}/{args.valid_ratio}/{args.test_ratio}")
    print(f"  pHash     : {args.hash_size**2} bit, threshold={args.threshold}")
    print("=" * 72)

    print("\n[1/6] Liệt kê ảnh...")
    try:
        negative = []
        for status in STATUSES:
            negative.extend(image_records(args.negative_root / status, status))
        supported = image_records(args.supported_root, "SUPPORTED")
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error
    counts = Counter(record["class"] for record in negative)
    print(
        f"  Negative: {len(negative)} "
        f"(UNSUPPORTED={counts['UNSUPPORTED']}, NON_FOOD={counts['NON_FOOD']})"
    )
    print(f"  Supported để đối chiếu: {len(supported)}")
    if any(counts[status] == 0 for status in STATUSES):
        raise SystemExit("Cả UNSUPPORTED và NON_FOOD phải có ảnh")

    print("\n[2/6] Tính SHA-256...")
    combined, sha_errors = add_sha256(negative + supported, args.workers, args.sha_cache)
    negative_paths = {record["path"] for record in negative}
    negative = [record for record in combined if record["path"] in negative_paths]
    supported = [record for record in combined if record["path"] not in negative_paths]
    print(f"  Thành công: {len(combined)} ảnh; lỗi: {len(sha_errors)}")

    print("\n[3/6] Loại exact duplicate và xung đột...")
    negative, excluded = deduplicate(negative, supported)
    for path, error in sha_errors:
        if path in negative_paths:
            excluded.append({
                "reason": f"sha256_error: {error}",
                "class": "",
                "sha256": "",
                "path": path,
                "kept_path": "",
            })
    write_excluded(args.excluded_report, excluded)
    reasons = Counter(row["reason"].split(":", 1)[0] for row in excluded)
    print(f"  Còn lại: {len(negative)} ảnh")
    for reason, count in sorted(reasons.items()):
        print(f"  Loại {reason}: {count}")
    print(f"  Báo cáo: {args.excluded_report}")

    print("\n[4/6] Tính pHash cho negative...")
    negative, phash_errors = encode_records(
        negative, args.hash_size, args.workers, args.phash_cache
    )
    if phash_errors:
        print(f"  Không đọc được: {len(phash_errors)} ảnh")
    print(f"  Thành công: {len(negative)} ảnh")

    print("\n[5/6] Gom cluster và chia 70/15/15...")
    clusters = cluster_records(negative, args.threshold)
    plan, _, forced_train = split_clusters(clusters, ratios, args.seed)
    write_audit_reports(clusters, plan, args.audit_summary, args.audit_members)
    print(f"  Clusters: {len(clusters)}; lớn nhất: {max(map(len, clusters), default=0)}")
    print(f"  Cluster ép vào train: {forced_train}")
    print_plan(plan)

    print("\n[6/6] Ghi output...")
    if args.dry_run:
        print("  DRY-RUN: không copy ảnh; cache và báo cáo audit đã được cập nhật.")
        return
    try:
        ensure_empty_splits(args.output)
    except FileExistsError as error:
        raise SystemExit(str(error)) from error
    copy_plan(plan, args.output)
    print(f"  Đã copy {sum(map(len, plan.values()))} ảnh vào {args.output}")


if __name__ == "__main__":
    main()
