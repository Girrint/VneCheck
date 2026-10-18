"""
Gộp các split hiện có rồi chia lại dataset theo SHA-256 và pHash cluster.

Mặc định script đọc đồng thời data/train và data/valid. File có cùng SHA-256
nhưng khác nhãn bị loại khỏi output; duplicate cùng nhãn chỉ giữ một đại diện.
pHash chỉ được so trong cùng class. Sau đó script chia nguyên cluster theo tỷ
lệ 70/15/15 để ảnh gần giống không bị rò rỉ giữa train, valid và test.

Usage:
    python scripts/resplit_dataset.py --dry-run
    python scripts/resplit_dataset.py
    python scripts/resplit_dataset.py --threshold 0 --workers 4 --dry-run
"""

import argparse
import csv
import hashlib
import os
import pickle
import random
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imagehash
from PIL import Image


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
CACHE_VERSION = 1


def class_key(path: Path, source: Path) -> str:
    """Class là thư mục chứa ảnh, tương đối với một split source."""
    return path.parent.relative_to(source).as_posix()


def list_images(sources):
    """Liệt kê ảnh từ nhiều split và bỏ tên split khỏi class key."""
    records = []
    seen_paths = set()

    for raw_source in sources:
        source = Path(raw_source).resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"Không tìm thấy source: {raw_source}")

        for path in source.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in IMG_EXTS:
                continue
            resolved = path.resolve()
            normalized = os.path.normcase(str(resolved))
            if normalized in seen_paths:
                continue
            seen_paths.add(normalized)
            stat = resolved.stat()
            records.append({
                "path": str(resolved),
                "class": class_key(resolved, source),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            })

    records.sort(key=lambda item: os.path.normcase(item["path"]))
    return records


def phash_worker(payload):
    path, hash_size = payload
    try:
        with Image.open(path) as image:
            value = imagehash.phash(image.convert("RGB"), hash_size=hash_size)
        return path, int(str(value), 16), None
    except Exception as error:
        return path, None, f"{type(error).__name__}: {error}"


def sha256_worker(path):
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return path, digest.hexdigest(), None
    except Exception as error:
        return path, None, f"{type(error).__name__}: {error}"


def load_cache(path: Path):
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        if not isinstance(payload, dict) or payload.get("version") != CACHE_VERSION:
            return {}
        return payload.get("entries", {})
    except (OSError, EOFError, ValueError, TypeError, pickle.UnpicklingError):
        return {}


def save_cache(path: Path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        pickle.dump(
            {"version": CACHE_VERSION, "entries": entries},
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    os.replace(temporary, path)


def encode_records(records, hash_size, workers, cache_path):
    """Tính pHash; cache chỉ được dùng khi size và mtime vẫn khớp."""
    cached = load_cache(cache_path)
    entries = {}
    pending = []

    for record in records:
        previous = cached.get(record["path"])
        if (
            previous
            and previous.get("size") == record["size"]
            and previous.get("mtime_ns") == record["mtime_ns"]
            and previous.get("hash_size") == hash_size
            and previous.get("phash") is not None
        ):
            record["phash"] = previous["phash"]
            entries[record["path"]] = previous
        else:
            pending.append(record)

    print(f"  Cache hợp lệ: {len(records) - len(pending)} ảnh")
    print(f"  Cần tính pHash: {len(pending)} ảnh")

    jobs = [(record["path"], hash_size) for record in pending]
    executor = None
    if workers == 0:
        results = map(phash_worker, jobs)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        results = executor.map(phash_worker, jobs, chunksize=64)

    result_by_path = {}
    errors = []
    try:
        for index, (path, value, error) in enumerate(results, start=1):
            if error is None:
                result_by_path[path] = value
            else:
                errors.append((path, error))
            if index % 5000 == 0:
                print(f"    ... đã xử lý {index}/{len(pending)} ảnh mới")
    finally:
        if executor is not None:
            executor.shutdown()

    encoded = []
    for record in records:
        if "phash" not in record:
            record["phash"] = result_by_path.get(record["path"])
        if record["phash"] is None:
            continue
        entry = dict(cached.get(record["path"], {}))
        entry.update({
            "size": record["size"],
            "mtime_ns": record["mtime_ns"],
            "hash_size": hash_size,
            "phash": record["phash"],
        })
        entries[record["path"]] = entry
        encoded.append(record)

    save_cache(cache_path, entries)
    return encoded, errors


def encode_sha256(records, workers, cache_path):
    """Tính SHA-256 để xác định byte-identical duplicate trên toàn dataset."""
    cached = load_cache(cache_path)
    pending = []

    for record in records:
        previous = cached.get(record["path"])
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
    if workers == 0:
        results = map(sha256_worker, paths)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        results = executor.map(sha256_worker, paths, chunksize=32)

    result_by_path = {}
    errors = []
    try:
        for index, (path, value, error) in enumerate(results, start=1):
            if error is None:
                result_by_path[path] = value
            else:
                errors.append((path, error))
            if index % 5000 == 0:
                print(f"    ... đã hash {index}/{len(pending)} ảnh mới")
    finally:
        if executor is not None:
            executor.shutdown()

    entries = {}
    hashed = []
    for record in records:
        if "sha256" not in record:
            record["sha256"] = result_by_path.get(record["path"])
        if record["sha256"] is None:
            continue
        entry = dict(cached.get(record["path"], {}))
        entry.update({
            "size": record["size"],
            "mtime_ns": record["mtime_ns"],
            "phash": record["phash"],
            "sha256": record["sha256"],
        })
        entries[record["path"]] = entry
        hashed.append(record)

    save_cache(cache_path, entries)
    return hashed, errors


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left, right):
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


class BKTree:
    """BK-tree dùng tìm pHash trong một bán kính Hamming."""

    def __init__(self):
        self.root = None

    @staticmethod
    def distance(left, right):
        return (left ^ right).bit_count()

    def add(self, value, index):
        if self.root is None:
            self.root = [value, [index], {}]
            return

        node = self.root
        while True:
            distance = self.distance(value, node[0])
            if distance == 0:
                node[1].append(index)
                return
            child = node[2].get(distance)
            if child is None:
                node[2][distance] = [value, [index], {}]
                return
            node = child

    def search(self, value, threshold):
        if self.root is None:
            return []
        matches = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = self.distance(value, node[0])
            if distance <= threshold:
                matches.extend(node[1])
            low = distance - threshold
            high = distance + threshold
            for edge, child in node[2].items():
                if low <= edge <= high:
                    stack.append(child)
        return matches


def cluster_records(records, threshold):
    """Gom SHA-256 toàn cục; chỉ so pHash giữa các ảnh cùng class."""
    union_find = UnionFind(len(records))

    first_by_sha256 = {}
    for index, record in enumerate(records):
        first = first_by_sha256.setdefault(record["sha256"], index)
        union_find.union(index, first)

    trees_by_class = defaultdict(BKTree)

    for index, record in enumerate(records):
        tree = trees_by_class[record["class"]]
        for match in tree.search(record["phash"], threshold):
            union_find.union(index, match)
        tree.add(record["phash"], index)
        if (index + 1) % 10000 == 0:
            print(f"    ... đã cluster {index + 1}/{len(records)} ảnh")

    grouped = defaultdict(list)
    for index, record in enumerate(records):
        grouped[union_find.find(index)].append(record)
    return sorted(
        grouped.values(),
        key=lambda cluster: (-len(cluster), min(item["path"] for item in cluster)),
    )


def cluster_split_map(plan):
    return {
        record["path"]: split_name
        for split_name, split_records in plan.items()
        for record in split_records
    }


def model_label(class_name):
    """Nhãn model gồm category/freshness; product chỉ là tầng tổ chức."""
    parts = class_name.replace("\\", "/").split("/")
    return "/".join(parts[:2]) if len(parts) >= 2 else class_name


def prepare_unique_records(records):
    """Loại xung đột nhãn và chỉ giữ một đại diện cho mỗi SHA cùng nhãn."""
    by_sha256 = defaultdict(list)
    for record in records:
        by_sha256[record["sha256"]].append(record)

    kept = []
    excluded = []
    for sha256 in sorted(by_sha256):
        group = sorted(
            by_sha256[sha256],
            key=lambda record: os.path.normcase(record["path"]),
        )
        labels = sorted({model_label(record["class"]) for record in group})

        if len(labels) > 1:
            for record in group:
                excluded.append({
                    "status": "label_conflict",
                    "sha256": sha256,
                    "model_labels": ";".join(labels),
                    "class": record["class"],
                    "path": record["path"],
                    "kept_path": "",
                })
            continue

        representative = group[0]
        kept.append(representative)
        for record in group[1:]:
            excluded.append({
                "status": "exact_duplicate",
                "sha256": sha256,
                "model_labels": labels[0],
                "class": record["class"],
                "path": record["path"],
                "kept_path": representative["path"],
            })

    kept.sort(key=lambda record: os.path.normcase(record["path"]))
    excluded.sort(key=lambda row: (row["status"], row["sha256"], row["path"]))
    return kept, excluded


def write_excluded_report(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "status",
        "sha256",
        "model_labels",
        "class",
        "path",
        "kept_path",
    )
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def cluster_has_label_conflict(cluster):
    labels_by_sha256 = defaultdict(set)
    for record in cluster:
        labels_by_sha256[record["sha256"]].add(model_label(record["class"]))
    return any(len(labels) > 1 for labels in labels_by_sha256.values())


def write_audit_reports(clusters, plan, summary_path: Path, members_path: Path):
    """Ghi summary mọi cluster và member của các cluster cần xem xét."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    members_path.parent.mkdir(parents=True, exist_ok=True)
    assigned_split = cluster_split_map(plan)

    with summary_path.open("w", newline="", encoding="utf-8-sig") as stream:
        fieldnames = (
            "cluster_id",
            "cluster_size",
            "class_count",
            "sha256_count",
            "phash_count",
            "assigned_split",
            "label_conflict",
            "classes",
            "model_labels",
        )
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for cluster_id, cluster in enumerate(clusters, start=1):
            classes = sorted({record["class"] for record in cluster})
            model_labels = sorted({model_label(name) for name in classes})
            writer.writerow({
                "cluster_id": cluster_id,
                "cluster_size": len(cluster),
                "class_count": len(classes),
                "sha256_count": len({record["sha256"] for record in cluster}),
                "phash_count": len({record["phash"] for record in cluster}),
                "assigned_split": assigned_split[cluster[0]["path"]],
                "label_conflict": cluster_has_label_conflict(cluster),
                "classes": ";".join(classes),
                "model_labels": ";".join(model_labels),
            })

    with members_path.open("w", newline="", encoding="utf-8-sig") as stream:
        fieldnames = (
            "cluster_id",
            "cluster_size",
            "assigned_split",
            "label_conflict",
            "class",
            "model_label",
            "phash",
            "sha256",
            "path",
        )
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for cluster_id, cluster in enumerate(clusters, start=1):
            conflict = cluster_has_label_conflict(cluster)
            if len(cluster) == 1 and not conflict:
                continue
            for record in cluster:
                writer.writerow({
                    "cluster_id": cluster_id,
                    "cluster_size": len(cluster),
                    "assigned_split": assigned_split[record["path"]],
                    "label_conflict": conflict,
                    "class": record["class"],
                    "model_label": model_label(record["class"]),
                    "phash": f"{record['phash']:016x}",
                    "sha256": record["sha256"],
                    "path": record["path"],
                })


def target_counts(total, ratios):
    raw = [total * ratio for ratio in ratios]
    counts = [int(value) for value in raw]
    remainder = total - sum(counts)
    order = sorted(
        range(len(ratios)),
        key=lambda index: raw[index] - counts[index],
        reverse=True,
    )
    for index in order[:remainder]:
        counts[index] += 1
    return counts


def split_clusters(clusters, ratios, seed, max_eval_cluster_fraction=0.25):
    """Chia theo class; cluster quá lớn cho valid/test được ép vào train."""
    split_names = ("train", "valid", "test")
    classes = sorted({record["class"] for cluster in clusters for record in cluster})
    totals = Counter(record["class"] for cluster in clusters for record in cluster)
    targets = {
        class_name: dict(zip(split_names, target_counts(totals[class_name], ratios)))
        for class_name in classes
    }
    current = {class_name: Counter() for class_name in classes}
    plan = {name: [] for name in split_names}

    rng = random.Random(seed)
    decorated = [(rng.random(), cluster) for cluster in clusters]
    decorated.sort(key=lambda item: (-len(item[1]), item[0]))
    forced_train = 0

    for _, cluster in decorated:
        cluster_counts = Counter(record["class"] for record in cluster)

        force_train = False
        for class_name, count in cluster_counts.items():
            eval_targets = [
                targets[class_name][split_name]
                for split_name in ("valid", "test")
                if targets[class_name][split_name] > 0
            ]
            if (
                eval_targets
                and count > min(eval_targets) * max_eval_cluster_fraction
            ):
                force_train = True
                break

        def placement_cost(split_name):
            cost = 0.0
            for class_name, count in cluster_counts.items():
                for candidate in split_names:
                    target = targets[class_name][candidate]
                    after = current[class_name][candidate]
                    if candidate == split_name:
                        after += count
                    cost += ((after - target) / max(1, target)) ** 2
            return cost

        if force_train:
            destination = "train"
            forced_train += 1
        else:
            destination = min(split_names, key=placement_cost)
        plan[destination].extend(cluster)
        for class_name, count in cluster_counts.items():
            current[class_name][destination] += count

    return plan, targets, forced_train


def destination_for(record, output: Path, split_name: str, used):
    source = Path(record["path"])
    base = output / split_name / record["class"] / source.name
    normalized = os.path.normcase(str(base))
    if normalized not in used and not base.exists():
        used.add(normalized)
        return base

    suffix = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:10]
    candidate = base.with_name(f"{base.stem}_{suffix}{base.suffix.lower()}")
    normalized = os.path.normcase(str(candidate))
    if normalized in used or candidate.exists():
        raise FileExistsError(f"Không thể tạo destination duy nhất cho {source}")
    used.add(normalized)
    return candidate


def print_plan(plan, clusters, targets):
    cluster_of = {}
    for cluster_index, cluster in enumerate(clusters):
        for record in cluster:
            cluster_of[record["path"]] = cluster_index

    print(
        f"\n{'Class':<45s} {'total':>7s} {'train':>7s} "
        f"{'valid':>7s} {'test':>7s}"
    )
    print("-" * 84)

    split_counts = {
        name: Counter(record["class"] for record in records)
        for name, records in plan.items()
    }
    for class_name in sorted(targets):
        train = split_counts["train"][class_name]
        valid = split_counts["valid"][class_name]
        test = split_counts["test"][class_name]
        print(
            f"{class_name:<45s} {train + valid + test:>7d} "
            f"{train:>7d} {valid:>7d} {test:>7d}"
        )

    totals = {name: len(records) for name, records in plan.items()}
    print("-" * 84)
    print(
        f"{'TỔNG':<45s} {sum(totals.values()):>7d} "
        f"{totals['train']:>7d} {totals['valid']:>7d} {totals['test']:>7d}"
    )
    split_cluster_counts = {
        name: len({cluster_of[record["path"]] for record in records})
        for name, records in plan.items()
    }
    print(
        "Clusters: "
        f"train={split_cluster_counts['train']}, "
        f"valid={split_cluster_counts['valid']}, "
        f"test={split_cluster_counts['test']}"
    )


def validate_args(args):
    ratios = (args.train_ratio, args.valid_ratio, args.test_ratio)
    if any(ratio < 0 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise SystemExit("Tỷ lệ train/valid/test phải không âm và có tổng bằng 1.0")
    if args.threshold < 0 or args.threshold > args.hash_size**2:
        raise SystemExit("--threshold phải nằm trong số bit của pHash")
    if args.hash_size < 4:
        raise SystemExit("--hash-size phải từ 4 trở lên")
    if args.workers < 0:
        raise SystemExit("--workers không được âm")
    if args.max_cluster_size < 1:
        raise SystemExit("--max-cluster-size phải từ 1 trở lên")
    if not 0 < args.max_eval_cluster_fraction <= 1:
        raise SystemExit("--max-eval-cluster-fraction phải nằm trong (0, 1]")


def main():
    parser = argparse.ArgumentParser(
        description="Gộp dataset và chia train/valid/test theo SHA-256 + pHash"
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["data/train", "data/valid"],
        help="Các split nguồn cần gộp",
    )
    parser.add_argument("--output", default="data_split")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--threshold", type=int, default=0)
    parser.add_argument("--hash-size", type=int, default=8)
    parser.add_argument(
        "--max-cluster-size",
        type=int,
        default=500,
        help="Chặn copy nếu còn cluster lớn hơn ngưỡng này",
    )
    parser.add_argument(
        "--max-eval-cluster-fraction",
        type=float,
        default=0.25,
        help="Ép cluster vào train nếu vượt tỷ lệ này của target valid/test",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Số process tính pHash; 0 để chạy tuần tự",
    )
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=Path("reports/resplit_phash_cache.pkl"),
    )
    parser.add_argument(
        "--audit-summary",
        type=Path,
        default=Path("reports/resplit_cluster_summary.csv"),
    )
    parser.add_argument(
        "--audit-members",
        type=Path,
        default=Path("reports/resplit_cluster_members.csv"),
    )
    parser.add_argument(
        "--excluded-report",
        type=Path,
        default=Path("reports/resplit_excluded_files.csv"),
    )
    parser.add_argument("--allow-large-clusters", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    validate_args(args)

    ratios = (args.train_ratio, args.valid_ratio, args.test_ratio)
    output = Path(args.output).resolve()
    source_paths = [Path(source).resolve() for source in args.sources]
    if any(output == source or output.is_relative_to(source) for source in source_paths):
        raise SystemExit("--output không được nằm bên trong bất kỳ --sources nào")

    print("=" * 72)
    print("CHIA LẠI DATASET THEO SHA-256 + PHASH CLUSTER")
    print(f"  Sources    : {', '.join(args.sources)}")
    print(f"  Output     : {output}")
    print(f"  Ratios     : {args.train_ratio}/{args.valid_ratio}/{args.test_ratio}")
    print(f"  pHash      : {args.hash_size**2} bit, threshold={args.threshold}")
    print(f"  Eval limit : {args.max_eval_cluster_fraction:.0%} target/class")
    print(f"  Workers    : {args.workers}")
    print(f"  Cache      : {args.cache_file}")
    print(f"  Seed       : {args.seed}")
    print("=" * 72)

    print("\n[1/6] Liệt kê ảnh từ các source...")
    try:
        records = list_images(args.sources)
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error
    if not records:
        raise SystemExit("Không tìm thấy ảnh trong các source")
    print(f"  Tổng: {len(records)} ảnh, {len(set(r['class'] for r in records))} class")

    print("\n[2/6] Tính pHash...")
    records, phash_errors = encode_records(
        records,
        hash_size=args.hash_size,
        workers=args.workers,
        cache_path=args.cache_file,
    )
    print(f"  Thành công: {len(records)} ảnh")
    if phash_errors:
        print(
            f"  Không đọc được: {len(phash_errors)} ảnh "
            "(sẽ không đưa vào split)"
        )
        for path, error in phash_errors[:10]:
            print(f"    {path}: {error}")

    print("\n[3/6] Tính SHA-256...")
    records, sha256_errors = encode_sha256(
        records,
        workers=args.workers,
        cache_path=args.cache_file,
    )
    print(f"  Thành công: {len(records)} ảnh")
    if sha256_errors:
        print(
            f"  Không hash được: {len(sha256_errors)} ảnh "
            "(sẽ không đưa vào split)"
        )
        for path, error in sha256_errors[:10]:
            print(f"    {path}: {error}")

    print("\n[4/6] Loại xung đột nhãn và bản sao SHA-256...")
    original_count = len(records)
    records, excluded = prepare_unique_records(records)
    conflict_files = sum(row["status"] == "label_conflict" for row in excluded)
    duplicate_files = sum(row["status"] == "exact_duplicate" for row in excluded)
    write_excluded_report(args.excluded_report, excluded)
    print(f"  Ảnh đầu vào: {original_count}")
    print(f"  Loại do xung đột nhãn: {conflict_files}")
    print(f"  Loại bản sao exact cùng nhãn: {duplicate_files}")
    print(f"  Ảnh còn lại để chia: {len(records)}")
    print(f"  Báo cáo loại trừ: {args.excluded_report}")

    print("\n[5/6] Gom pHash trong cùng class...")
    clusters = cluster_records(records, args.threshold)
    cross_class = sum(
        1 for cluster in clusters if len({record["class"] for record in cluster}) > 1
    )
    label_conflicts = sum(cluster_has_label_conflict(cluster) for cluster in clusters)
    largest = max(map(len, clusters), default=0)
    large_clusters = [
        cluster for cluster in clusters if len(cluster) > args.max_cluster_size
    ]
    print(f"  Clusters: {len(clusters)}")
    print(f"  Cluster lớn nhất: {largest} ảnh")
    print(f"  Cluster chứa nhiều class: {cross_class}")
    print(f"  Xung đột nhãn exact SHA-256: {label_conflicts}")
    print(
        f"  Cluster vượt {args.max_cluster_size} ảnh: {len(large_clusters)}"
    )

    print("\n[6/6] Chia nguyên cluster vào train/valid/test...")
    plan, targets, forced_train = split_clusters(
        clusters,
        ratios,
        args.seed,
        max_eval_cluster_fraction=args.max_eval_cluster_fraction,
    )
    print_plan(plan, clusters, targets)
    assigned_split = cluster_split_map(plan)
    large_eval_clusters = [
        cluster
        for cluster in large_clusters
        if assigned_split[cluster[0]["path"]] != "train"
    ]
    print(f"Cluster được ép vào train để bảo vệ valid/test: {forced_train}")
    print(f"Cluster > {args.max_cluster_size} còn trong valid/test: {len(large_eval_clusters)}")
    write_audit_reports(
        clusters,
        plan,
        summary_path=args.audit_summary,
        members_path=args.audit_members,
    )
    print(f"\nAudit summary: {args.audit_summary}")
    print(f"Audit members: {args.audit_members}")

    if args.dry_run:
        print("\nDRY-RUN: không copy ảnh. Cache hash và audit đã được cập nhật.")
        if large_eval_clusters:
            print(
                f"CẢNH BÁO: còn {len(large_eval_clusters)} cluster vượt "
                f"{args.max_cluster_size} ảnh trong valid/test."
            )
        if label_conflicts:
            print(
                f"LỖI: còn {label_conflicts} cluster xung đột nhãn sau lọc. "
                "Không được copy dataset này."
            )
        return

    if large_eval_clusters and not args.allow_large_clusters:
        raise SystemExit(
            f"Từ chối copy: còn {len(large_eval_clusters)} cluster vượt "
            f"{args.max_cluster_size} ảnh trong valid/test. "
            f"Xem {args.audit_summary} hoặc dùng "
            "--allow-large-clusters sau khi đã xác minh."
        )
    if label_conflicts:
        raise SystemExit(
            f"Từ chối copy: còn {label_conflicts} xung đột nhãn sau lọc. "
            f"Xem {args.audit_members}."
        )

    if output.exists() and any(output.iterdir()):
        raise SystemExit(
            f"Output đã tồn tại và không rỗng: {output}. "
            "Hãy dùng thư mục output mới để tránh trộn các lần split."
        )

    print(f"\nSẽ COPY {len(records)} ảnh vào {output}")
    print("Gõ 'SPLIT' để xác nhận: ", end="")
    if input().strip() != "SPLIT":
        print("Đã huỷ.")
        return

    used_destinations = set()
    copied = 0
    for split_name, split_records in plan.items():
        for record in split_records:
            destination = destination_for(
                record, output, split_name, used_destinations
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(record["path"], destination)
            copied += 1
            if copied % 5000 == 0:
                print(f"  ... đã copy {copied}/{len(records)} ảnh")

    print(f"\nĐã copy {copied} ảnh")
    print(f"Output: {output / 'train'}, {output / 'valid'}, {output / 'test'}")


if __name__ == "__main__":
    main()
