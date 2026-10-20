"""Trích một tập Food-101 cân bằng từ Hugging Face Parquet.

Script ghi nguyên bytes ảnh từ Parquet, không giải mã hay chỉnh sửa ảnh. Mặc định
chỉ đọc các shard validation để không phụ thuộc vào bộ train đang tải thiếu.
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def label_names(parquet_file):
    metadata = parquet_file.schema_arrow.metadata or {}
    raw = metadata.get(b"huggingface")
    if not raw:
        raise ValueError("Parquet không có metadata Hugging Face")
    payload = json.loads(raw)
    return payload["info"]["features"]["label"]["names"]


def image_suffix(source_path):
    suffix = Path(source_path or "").suffix.lower()
    return suffix if suffix in {".jpg", ".jpeg", ".png", ".webp"} else ".jpg"


def balanced_targets(labels, total, seed):
    if total < len(labels):
        raise ValueError(
            f"--count phải >= số class ({len(labels)}) để lấy mẫu cân bằng"
        )
    base, remainder = divmod(total, len(labels))
    ranked = sorted(
        labels,
        key=lambda label: hashlib.sha256(
            f"{seed}:remainder:{label}".encode("utf-8")
        ).digest(),
    )
    extra = set(ranked[:remainder])
    return {label: base + (label in extra) for label in labels}


def scan_candidates(shards, seed):
    candidates = defaultdict(list)
    names = None
    for shard in shards:
        parquet_file = pq.ParquetFile(shard)
        current_names = label_names(parquet_file)
        if names is None:
            names = current_names
        elif names != current_names:
            raise ValueError(f"Metadata label không đồng nhất: {shard}")

        row_offset = 0
        for batch in parquet_file.iter_batches(
            batch_size=2048,
            columns=["label", "image.path"],
        ):
            labels = batch.column("label").to_pylist()
            image_column = batch.column("image")
            paths = image_column.field("path").to_pylist()
            for local_index, (label_id, source_path) in enumerate(zip(labels, paths)):
                row_index = row_offset + local_index
                score = hashlib.sha256(
                    f"{seed}:{shard.name}:{row_index}".encode("utf-8")
                ).digest()
                candidates[label_id].append({
                    "score": score,
                    "shard": shard,
                    "row_index": row_index,
                    "source_path": source_path or "",
                })
            row_offset += batch.num_rows
    return names or [], candidates


def choose_candidates(names, candidates, count, seed):
    targets = balanced_targets(list(range(len(names))), count, seed)
    selected = {}
    for label_id, target in targets.items():
        available = sorted(candidates[label_id], key=lambda item: item["score"])
        if len(available) < target:
            raise ValueError(
                f"Class {names[label_id]} chỉ có {len(available)} ảnh, cần {target}"
            )
        for item in available[:target]:
            selected[(item["shard"], item["row_index"])] = {
                **item,
                "label_id": label_id,
                "label_name": names[label_id],
            }
    return selected


def ensure_empty_output(output):
    if output.exists() and any(output.rglob("*")):
        raise FileExistsError(
            f"Output không rỗng: {output}. Hãy dùng output mới hoặc xóa thủ công."
        )


def extract_selected(shards, selected, output, metadata_path):
    rows = []
    for shard in shards:
        parquet_file = pq.ParquetFile(shard)
        row_offset = 0
        for batch in parquet_file.iter_batches(batch_size=256, columns=["image"]):
            images = batch.column("image")
            image_bytes = images.field("bytes").to_pylist()
            image_paths = images.field("path").to_pylist()
            for local_index, (content, source_path) in enumerate(
                zip(image_bytes, image_paths)
            ):
                row_index = row_offset + local_index
                item = selected.get((shard, row_index))
                if item is None:
                    continue
                if content is None:
                    raise ValueError(f"Ảnh không có bytes: {shard.name} row {row_index}")
                suffix = image_suffix(source_path)
                filename = f"{shard.stem}_row{row_index:06d}{suffix}"
                destination = output / item["label_name"] / filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
                rows.append({
                    "output_path": destination.as_posix(),
                    "dataset": "ethz/food101",
                    "source_shard": shard.name,
                    "source_row": row_index,
                    "source_image_path": source_path or "",
                    "label_id": item["label_id"],
                    "label_name": item["label_name"],
                })
            row_offset += batch.num_rows

    rows.sort(key=lambda row: row["output_path"])
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Trích Food-101 cân bằng từ Parquet làm UNSUPPORTED"
    )
    parser.add_argument("--parquet-root", type=Path, default=Path("data_negative/_parquet"))
    parser.add_argument(
        "--pattern",
        default="validation-*.parquet",
        help="Glob shard; mặc định dùng validation hoàn chỉnh",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data_negative/_staging/UNSUPPORTED/food101"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data_negative/metadata/food101_selected.csv"),
    )
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.count < 1:
        raise SystemExit("--count phải lớn hơn 0")
    shards = sorted(args.parquet_root.glob(args.pattern))
    if not shards:
        raise SystemExit(f"Không tìm thấy shard: {args.parquet_root / args.pattern}")

    print("=" * 72)
    print("TRÍCH FOOD-101 PARQUET CHO LỚP UNSUPPORTED")
    print(f"  Shards : {len(shards)} ({args.pattern})")
    print(f"  Count  : {args.count}")
    print(f"  Output : {args.output}")
    print(f"  Seed   : {args.seed}")
    print("=" * 72)

    try:
        names, candidates = scan_candidates(shards, args.seed)
        selected = choose_candidates(names, candidates, args.count, args.seed)
    except (KeyError, ValueError) as error:
        raise SystemExit(str(error)) from error

    selected_counts = Counter(item["label_name"] for item in selected.values())
    print(f"Đã đọc metadata: {sum(map(len, candidates.values()))} record")
    print(
        f"Kế hoạch: {len(selected)} ảnh / {len(selected_counts)} class; "
        f"mỗi class {min(selected_counts.values())}–{max(selected_counts.values())} ảnh"
    )
    if args.dry_run:
        print("DRY-RUN: không trích ảnh và không ghi metadata.")
        return

    try:
        ensure_empty_output(args.output)
    except FileExistsError as error:
        raise SystemExit(str(error)) from error
    rows = extract_selected(shards, selected, args.output, args.metadata)
    if len(rows) != len(selected):
        raise SystemExit(f"Chỉ trích được {len(rows)}/{len(selected)} ảnh")
    print(f"Đã trích: {len(rows)} ảnh")
    print(f"Metadata: {args.metadata}")


if __name__ == "__main__":
    main()
