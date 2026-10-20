import csv
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from food_ai.config import CATEGORIES, FOOD_STATUS_CLASSES, FRESHNESS_CLASSES
from food_ai.data import build_transforms


IGNORE_INDEX = -100
CATEGORY_TO_INDEX = {name: index for index, name in enumerate(CATEGORIES)}
FRESHNESS_TO_INDEX = {
    name: index for index, name in enumerate(FRESHNESS_CLASSES)
}
STATUS_TO_INDEX = {
    name: index for index, name in enumerate(FOOD_STATUS_CLASSES)
}


class UnifiedFoodDataset(Dataset):
    """Dataset đa nhiệm đọc nhãn từ manifest, không thay đổi dữ liệu legacy."""

    REQUIRED_COLUMNS = {
        "path",
        "split",
        "food_status",
        "category",
        "freshness",
    }

    def __init__(self, manifest, split, transform):
        self.manifest = Path(manifest)
        self.split = split
        self.transform = transform
        self.records = self._read_records()
        if not self.records:
            raise ValueError(f"Manifest không có mẫu cho split '{split}'")
        self.status_counts = Counter(row["food_status"] for row in self.records)

    def _read_records(self):
        if not self.manifest.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy {self.manifest}. Chạy "
                "python scripts/build_unified_manifest.py trước."
            )

        with self.manifest.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            missing = self.REQUIRED_COLUMNS - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"Manifest thiếu cột: {sorted(missing)}")
            rows = [row for row in reader if row["split"] == self.split]

        for row in rows:
            status = row["food_status"]
            if status not in STATUS_TO_INDEX:
                raise ValueError(f"food_status không hợp lệ: {status}")
            if status == "SUPPORTED":
                if row["category"] not in CATEGORY_TO_INDEX:
                    raise ValueError(f"category không hợp lệ: {row['category']}")
                if row["freshness"] not in FRESHNESS_TO_INDEX:
                    raise ValueError(f"freshness không hợp lệ: {row['freshness']}")
            elif row["category"] or row["freshness"]:
                raise ValueError(
                    "Ảnh UNSUPPORTED/NON_FOOD không được có category/freshness"
                )
        return rows

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        path = Path(record["path"])
        with Image.open(path) as image:
            tensor = self.transform(image)

        status = STATUS_TO_INDEX[record["food_status"]]
        category = CATEGORY_TO_INDEX.get(record["category"], IGNORE_INDEX)
        freshness = FRESHNESS_TO_INDEX.get(
            record["freshness"], IGNORE_INDEX
        )
        return tensor, {
            "food_status": torch.tensor(status, dtype=torch.long),
            "category": torch.tensor(category, dtype=torch.long),
            "freshness": torch.tensor(freshness, dtype=torch.long),
        }


def create_unified_loaders(manifest, batch_size, workers):
    train_transform, eval_transform = build_transforms()
    datasets = {
        "train": UnifiedFoodDataset(manifest, "train", train_transform),
        "valid": UnifiedFoodDataset(manifest, "valid", eval_transform),
        "test": UnifiedFoodDataset(manifest, "test", eval_transform),
    }
    use_cuda = torch.cuda.is_available()
    options = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": use_cuda,
    }
    if workers > 0:
        options.update(persistent_workers=True, prefetch_factor=2)
    loaders = {
        split: DataLoader(
            dataset,
            shuffle=split == "train",
            drop_last=split == "train" and use_cuda,
            **options,
        )
        for split, dataset in datasets.items()
    }
    return loaders, datasets
