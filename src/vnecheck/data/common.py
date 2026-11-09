from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def serialize(self, class_id: int | None = None) -> str:
        cid = self.class_id if class_id is None else class_id
        return (
            f"{cid} {self.x_center:.10f} {self.y_center:.10f} "
            f"{self.width:.10f} {self.height:.10f}"
        )


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_class_config(path: Path | None = None) -> dict:
    config_path = path or project_root() / "configs" / "classes.json"
    return json.loads(config_path.read_text(encoding="utf-8"))


def list_images(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def read_yolo_label(path: Path, max_class_id: int | None = None) -> list[YoloBox]:
    boxes: list[YoloBox] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 columns, got {len(parts)}")
        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid numeric value") from exc
        if class_id < 0 or (max_class_id is not None and class_id > max_class_id):
            raise ValueError(f"{path}:{line_number}: invalid class id {class_id}")
        if width <= 0 or height <= 0:
            raise ValueError(f"{path}:{line_number}: non-positive box size")
        tolerance = 1e-4
        if (
            x_center - width / 2 < -tolerance
            or y_center - height / 2 < -tolerance
            or x_center + width / 2 > 1 + tolerance
            or y_center + height / 2 > 1 + tolerance
        ):
            raise ValueError(f"{path}:{line_number}: bounding box is outside the image")
        boxes.append(YoloBox(class_id, x_center, y_center, width, height))
    return boxes


def source_group_id(filename: str) -> str:
    """Return the pre-Roboflow source identifier used to prevent split leakage."""
    stem = Path(filename).stem
    stem = re.sub(r"\.rf\.[0-9a-f]+$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_(?:jpg|jpeg|png)$", "", stem, flags=re.IGNORECASE)
    return stem.casefold()


def deterministic_split(group_id: str, train: int = 70, valid: int = 15) -> str:
    bucket = int(hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < train:
        return "train"
    if bucket < train + valid:
        return "valid"
    return "test"


def parse_filename_identity(filename: str) -> tuple[str | None, str | None]:
    """Parse explicit freshness and produce name from common source naming schemes."""
    stem = source_group_id(filename)
    names = (
        "bellpepper",
        "strawberry",
        "cucumber",
        "banana",
        "carrot",
        "orange",
        "potato",
        "tomato",
        "apple",
        "mango",
    )
    aliases = {"pepper": "bellpepper", "sweetpepper": "bellpepper", "capsicum": "bellpepper"}

    match = re.match(r"^(fresh|rotten)[_\- ]*([a-z]+)", stem, re.IGNORECASE)
    if match:
        freshness = match.group(1).casefold()
        produce = aliases.get(match.group(2).casefold(), match.group(2).casefold())
        return freshness, produce if produce in names else None

    for produce in names:
        match = re.match(rf"^{re.escape(produce)}[_\-]([fr])(?:[_\-]|$)", stem, re.IGNORECASE)
        if match:
            return ("fresh" if match.group(1).casefold() == "f" else "rotten"), produce
    return None, None
