from pathlib import Path

CATEGORIES = ("cu", "gia_cam", "gia_suc", "qua", "rau")
CATEGORY_LABELS = {
    "gia_suc": "Gia súc",
    "gia_cam": "Gia cầm",
    "rau": "Rau",
    "cu": "Củ",
    "qua": "Quả",
}

FRESHNESS_CLASSES = ("FRESH", "HALF-FRESH", "SPOILED")
FRESHNESS_LABELS = {
    "FRESH": "Tươi",
    "HALF-FRESH": "Kém tươi",
    "SPOILED": "Hư hỏng",
}
FRESHNESS_ANCHORS = {
    "FRESH": 10.0,
    "HALF-FRESH": 5.5,
    "SPOILED": 1.0,
}

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
ROUTER_MODEL = MODEL_DIR / "router.pth"


def freshness_model(category: str) -> Path:
    return MODEL_DIR / "freshness" / f"{category}.pth"

