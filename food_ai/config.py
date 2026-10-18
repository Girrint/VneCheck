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

IMAGE_SIZE = 224
EVAL_RESIZE_SIZE = 232
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DATA_DIR = Path("data_split")
MODEL_DIR = Path("models")
ROUTER_MODEL = MODEL_DIR / "router.pth"


def freshness_model(category: str) -> Path:
    return MODEL_DIR / "freshness" / f"{category}.pth"

