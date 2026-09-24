"""
Script cào ảnh bổ sung cho dataset.
Cấu trúc đích: data/{train,valid}/<group>/<LEVEL>/<food>/

Usage:
    python scripts/crawl_images.py --plan crawl_plan.json
    python scripts/crawl_images.py --auto-suggest
"""
import argparse
import json
import os
import time
from pathlib import Path

from icrawler.builtin import BingImageCrawler


LEVELS = ["FRESH", "HALF-FRESH", "SPOILED"]

# Từ khóa cho từng level, dùng {food} làm placeholder
KEYWORD_TEMPLATES = {
    "FRESH": [
        "fresh {food}",
        "raw {food}",
        "{food} close up fresh",
    ],
    "HALF-FRESH": [
        "overripe {food}",
        "{food} browning",
        "{food} wilting",
        "slightly spoiled {food}",
    ],
    "SPOILED": [
        "rotten {food}",
        "moldy {food}",
        "spoiled {food}",
        "decayed {food}",
    ],
}

# Mapping group → tên tiếng Việt (nếu muốn cào ảnh VN)
VN_KEYWORDS = {
    "FRESH": "{food} tươi",
    "HALF-FRESH": "{food} héo",
    "SPOILED": "{food} hư thối",
}


def crawl_one(keyword: str, out_dir: str, max_num: int = 100):
    """Cào ảnh từ Bing vào out_dir."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    existing = len([f for f in os.listdir(out_dir)
                    if f.lower().endswith(('.jpg', '.png', '.jpeg', '.webp'))])
    if existing >= max_num:
        print(f"  ⏭️  Đã đủ ({existing} ảnh), bỏ qua: {out_dir}")
        return

    print(f"  🔍 Cào '{keyword}' → {out_dir}")
    crawler = BingImageCrawler(
        storage={"root_dir": out_dir},
        feeder_threads=1,
        parser_threads=2,
        downloader_threads=4,
        log_level=50,
    )
    try:
        crawler.crawl(keyword=keyword, max_num=max_num - existing,
                      min_size=(300, 300))
        time.sleep(1)  # tránh rate limit
    except Exception as e:
        print(f"  ❌ Lỗi: {e}")


def crawl_plan(plan: dict, data_root: str, split: str = "train"):
    """
    plan = {
        "cu": {
            "carrot": {"FRESH": 100, "HALF-FRESH": 50, "SPOILED": 80},
            "potato": {"FRESH": 100, "HALF-FRESH": 50, "SPOILED": 80},
        },
        ...
    }
    """
    for group, foods in plan.items():
        for food, level_targets in foods.items():
            for level, target in level_targets.items():
                out_dir = os.path.join(data_root, split, group, level, food)
                for tpl in KEYWORD_TEMPLATES[level][:1]:  # dùng 1 keyword chính
                    keyword = tpl.format(food=food.replace("_", " "))
                    crawl_one(keyword, out_dir, max_num=target)


def auto_suggest(data_root: str, min_per_class: int = 100):
    """Quét dataset và in ra class nào đang thiếu ảnh."""
    print("=" * 60)
    print("KIỂM TRA DATASET")
    print("=" * 60)

    for split in ["train", "valid"]:
        split_path = os.path.join(data_root, split)
        if not os.path.isdir(split_path):
            continue

        print(f"\n[{split}]")
        for group in sorted(os.listdir(split_path)):
            gp = os.path.join(split_path, group)
            if not os.path.isdir(gp):
                continue
            print(f"\n  {group}/")

            for level in LEVELS:
                lp = os.path.join(gp, level)
                if not os.path.isdir(lp):
                    continue
                for food in sorted(os.listdir(lp)):
                    fp = os.path.join(lp, food)
                    if not os.path.isdir(fp):
                        continue
                    n = len([f for f in os.listdir(fp)
                             if f.lower().endswith(('.jpg', '.png', '.jpeg', '.webp'))])
                    flag = "  ⚠️ THIẾU" if n < min_per_class else ""
                    print(f"    {level:12s}/{food:15s}: {n:4d}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--split", default="train", choices=["train", "valid"])
    ap.add_argument("--plan", help="File JSON chứa crawl plan")
    ap.add_argument("--auto-suggest", action="store_true",
                    help="Chỉ in báo cáo class thiếu ảnh")
    ap.add_argument("--min-per-class", type=int, default=100)
    args = ap.parse_args()

    if args.auto_suggest:
        auto_suggest(args.data_root, args.min_per_class)
        return

    if not args.plan:
        print("Cần --plan hoặc --auto-suggest")
        return

    with open(args.plan, "r", encoding="utf-8") as f:
        plan = json.load(f)

    crawl_plan(plan, args.data_root, args.split)


if __name__ == "__main__":
    main()