import argparse
import json
import sys
from pathlib import Path

import torch

from food_ai.config import UNIFIED_MANIFEST, UNIFIED_MODEL
from food_ai.unified_data import create_unified_loaders
from food_ai.unified_loss import MaskedMultiTaskLoss
from food_ai.unified_model import load_unified_checkpoint
from train_unified import run_epoch


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Đánh giá unified model")
    parser.add_argument("--checkpoint", type=Path, default=UNIFIED_MODEL)
    parser.add_argument("--manifest", type=Path, default=UNIFIED_MANIFEST)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_unified_checkpoint(args.checkpoint, device)
    loaders, datasets = create_unified_loaders(
        args.manifest, args.batch_size, args.workers
    )
    config = checkpoint.get("config", {})
    criterion = MaskedMultiTaskLoss(
        food_weight=float(config.get("effective_food_loss_weight", 1.0)),
        category_weight=float(config.get("category_loss_weight", 1.0)),
        freshness_weight=float(config.get("freshness_loss_weight", 1.0)),
        label_smoothing=float(config.get("label_smoothing", 0.05)),
    )
    result = run_epoch(model, loaders["test"], criterion)
    result.update({
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "samples": len(datasets["test"]),
        "food_gate_ready": bool(checkpoint.get("food_gate_ready", False)),
        "device": str(device),
    })
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
