from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def evaluate(weights: Path, data: Path, output: Path, imgsz: int = 640, batch: int = 1) -> dict:
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install CUDA PyTorch and project dependencies before evaluation") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    output.mkdir(parents=True, exist_ok=True)
    result = YOLO(str(weights)).val(
        data=str(data), split="test", imgsz=imgsz, batch=batch, device=0,
        plots=True, project=str(output), name="ultralytics", exist_ok=True,
    )
    summary = {
        "weights": str(weights.resolve()),
        "data": str(data.resolve()),
        "precision": float(result.box.mp),
        "recall": float(result.box.mr),
        "map50": float(result.box.map50),
        "map50_95": float(result.box.map),
    }
    (output / "test_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output / "test_per_class.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "class_name", "precision", "recall", "map50", "map50_95"])
        for index, class_id in enumerate(result.box.ap_class_index):
            values = [float(value) for value in result.box.class_result(index)]
            writer.writerow([int(class_id), result.names[int(class_id)], *values])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate YOLO11l once on the frozen test split")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("datasets/detection/data.yaml"))
    parser.add_argument("--output", type=Path, default=Path("reports/detection_test_e30"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.weights, args.data, args.output, args.imgsz, args.batch), indent=2))


if __name__ == "__main__":
    main()
