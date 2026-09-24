import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

TRIALS = (
    ("mobilenet_v3_large", "fine_tune"),
    ("mobilenet_v3_small", "fine_tune"),
    ("mobilenet_v3_small", "head_only"),
)


def newest_run(previous):
    candidates = set(Path("runs").glob("*/summary.json")) - previous
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def main():
    parser = argparse.ArgumentParser(description="Benchmark công bằng các biến thể MobileNetV3")
    parser.add_argument("task", choices=("router", "freshness"))
    parser.add_argument("--category")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.task == "freshness" and not args.category:
        raise SystemExit("Task freshness yêu cầu --category")

    results = []
    for index, (model, method) in enumerate(TRIALS, start=1):
        print(f"\n=== Trial {index}/{len(TRIALS)}: {model}, {method} ===")
        previous = set(Path("runs").glob("*/summary.json"))
        command = [
            sys.executable,
            "train.py",
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--workers", str(args.workers),
            "--seed", str(args.seed),
            "--model", model,
            "--method", method,
            "--optimizer", "adamw",
            "--scheduler", "plateau",
            "--learning-rate", "5e-4",
            "--weight-decay", "1e-4",
            "--label-smoothing", "0.05",
            "--run-name", f"benchmark_{index}_{model}_{method}",
            args.task,
        ]
        if args.category:
            command.extend(["--category", args.category])
        completed = subprocess.run(command)
        if completed.returncode != 0:
            print(f"Trial {index} failed; continuing.")
            continue
        summary_path = newest_run(previous)
        if not summary_path:
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        results.append({
            "model": model,
            "method": method,
            "epochs": args.epochs,
            "seed": args.seed,
            "best_valid_loss": summary["best_valid_loss"],
            "best_valid_accuracy": summary["best_valid_accuracy"],
            "test_accuracy": summary["test_accuracy"],
            "macro_f1": summary["macro_f1"],
            "run": summary_path.parent.name,
        })

    if not results:
        raise SystemExit("Không có trial nào hoàn thành.")
    output = Path("runs") / f"mobilenet_benchmark_{datetime.now():%Y%m%d_%H%M%S}.csv"
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Đã lưu leaderboard: {output}")


if __name__ == "__main__":
    main()
