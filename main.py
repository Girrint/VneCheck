import subprocess
import sys
import json
from pathlib import Path

import torch

from food_ai.config import CATEGORIES, CATEGORY_LABELS, DATA_DIR, FRESHNESS_CLASSES
from food_ai.model import MODEL_NAMES, TRAINING_METHODS

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def configure_console():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")


def image_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(
        1
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )


def run(command: list[str]) -> None:
    print("\nRunning:", " ".join(command), "\n")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        print(f"\nCommand failed with exit code {error.returncode}.")
    except KeyboardInterrupt:
        print("\nCommand interrupted.")


def ask_positive_int(label: str, default: int) -> int:
    while True:
        value = input(f"{label} [{default}]: ").strip()
        if not value:
            return default
        if value.isdigit() and int(value) > 0:
            return int(value)
        print("Please enter a positive integer.")


def ask_positive_float(label: str, default: float) -> float:
    while True:
        value = input(f"{label} [{default:g}]: ").strip()
        if not value:
            return default
        try:
            parsed = float(value)
            if parsed > 0:
                return parsed
        except ValueError:
            pass
        print("Please enter a positive number.")


def choose(label: str, options: tuple[str, ...], default: str) -> str:
    print(f"\n{label}:")
    for index, option in enumerate(options, start=1):
        suffix = " (default)" if option == default else ""
        print(f"  {index}. {option}{suffix}")
    value = input("Select: ").strip()
    if not value:
        return default
    if value.isdigit() and 1 <= int(value) <= len(options):
        return options[int(value) - 1]
    print(f"Invalid selection; using {default}.")
    return default


def training_options() -> list[str]:
    epochs = ask_positive_int("Epochs", 15)
    batch_size = ask_positive_int("Batch size", 16)
    workers = ask_positive_int("Data workers", 4)
    model = choose("Model architecture", MODEL_NAMES, "mobilenet_v3_large")
    method = choose("Training method", TRAINING_METHODS, "fine_tune")
    optimizer = choose("Optimizer", ("adamw", "sgd"), "adamw")
    scheduler = choose("LR scheduler", ("plateau", "cosine"), "plateau")
    learning_rate = ask_positive_float("Learning rate", 5e-4)
    weight_decay = ask_positive_float("Weight decay", 1e-4)
    return [
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--workers",
        str(workers),
        "--model",
        model,
        "--method",
        method,
        "--optimizer",
        optimizer,
        "--scheduler",
        scheduler,
        "--learning-rate",
        str(learning_rate),
        "--weight-decay",
        str(weight_decay),
    ]


def choose_category() -> str | None:
    print("\nFood categories:")
    for index, category in enumerate(CATEGORIES, start=1):
        print(f"  {index}. {CATEGORY_LABELS[category]} ({category})")
    value = input("Select category: ").strip()
    if value.isdigit() and 1 <= int(value) <= len(CATEGORIES):
        return CATEGORIES[int(value) - 1]
    print("Invalid category.")
    return None


def train_router() -> None:
    run([sys.executable, "train.py", *training_options(), "router"])


def train_expert() -> None:
    category = choose_category()
    if category:
        run([
            sys.executable,
            "train.py",
            *training_options(),
            "freshness",
            "--category",
            category,
        ])


def predict_image() -> None:
    raw_path = input("Image path: ").strip().strip('"')
    image_path = Path(raw_path)
    if not image_path.is_file():
        print(f"Image not found: {image_path}")
        return
    run([sys.executable, "predict.py", str(image_path)])


def dataset_status() -> None:
    print("\nDataset status")
    print("=" * 72)
    for category in CATEGORIES:
        print(f"\n{CATEGORY_LABELS[category]} ({category})")
        for freshness in FRESHNESS_CLASSES:
            train_count = image_count(DATA_DIR / "train" / category / freshness)
            valid_count = image_count(DATA_DIR / "valid" / category / freshness)
            test_count = image_count(DATA_DIR / "test" / category / freshness)
            state = "OK" if train_count and valid_count and test_count else "MISSING"
            print(
                f"  {freshness:<11} train={train_count:<7} "
                f"valid={valid_count:<7} test={test_count:<7} {state}"
            )


def tune_parameters() -> None:
    category = choose_category()
    if not category:
        return
    epochs = ask_positive_int("Epochs per trial", 3)
    run([
        sys.executable,
        "tune.py",
        "--task",
        "freshness",
        "--category",
        category,
        "--epochs",
        str(epochs),
    ])


def show_results() -> None:
    run_root = Path("runs")
    summaries = []
    for path in run_root.glob("*/summary.json") if run_root.exists() else []:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            config = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
            summaries.append((path.parent.name, config, payload))
        except (OSError, ValueError):
            continue

    if not summaries:
        print("No completed training runs found.")
        return
    print("\nCompleted runs (newest first)")
    print("=" * 110)
    for name, config, summary in sorted(summaries, reverse=True)[:20]:
        print(
            f"{name} | {config['model']}/{config['method']} | "
            f"test={summary['test_accuracy']:.2f}% | "
            f"F1={summary['macro_f1']:.4f} | best epoch={summary['best_epoch']}"
        )


def show_help() -> None:
    print(
        """
HELP
====

1. Train seven-class router
   Trains five supported food categories plus UNSUPPORTED and NON_FOOD.
   Supported images come from data_split; negative images come from
   data_negative. Negative predictions stop before the freshness experts.

2. Train freshness expert
   Trains one category-specific model. Its outputs are inferred from the
   freshness directories that actually exist for that category.

3. Predict an image
   Runs the router first, then the matching freshness expert. All required
   checkpoints must already exist under models/.

4. Dataset status
   Counts supported image files for every category, freshness class, and
   split. MISSING means that train, validation, or test data is absent.

5. Hyperparameter tuning
   Runs a short, controlled grid of model architectures, transfer-learning
   methods, and learning rates. Each trial is stored separately.

6. Training results
   Lists completed runs with model, method, test accuracy, macro F1, and the
   best epoch.

Training settings
-----------------
Epochs controls the maximum training passes. Batch size controls GPU memory
usage; use 8 or 16 for a 4 GB GPU. Data workers controls parallel image
loading; 4 is a reasonable Windows default.

Output models
-------------
Router: models/router.pth
Experts: models/freshness/<category>.pth
Experiments: runs/<timestamp>_<run-name>/

The estimated freshness score is image-based and must not be treated as a
food-safety certification.
""".strip()
    )


def device_summary() -> str:
    if not torch.cuda.is_available():
        return f"CPU only | PyTorch {torch.__version__} | CUDA unavailable"

    properties = torch.cuda.get_device_properties(0)
    vram_gb = properties.total_memory / (1024**3)
    return (
        f"GPU: {properties.name} | CUDA {torch.version.cuda} | "
        f"VRAM {vram_gb:.1f} GB"
    )


def print_menu() -> None:
    print("\n")
    print(
        """
Food Freshness AI
=================
""".strip()
    )
    print(f"Device: {device_summary()}")
    print(
        """

1. Train 7-class router
2. Train freshness expert
3. Predict an image
4. Dataset status
5. Hyperparameter tuning
6. Training results
H. Help (English)
Q. Quit
""".rstrip()
    )


def main() -> None:
    configure_console()
    actions = {
        "1": train_router,
        "2": train_expert,
        "3": predict_image,
        "4": dataset_status,
        "5": tune_parameters,
        "6": show_results,
        "h": show_help,
        "help": show_help,
    }

    while True:
        print_menu()
        choice = input("Select an option: ").strip().lower()
        if choice in {"q", "quit", "exit"}:
            print("Goodbye.")
            return
        action = actions.get(choice)
        if action:
            action()
        else:
            print("Invalid option. Enter H for help.")


if __name__ == "__main__":
    main()
