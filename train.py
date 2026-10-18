import argparse
import os
import random
import shutil
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from food_ai.config import (
    CATEGORIES,
    DATA_DIR,
    FRESHNESS_CLASSES,
    MODEL_DIR,
    freshness_model,
)
from food_ai.data import create_loaders, preprocessing_metadata
from food_ai.experiment import (
    ExperimentTracker,
    classification_report,
    create_run_directory,
)
from food_ai.model import (
    MODEL_NAMES,
    TRAINING_METHODS,
    configure_training_method,
    create_model,
    trainable_parameters,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = DEVICE.type == "cuda"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)

def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện AI phân tầng")
    parser.add_argument("--epochs", type=int, default=int(os.getenv("EPOCHS", 15)))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", 16)))
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("NUM_WORKERS", min(4, os.cpu_count() or 1))),
    )
    parser.add_argument("--model", choices=MODEL_NAMES, default="mobilenet_v3_large")
    parser.add_argument("--method", choices=TRAINING_METHODS, default="fine_tune")
    parser.add_argument("--optimizer", choices=("adamw", "sgd"), default="adamw")
    parser.add_argument("--scheduler", choices=("plateau", "cosine"), default="plateau")
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name")
    parser.add_argument(
        "--resume",
        help="Resume từ last_checkpoint.pth hoặc best_model.pth; epochs là tổng mục tiêu.",
    )

    subparsers = parser.add_subparsers(dest="task", required=True)
    subparsers.add_parser("router", help="Huấn luyện nhận diện 5 nhóm")
    expert = subparsers.add_parser("freshness", help="Huấn luyện expert độ tươi")
    expert.add_argument("--category", required=True, choices=CATEGORIES)
    return parser.parse_args()


def resolve_task(args):
    if args.task == "router":
        return (
            DATA_DIR / "train",
            DATA_DIR / "valid",
            DATA_DIR / "test",
            MODEL_DIR / "router.pth",
        )
    return (
        DATA_DIR / "train" / args.category,
        DATA_DIR / "valid" / args.category,
        DATA_DIR / "test" / args.category,
        freshness_model(args.category),
    )


def move_batch(images, labels):
    memory_format = torch.channels_last if USE_CUDA else torch.preserve_format
    images = images.to(DEVICE, non_blocking=USE_CUDA, memory_format=memory_format)
    labels = labels.to(DEVICE, non_blocking=USE_CUDA)
    return images, labels


def set_training_mode(model, training):
    model.train(training)
    if training and getattr(model, "training_method", None) == "head_only":
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()


def run_epoch(model, loader, criterion, num_classes, optimizer=None, scaler=None):
    training = optimizer is not None
    set_training_mode(model, training)
    loss_sum = correct = count = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    context = nullcontext() if training else torch.inference_mode()

    with context:
        for images, labels in loader:
            images, labels = move_batch(images, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)

            amp = torch.autocast("cuda", dtype=torch.float16) if USE_CUDA else nullcontext()
            with amp:
                outputs = model(images)
                loss = criterion(outputs, labels)

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            predictions = outputs.argmax(1)
            size = labels.size(0)
            loss_sum += loss.detach().item() * size
            correct += predictions.eq(labels).sum().item()
            count += size
            encoded = (labels * num_classes + predictions).detach().cpu()
            confusion += torch.bincount(encoded, minlength=num_classes**2).reshape(
                num_classes, num_classes
            )

    return loss_sum / count, 100 * correct / count, confusion


def build_optimizer(args, parameters):
    options = {"lr": args.learning_rate, "weight_decay": args.weight_decay}
    if args.optimizer == "sgd":
        return optim.SGD(parameters, momentum=0.9, nesterov=True, **options)
    if USE_CUDA:
        options["fused"] = True
    try:
        return optim.AdamW(parameters, **options)
    except (TypeError, RuntimeError):
        options.pop("fused", None)
        return optim.AdamW(parameters, **options)


def validate_args(args):
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise SystemExit("Epoch, batch size phải dương; workers không được âm.")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise SystemExit("Learning rate phải dương; weight decay không được âm.")
    if not 0 <= args.label_smoothing < 1:
        raise SystemExit("Label smoothing phải nằm trong [0, 1).")


def main():
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if USE_CUDA:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    train_root, valid_root, test_root, output_path = resolve_task(args)
    try:
        train_loader, valid_loader, test_loader, classes = create_loaders(
            train_root,
            valid_root,
            test_root,
            args.batch_size,
            args.workers,
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Dữ liệu chưa sẵn sàng: {error}") from error

    if args.task == "router":
        missing = sorted(set(CATEGORIES) - set(classes))
        if missing:
            print(f"Cảnh báo: router thiếu nhóm dữ liệu {missing}; train theo các nhóm hiện có.")
    if args.task == "freshness":
        unknown = sorted(set(classes) - set(FRESHNESS_CLASSES))
        if unknown:
            raise SystemExit(f"Nhãn độ tươi không hợp lệ: {unknown}")
        if len(classes) < 2:
            raise SystemExit("Expert cần ít nhất hai mức độ tươi có dữ liệu.")

    category = getattr(args, "category", None)
    resume_path = Path(args.resume).resolve() if args.resume else None
    run_dir = (
        resume_path.parent
        if resume_path is not None
        else create_run_directory(args.task, category, args.model, args.run_name)
    )
    device_name = torch.cuda.get_device_name(0) if USE_CUDA else "CPU"
    config = {
        **vars(args),
        "classes": classes,
        "train_root": str(train_root),
        "valid_root": str(valid_root),
        "test_root": str(test_root),
        "train_images": len(train_loader.dataset),
        "valid_images": len(valid_loader.dataset),
        "test_images": len(test_loader.dataset),
        "preprocessing": preprocessing_metadata(),
        "device": device_name,
        "python": sys.version,
        "pytorch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda,
    }
    print(f"Run: {run_dir}")
    print(f"Task: {args.task} | lớp: {classes}")
    print(
        f"Train: {len(train_loader.dataset)} | Valid: {len(valid_loader.dataset)} | "
        f"Test: {len(test_loader.dataset)}"
    )
    print(
        f"Model: {args.model} | Method: {args.method} | "
        f"Optimizer: {args.optimizer} | LR: {args.learning_rate:g}"
    )
    print(f"Thiết bị: {device_name}")

    model = create_model(len(classes), args.model, pretrained=True)
    model = configure_training_method(model, args.model, args.method).to(DEVICE)
    resume_checkpoint = None
    if resume_path is not None:
        resume_checkpoint = torch.load(
            resume_path, map_location=DEVICE, weights_only=False
        )
        model.load_state_dict(resume_checkpoint["state_dict"])
        print(f"Resume checkpoint: {resume_path}")
    if USE_CUDA:
        model = model.to(memory_format=torch.channels_last)

    parameters = trainable_parameters(model)
    config["total_parameters"] = sum(parameter.numel() for parameter in model.parameters())
    config["trainable_parameters"] = sum(parameter.numel() for parameter in parameters)
    tracker = ExperimentTracker(run_dir, config, resume=resume_path is not None)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = build_optimizer(args, parameters)
    if args.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=1
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_CUDA)

    start_epoch = 1
    best_loss = float("inf")
    best_valid_accuracy = 0.0
    best_epoch = 0
    stale_epochs = 0
    if resume_checkpoint is not None:
        if resume_checkpoint.get("optimizer_state_dict"):
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        if resume_checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
        if resume_checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(resume_checkpoint["scaler_state_dict"])
        start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
        best_loss = float(resume_checkpoint.get("best_loss", best_loss))
        best_valid_accuracy = float(
            resume_checkpoint.get("best_valid_accuracy", best_valid_accuracy)
        )
        best_epoch = int(resume_checkpoint.get("best_epoch", best_epoch))
        stale_epochs = int(resume_checkpoint.get("stale_epochs", stale_epochs))
        if resume_checkpoint.get("rng_state") is not None:
            random.setstate(resume_checkpoint["rng_state"])
        if resume_checkpoint.get("torch_rng_state") is not None:
            torch.set_rng_state(resume_checkpoint["torch_rng_state"])
        if USE_CUDA and resume_checkpoint.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(resume_checkpoint["cuda_rng_state_all"])
        print(f"Continuing at epoch {start_epoch}/{args.epochs}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_checkpoint = run_dir / "best_model.pth"
    last_checkpoint = run_dir / "last_checkpoint.pth"

    def checkpoint_payload(epoch):
        return {
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
            "best_loss": best_loss,
            "best_valid_accuracy": best_valid_accuracy,
            "best_epoch": best_epoch,
            "stale_epochs": stale_epochs,
            "classes": classes,
            "task": args.task,
            "category": category,
            "model_name": args.model,
            "training_method": args.method,
            "config": config,
            "rng_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if USE_CUDA else None,
        }

    if not last_checkpoint.exists():
        atomic_torch_save(checkpoint_payload(start_epoch - 1), last_checkpoint)

    for epoch in range(start_epoch, args.epochs + 1):
        started = time.perf_counter()
        train_loss, train_acc, _ = run_epoch(
            model, train_loader, criterion, len(classes), optimizer, scaler
        )
        valid_loss, valid_acc, _ = run_epoch(
            model, valid_loader, criterion, len(classes)
        )

        improved = valid_loss < best_loss
        if improved:
            best_loss = valid_loss
            best_valid_accuracy = valid_acc
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1

        duration = time.perf_counter() - started
        current_lr = optimizer.param_groups[0]["lr"]
        tracker.log_epoch({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_accuracy": round(train_acc, 4),
            "valid_loss": round(valid_loss, 6),
            "valid_accuracy": round(valid_acc, 4),
            "learning_rate": current_lr,
            "duration_seconds": round(duration, 3),
            "saved": improved,
        })

        if args.scheduler == "plateau":
            scheduler.step(valid_loss)
        else:
            scheduler.step()

        checkpoint = checkpoint_payload(epoch)
        atomic_torch_save(checkpoint, last_checkpoint)
        if improved:
            atomic_torch_save(checkpoint, run_checkpoint)
            shutil.copy2(run_checkpoint, output_path)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train {train_acc:.2f}%/{train_loss:.4f} | "
            f"valid {valid_acc:.2f}%/{valid_loss:.4f} | "
            f"lr {current_lr:.2e} | {duration:.1f}s"
            + (" | saved" if improved else ""),
            flush=True,
        )
        if stale_epochs >= 3:
            print("Early stopping: 3 epoch không cải thiện.")
            break

    evaluation_checkpoint = run_checkpoint if run_checkpoint.exists() else last_checkpoint
    checkpoint = torch.load(evaluation_checkpoint, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    test_loss, test_acc, confusion = run_epoch(
        model, test_loader, criterion, len(classes)
    )
    report = classification_report(confusion, classes)
    summary = {
        "best_epoch": best_epoch,
        "best_valid_loss": round(best_loss, 6),
        "best_valid_accuracy": round(best_valid_accuracy, 4),
        "test_loss": round(test_loss, 6),
        "test_accuracy": round(test_acc, 4),
        **report,
        "deployed_checkpoint": str(output_path),
        "run_checkpoint": str(run_checkpoint),
        "last_checkpoint": str(last_checkpoint),
    }
    tracker.save_summary(summary)

    print(
        f"Test cuối | accuracy {test_acc:.2f}% | loss {test_loss:.4f} | "
        f"macro F1 {report['macro_f1']:.4f}"
    )
    print(f"Đã lưu toàn bộ kết quả tại: {run_dir}")


if __name__ == "__main__":
    main()
