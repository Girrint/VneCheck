from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def train(data: Path, config_path: Path, project: Path, name: str, smoke: bool = False) -> Path:
    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install CUDA PyTorch and project dependencies before training") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; YOLO11l training is not allowed to fall back to CPU")
    if not data.is_file():
        raise FileNotFoundError(data)

    config = load_config(config_path)
    if config.get("model") != "yolo11l.pt":
        raise ValueError("This project requires model=yolo11l.pt")
    if smoke:
        config.update({"epochs": 1, "patience": 1, "cache": False, "plots": False})

    project.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "data": str(data.resolve()),
        "config": dict(config),
        "smoke": smoke,
    }
    run_metadata = project / f"{name}_launch.json"
    run_metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    train_args = dict(config)
    model_path = train_args.pop("model")
    model = YOLO(model_path)
    model.train(data=str(data.resolve()), project=str(project.resolve()), name=name, **train_args)
    run_dir = Path(model.trainer.save_dir)
    (run_dir / "environment.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"Training completed without best checkpoint: {best}")
    export_dir = run_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    export_model = YOLO(str(best))
    exported = Path(
        export_model.export(
            format="onnx",
            imgsz=int(config["imgsz"]),
            batch=1,
            dynamic=False,
            opset=17,
            simplify=False,
        )
    )
    final_onnx = export_dir / "yolo11l_final.onnx"
    if exported.resolve() != final_onnx.resolve():
        shutil.copy2(exported, final_onnx)
    import onnx

    onnx.checker.check_model(onnx.load(str(final_onnx)))
    export_metadata = {
        "architecture": "YOLO11l",
        "initialization": "COCO pretrained",
        "source_checkpoint": str(best.resolve()),
        "onnx": str(final_onnx.resolve()),
        "input_shape": [1, 3, int(config["imgsz"]), int(config["imgsz"])],
        "opset": 17,
        "class_names": model.names,
    }
    (export_dir / "export_info.json").write_text(
        json.dumps(export_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Exported and verified detector ONNX: {final_onnx}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the mandatory YOLO11l detector")
    parser.add_argument("--data", type=Path, default=Path("datasets/detection/data.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/detector_yolo11l.json"))
    parser.add_argument("--project", type=Path, default=Path("runs/detection"))
    parser.add_argument("--name", default="yolo11l_640_e30_b4")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(train(args.data, args.config, args.project, args.name, args.smoke))


if __name__ == "__main__":
    main()
