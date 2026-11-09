from __future__ import annotations

import json
from pathlib import Path


def export_classifier(checkpoint_path: Path, output_path: Path | None = None) -> Path:
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch

    from .train import create_model

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = create_model(checkpoint["architecture"])
    model.load_state_dict(checkpoint["model"])
    model.eval()

    config = checkpoint["config"]
    input_size = int(config["input_size"])
    destination = output_path or checkpoint_path.with_suffix(".onnx")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sample = torch.zeros((1, 3, input_size, input_size), dtype=torch.float32)
    torch.onnx.export(
        model,
        sample,
        str(destination),
        input_names=["images"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
        dynamic_axes=None,
    )

    onnx_model = onnx.load(str(destination))
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
    output = session.run(None, {"images": np.zeros((1, 3, input_size, input_size), np.float32)})[0]
    if tuple(output.shape) != (1, 2):
        raise RuntimeError(f"Unexpected ONNX output shape: {output.shape}")

    metadata = {
        "architecture": checkpoint["architecture"],
        "produce": checkpoint["produce"],
        "classes": checkpoint["classes"],
        "input_name": "images",
        "output_name": "logits",
        "input_shape": [1, 3, input_size, input_size],
        "output_shape": [1, 2],
        "opset": 17,
        "source_checkpoint": str(checkpoint_path.resolve()),
        "onnx": str(destination.resolve()),
        "normalization_mean": config["normalization_mean"],
        "normalization_std": config["normalization_std"],
        "contrast_enhancement": config["contrast_enhancement"],
        "sharpen_factor": config["sharpen_factor"],
        "padding_rgb": [114, 114, 114],
        "validation_macro_f1": checkpoint["val_metrics"]["macro_f1"],
        "checkpoint_epoch": checkpoint["epoch"],
        "training_max_epochs": config["max_epochs"],
    }
    destination.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return destination
