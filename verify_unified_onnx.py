import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image

from food_ai.config import UNIFIED_MODEL, UNIFIED_ONNX_MODEL
from food_ai.data import build_transforms
from food_ai.unified_model import load_unified_checkpoint


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="So sánh unified PyTorch/ONNX")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=UNIFIED_MODEL)
    parser.add_argument("--onnx", type=Path, default=UNIFIED_ONNX_MODEL)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    onnx.checker.check_model(onnx.load(str(args.onnx)))
    _, transform = build_transforms()
    with Image.open(args.image) as image:
        tensor = transform(image).unsqueeze(0)

    model, _ = load_unified_checkpoint(args.checkpoint, "cpu")
    with torch.inference_mode():
        torch_outputs = [value.numpy() for value in model(tensor)]

    session = ort.InferenceSession(
        str(args.onnx), providers=["CPUExecutionProvider"]
    )
    onnx_outputs = session.run(
        None, {session.get_inputs()[0].name: tensor.numpy()}
    )
    names = ("food", "category", "freshness")
    passed = True
    for name, torch_value, onnx_value in zip(
        names, torch_outputs, onnx_outputs
    ):
        difference = float(np.max(np.abs(torch_value - onnx_value)))
        head_passed = difference < args.tolerance
        passed &= head_passed
        print(f"{name:<10} max_diff={difference:.10f} {'PASS' if head_passed else 'FAIL'}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
