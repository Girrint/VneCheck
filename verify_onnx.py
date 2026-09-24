from pathlib import Path
import argparse
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from food_ai.config import (
    CATEGORIES,
    MODEL_DIR,
    ROUTER_MODEL,
    freshness_model,
)
from food_ai.data import build_transforms
from food_ai.model import load_checkpoint


DEVICE = "cpu"


def preprocess(image_path: Path) -> torch.Tensor:
    _, transform = build_transforms()

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    return tensor


def verify_graph(model_path: Path):
    model = onnx.load(str(model_path))
    onnx.checker.check_model(model)


def run_pytorch(checkpoint: Path, tensor: torch.Tensor):
    model, classes = load_checkpoint(checkpoint, DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)

    return classes, probs.squeeze().cpu().numpy()


def run_onnx(model_path: Path, tensor: torch.Tensor):
    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    logits = session.run(
        None,
        {
            input_name: tensor.numpy(),
        },
    )[0]

    probs = torch.softmax(torch.from_numpy(logits), dim=1)

    return probs.squeeze().numpy()


def compare(name, classes, pt_probs, onnx_probs):

    print()
    print("=" * 80)
    print(name)
    print("=" * 80)

    print(
        f"{'Class':20}"
        f"{'PyTorch':>12}"
        f"{'ONNX':>12}"
        f"{'Diff':>12}"
    )

    print("-" * 80)

    for c, p1, p2 in zip(classes, pt_probs, onnx_probs):
        print(
            f"{c:20}"
            f"{p1:12.6f}"
            f"{p2:12.6f}"
            f"{abs(p1-p2):12.8f}"
        )

    pt_idx = np.argmax(pt_probs)
    onnx_idx = np.argmax(onnx_probs)

    max_diff = np.max(np.abs(pt_probs - onnx_probs))
    mean_diff = np.mean(np.abs(pt_probs - onnx_probs))

    print()
    print(f"PyTorch Top1 : {classes[pt_idx]}")
    print(f"ONNX Top1    : {classes[onnx_idx]}")
    print(f"Max diff     : {max_diff:.10f}")
    print(f"Mean diff    : {mean_diff:.10f}")

    ok = (
        pt_idx == onnx_idx
        and max_diff < 1e-5
    )

    print()

    if ok:
        print("PASS")
    else:
        print("FAIL")

    return ok


def verify_model(title, checkpoint, onnx_model, tensor):

    if not checkpoint.exists():
        print(f"Missing checkpoint: {checkpoint}")
        return False

    if not onnx_model.exists():
        print(f"Missing ONNX: {onnx_model}")
        return False

    verify_graph(onnx_model)

    classes, pt_probs = run_pytorch(
        checkpoint,
        tensor,
    )

    onnx_probs = run_onnx(
        onnx_model,
        tensor,
    )

    return compare(
        title,
        classes,
        pt_probs,
        onnx_probs,
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
        help="Ảnh dùng để verify",
    )

    args = parser.parse_args()

    image = Path(args.image)

    if not image.exists():
        raise FileNotFoundError(image)

    tensor = preprocess(image)

    results = []

    results.append(
        verify_model(
            "ROUTER",
            ROUTER_MODEL,
            MODEL_DIR / "router.onnx",
            tensor,
        )
    )

    for category in CATEGORIES:

        checkpoint = freshness_model(category)
        onnx_model = MODEL_DIR / "freshness" / f"{category}.onnx"

        results.append(
            verify_model(
                f"FRESHNESS - {category}",
                checkpoint,
                onnx_model,
                tensor,
            )
        )

    print()
    print("=" * 80)

    if all(results):
        print("ALL MODELS VERIFIED SUCCESSFULLY")
    else:
        print("VERIFY FAILED")

    print("=" * 80)


if __name__ == "__main__":
    main()