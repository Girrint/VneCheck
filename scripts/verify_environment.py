from __future__ import annotations

import json
import platform
import sys


def main() -> None:
    try:
        import cv2
        import torch
        import torchvision
        import ultralytics
    except ImportError as exc:
        raise SystemExit(f"Missing dependency: {exc}") from exc

    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "ultralytics": ultralytics.__version__,
        "opencv": cv2.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        report.update(
            {
                "gpu": properties.name,
                "gpu_memory_gb": round(properties.total_memory / 1024**3, 2),
                "compute_capability": f"{properties.major}.{properties.minor}",
            }
        )
        tensor = torch.rand((64, 64), device="cuda")
        report["cuda_smoke_sum"] = float(tensor.sum().item())
    print(json.dumps(report, indent=2))
    if not report["cuda_available"]:
        raise SystemExit("CUDA verification failed")


if __name__ == "__main__":
    main()
