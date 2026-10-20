import argparse
import json
import os
from pathlib import Path

from predict import predict as predict_legacy
from predict_unified import predict_unified


def main():
    parser = argparse.ArgumentParser(description="Chọn pipeline legacy/unified")
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--pipeline",
        choices=("legacy", "unified", "both"),
        default=os.getenv("AI_PIPELINE", "legacy"),
    )
    args = parser.parse_args()

    if args.pipeline == "legacy":
        result = {"pipeline": "legacy", "result": predict_legacy(args.image)}
    elif args.pipeline == "unified":
        result = {"pipeline": "unified", "result": predict_unified(args.image)}
    else:
        result = {
            "pipeline": "both",
            "legacy": predict_legacy(args.image),
            "unified": predict_unified(args.image),
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
