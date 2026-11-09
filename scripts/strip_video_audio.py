from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-encode a video without an audio stream")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open input video: {args.input}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot initialize output video: {args.output}")

    written = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        writer.write(frame)
        written += 1

    capture.release()
    writer.release()
    if written == 0:
        raise RuntimeError("The input video contained no readable frames")

    print(json.dumps({
        "input": str(args.input),
        "output": str(args.output),
        "width": width,
        "height": height,
        "fps": fps,
        "expected_frames": expected_frames,
        "written_frames": written,
        "duration_seconds": written / fps,
        "audio": False,
    }, indent=2))


if __name__ == "__main__":
    main()
