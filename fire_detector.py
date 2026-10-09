"""Live camera/video fire-warning demo using colour and motion cues.

This detector is intentionally conservative: it searches for bright red/orange/yellow
regions that are also changing over time. It is a demonstration only, not a safety-
critical fire detection system.
"""

from __future__ import annotations

import argparse
import platform
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def parse_source(value: str) -> int | str:
    """Treat numeric sources as webcam indexes; preserve file/stream sources."""
    return int(value) if value.isdigit() else value

def play_alert() -> None:
    """Make a short local alert sound without requiring an extra package."""
    if platform.system() == "Windows":
        try:
            import winsound

            winsound.Beep(1100, 300)
        except RuntimeError:
            pass
    else:
        print("\a", end="", flush=True)


def flame_mask(frame: np.ndarray, previous_gray: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Return likely flame pixels combining HSV colour, brightness, and motion."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red wraps around hue=0 in HSV; orange/yellow extend up to hue 40.
    lower_red = cv2.inRange(hsv, (0, 110, 150), (15, 255, 255))
    orange_yellow = cv2.inRange(hsv, (16, 100, 170), (40, 255, 255))
    colour = cv2.bitwise_or(lower_red, orange_yellow)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if previous_gray is None:
        motion = np.zeros_like(gray)
    else:
        difference = cv2.absdiff(gray, previous_gray)
        _, motion = cv2.threshold(difference, 20, 255, cv2.THRESH_BINARY)
        motion = cv2.dilate(motion, np.ones((5, 5), np.uint8), iterations=2)

    # A flame must look like fire and be dynamic. Include some colour pixels beside
    # motion to keep the flame shape coherent.
    moving_colour = cv2.bitwise_and(colour, motion)
    mask = cv2.dilate(moving_colour, np.ones((7, 7), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return mask, gray


def find_fire_regions(mask: np.ndarray, min_area: int) -> list[tuple[int, int, int, int]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area]


def draw_overlay(frame: np.ndarray, regions: list[tuple[int, int, int, int]], alerted: bool) -> np.ndarray:
    output = frame.copy()
    for x, y, width, height in regions:
        cv2.rectangle(output, (x, y), (x + width, y + height), (0, 0, 255), 2)
        cv2.putText(output, "Possible fire", (x, max(y - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    status = "FIRE ALERT" if alerted else "Monitoring"
    colour = (0, 0, 255) if alerted else (0, 220, 0)
    cv2.rectangle(output, (10, 10), (310, 55), (0, 0, 0), -1)
    cv2.putText(output, status, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
    cv2.putText(output, "Press Q or Esc to exit", (10, output.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect probable fire in a webcam feed or video file.")
    parser.add_argument("--source", default="0", help="Camera index (default: 0) or path to a video file.")
    parser.add_argument("--min-area", type=int, default=1200, help="Minimum candidate area in pixels.")
    parser.add_argument("--confirm-frames", type=int, default=8, help="Consecutive candidate frames required for an alert.")
    parser.add_argument("--cooldown", type=float, default=10.0, help="Seconds between saved alert snapshots.")
    args = parser.parse_args()

    cap = cv2.VideoCapture(parse_source(args.source))
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source!r}. Check the camera index or video path.")

    output_dir = Path(__file__).parent / "detections"
    output_dir.mkdir(exist_ok=True)
    previous_gray: np.ndarray | None = None
    candidate_frames = 0
    last_saved = 0.0

    print("Monitoring started. Press Q or Esc in the video window to stop.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Video ended or camera frame could not be read.")
                break

            mask, previous_gray = flame_mask(frame, previous_gray)
            regions = find_fire_regions(mask, args.min_area)
            candidate_frames = candidate_frames + 1 if regions else 0
            alerted = candidate_frames >= args.confirm_frames

            now = time.monotonic()
            if alerted and now - last_saved >= args.cooldown:
                filename = output_dir / f"fire_alert_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                cv2.imwrite(str(filename), frame)
                print(f"FIRE ALERT: possible fire detected. Snapshot saved to {filename}")
                play_alert()
                last_saved = now

            cv2.imshow("Fire Camera Detector", draw_overlay(frame, regions, alerted))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
