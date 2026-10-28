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
from typing import Protocol

import cv2
import numpy as np

from fire_model import FireModel


class RegionScorer(Protocol):
    available: bool
    threshold: float

    def predict_region(self, frame: np.ndarray, box: tuple[int, int, int, int]) -> float | None: ...


class ConsecutiveFireConfirmation:
    """Require verified fire in consecutive frames; any negative resets the streak."""

    def __init__(self, required_frames: int) -> None:
        self.required_frames = max(1, int(required_frames))
        self.count = 0

    def update(self, verified_fire: bool, required_frames: int | None = None) -> tuple[int, bool]:
        if required_frames is not None:
            required_frames = max(1, int(required_frames))
            if required_frames != self.required_frames:
                self.required_frames = required_frames
                self.count = 0
        self.count = self.count + 1 if verified_fire else 0
        return self.count, self.count >= self.required_frames


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


FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def flame_mask(frame: np.ndarray, previous_gray: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Return likely flame pixels combining color, incandescence, motion, and skin/face rejection."""
    b = frame[:, :, 0].astype(int)
    g = frame[:, :, 1].astype(int)
    r = frame[:, :, 2].astype(int)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Strict flame color spectrum:
    # 1. Red / Deep Orange combustion: Hue in [0, 16] or [165, 180], S >= 135, V >= 165, R >= 190, R - B >= 80
    fire_red = ((h <= 16) | (h >= 165)) & (s >= 135) & (v >= 165) & (r >= 190) & ((r - b) >= 80)
    # 2. Bright orange-yellow flame core: H in [17, 21], S >= 180, V >= 220, R >= 220, R - G >= 40, B <= 60
    fire_orange = (h >= 17) & (h <= 21) & (s >= 180) & (v >= 220) & (r >= 220) & ((r - g) >= 40) & (b <= 60)
    colour = ((fire_red | fire_orange).astype(np.uint8)) * 255

    # Reject human skin tones (Kovac / Peer model in RGB + YCrCb)
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    skin_rgb = (r > 95) & (g > 40) & (b > 20) & ((r - g) > 15) & (r > b)
    skin_ycrcb = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
    colour[skin_rgb & skin_ycrcb] = 0

    # Human face & head suppression: zero out detected face regions
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if not FACE_CASCADE.empty():
        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(50, 50))
        for (fx, fy, fw, fh) in faces:
            pad_w = int(fw * 0.30)
            pad_h = int(fh * 0.40)
            x1 = max(0, fx - pad_w)
            y1 = max(0, fy - pad_h)
            x2 = min(frame.shape[1], fx + fw + pad_w)
            y2 = min(frame.shape[0], fy + fh + int(pad_h * 1.5))
            colour[y1:y2, x1:x2] = 0

    # Dynamic motion differencing: flames are constantly moving and turbulent
    if previous_gray is None:
        motion = np.zeros_like(gray)
    else:
        difference = cv2.absdiff(gray, previous_gray)
        # Suppress camera pan or global exposure changes if > 35% of frame moved
        if np.mean(difference > 15) > 0.35:
            motion = np.zeros_like(gray)
        else:
            _, motion = cv2.threshold(difference, 15, 255, cv2.THRESH_BINARY)
            motion = cv2.dilate(motion, np.ones((5, 5), np.uint8), iterations=2)

    moving_colour = cv2.bitwise_and(colour, motion)
    mask = cv2.dilate(moving_colour, np.ones((7, 7), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return mask, gray


def find_fire_regions(mask: np.ndarray, min_area: int) -> list[tuple[int, int, int, int]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area]


def detect_verified_fire(
    frame: np.ndarray,
    previous_gray: np.ndarray | None,
    classifier: RegionScorer | None,
    min_area: int,
    confidence_threshold: float,
    allow_heuristic_fallback: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[tuple[tuple[int, int, int, int], int, float]], int]:
    """Generate flame-like proposals, then keep only regions verified by AI.

    If the AI model is unavailable and allow_heuristic_fallback is True, dynamic
    flame mask proposals (HSV color + motion) are accepted so surveillance continues.
    """
    mask, gray = flame_mask(frame, previous_gray)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    verified: list[tuple[tuple[int, int, int, int], int, float]] = []
    candidate_count = 0
    if classifier is None or not classifier.available:
        if allow_heuristic_fallback:
            for contour in contours:
                area = int(cv2.contourArea(contour))
                if area < min_area:
                    continue
                candidate_count += 1
                box = tuple(map(int, cv2.boundingRect(contour)))
                verified.append((box, area, 1.0))
            return mask, gray, verified, candidate_count
        return mask, gray, verified, 0

    threshold = float(np.clip(confidence_threshold, 0.50, 0.999))
    for contour in contours:
        area = int(cv2.contourArea(contour))
        if area < min_area:
            continue
        candidate_count += 1
        box = tuple(map(int, cv2.boundingRect(contour)))
        confidence = classifier.predict_region(frame, box)
        if confidence is not None and confidence >= threshold:
            verified.append((box, area, confidence))
    return mask, gray, verified, candidate_count


def draw_overlay(
    frame: np.ndarray,
    regions: list[tuple[tuple[int, int, int, int], int, float]],
    alerted: bool,
    model_ready: bool = True,
) -> np.ndarray:
    output = frame.copy()
    for (x, y, width, height), area, confidence in regions:
        cv2.rectangle(output, (x, y), (x + width, y + height), (0, 0, 255), 2)
        tag = f"VERIFIED FIRE {confidence:.0%} | {area}px" if model_ready else f"FLAME {area}px (OPTICAL)"
        cv2.putText(output, tag, (x, max(y - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    status = "FIRE ALERT" if alerted else "Monitoring"
    colour = (0, 0, 255) if alerted else (0, 220, 0)
    cv2.rectangle(output, (10, 10), (340, 55), (0, 0, 0), -1)
    cv2.putText(output, status, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
    if not model_ready:
        cv2.putText(output, "OPTICAL FALLBACK ACTIVE (COLOR & MOTION)", (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 215, 255), 2)
    cv2.putText(output, "Press Q or Esc to exit", (10, output.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect probable fire in a webcam feed or video file.")
    parser.add_argument("--source", default="0", help="Camera index (default: 0) or path to a video file.")
    parser.add_argument("--min-area", type=int, default=1200, help="Minimum candidate area in pixels.")
    parser.add_argument("--confirm-frames", type=int, default=15, help="Consecutive candidate frames required for an alert.")
    parser.add_argument("--cooldown", type=float, default=10.0, help="Seconds between saved alert snapshots.")
    parser.add_argument("--confidence", type=float, default=0.90, help="Minimum AI fire confidence (0.50-0.999).")
    parser.add_argument("--model", type=Path, default=Path(__file__).parent / "models" / "fire_classifier.pt")
    parser.add_argument("--strict-ai", action="store_true", help="Strictly mute alerts if AI model weights are missing.")
    args = parser.parse_args()

    cap = cv2.VideoCapture(parse_source(args.source))
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source!r}. Check the camera index or video path.")

    output_dir = Path(__file__).parent / "detections"
    output_dir.mkdir(exist_ok=True)
    classifier = FireModel(args.model, args.confidence)
    allow_fallback = not args.strict_ai
    if not classifier.available:
        if allow_fallback:
            print(f"Notice: AI weights unavailable ({classifier.error}). Operating in Optical Fallback mode (HSV Color + Motion Dynamics).")
        else:
            print(f"AI fire verification unavailable; automatic alarms are muted. {classifier.error}")
    previous_gray: np.ndarray | None = None
    confirmation = ConsecutiveFireConfirmation(args.confirm_frames)
    last_saved = 0.0

    print("Monitoring started. Press Q or Esc in the video window to stop.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Video ended or camera frame could not be read.")
                break

            mask, previous_gray, regions, candidate_count = detect_verified_fire(
                frame, previous_gray, classifier, args.min_area, args.confidence,
                allow_heuristic_fallback=allow_fallback,
            )
            candidate_frames, alerted = confirmation.update(bool(regions))

            now = time.monotonic()
            if alerted and now - last_saved >= args.cooldown:
                filename = output_dir / f"fire_alert_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                cv2.imwrite(str(filename), frame)
                print(f"FIRE ALERT: possible fire detected. Snapshot saved to {filename}")
                play_alert()
                last_saved = now

            cv2.imshow("Fire Camera Detector", draw_overlay(frame, regions, alerted, classifier.available))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
