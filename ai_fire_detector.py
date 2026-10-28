"""Monitor a webcam/video using a trained fire classifier and confirmed sound alerts."""

from __future__ import annotations

import argparse
import csv
import platform
import time
from datetime import datetime
from pathlib import Path

import cv2

from fire_detector import ConsecutiveFireConfirmation, detect_verified_fire, draw_overlay
from fire_model import FireModel


def parse_source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def alert_sound(sound_path: Path) -> None:
    if platform.system() == "Windows":
        import winsound
        try:
            if sound_path.is_file():
                winsound.PlaySound(str(sound_path), winsound.SND_FILENAME | winsound.SND_ASYNC)
            else:
                winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS | winsound.SND_ASYNC)
        except RuntimeError:
            winsound.Beep(1150, 500)
    else:
        print("\a", end="", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")
    parser.add_argument("--model", type=Path, default=Path(__file__).parent / "models" / "fire_classifier.pt")
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--min-area", type=int, default=350)
    parser.add_argument("--confirm-frames", type=int, default=8)
    parser.add_argument("--cooldown", type=float, default=15)
    parser.add_argument("--alarm", type=Path, default=Path(__file__).parent / "alerm_audio.wav")
    parser.add_argument("--test-alarm", action="store_true", help="Play the alarm once, then exit.")
    args = parser.parse_args()
    if args.test_alarm:
        alert_sound(args.alarm)
        print(f"Alarm test requested: {args.alarm}")
        return
    classifier = FireModel(args.model, args.threshold)
    if not classifier.available:
        raise SystemExit(f"AI fire verification unavailable; alarms cannot run. {classifier.error}")
    capture = cv2.VideoCapture(parse_source(args.source))
    if not capture.isOpened():
        raise SystemExit(f"Cannot open source: {args.source}")
    out_dir = Path(__file__).parent / "detections"
    out_dir.mkdir(exist_ok=True)
    log_path = out_dir / "alerts.csv"
    is_new_log = not log_path.exists()
    confirmation = ConsecutiveFireConfirmation(args.confirm_frames)
    last_alert = 0.0
    previous_gray = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            _, previous_gray, regions, _ = detect_verified_fire(
                frame, previous_gray, classifier, args.min_area, args.threshold
            )
            confidence = max((score for _, _, score in regions), default=0.0)
            candidate_frames, confirmed = confirmation.update(bool(regions))
            now = time.monotonic()
            if confirmed and now - last_alert >= args.cooldown:
                timestamp = datetime.now()
                image_path = out_dir / f"fire_{timestamp:%Y%m%d_%H%M%S}.jpg"
                cv2.imwrite(str(image_path), frame)
                with log_path.open("a", newline="", encoding="utf-8") as file:
                    writer = csv.writer(file)
                    if is_new_log:
                        writer.writerow(["timestamp", "confidence", "snapshot"])
                        is_new_log = False
                    writer.writerow([timestamp.isoformat(timespec="seconds"), f"{confidence:.4f}", image_path.name])
                alert_sound(args.alarm)
                last_alert = now
            label = "FIRE ALERT" if confirmed else "Monitoring"
            colour = (0, 0, 255) if confirmed else (0, 220, 0)
            frame = draw_overlay(frame, regions, confirmed, classifier.available)
            cv2.putText(frame, f"Max verified confidence: {confidence:.1%}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
            cv2.putText(frame, "A: test alarm | Q / Esc: exit", (10, frame.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.imshow("AI Fire Detector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("a"):
                alert_sound(args.alarm)
            if key in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
