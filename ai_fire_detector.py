"""Monitor a webcam/video using a trained fire classifier and confirmed sound alerts."""

from __future__ import annotations

import argparse
import csv
import platform
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import torch
from PIL import Image
from torchvision import models, transforms

NORMALISE = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


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


def localise_fire(frame, previous_gray):
    """Find flame-coloured moving regions for display only (not AI training labels)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0, 105, 145), (15, 255, 255))
    orange_yellow = cv2.inRange(hsv, (16, 95, 165), (42, 255, 255))
    colour_mask = cv2.bitwise_or(red, orange_yellow)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if previous_gray is not None:
        difference = cv2.absdiff(gray, previous_gray)
        _, movement = cv2.threshold(difference, 18, 255, cv2.THRESH_BINARY)
        movement = cv2.dilate(movement, None, iterations=2)
        colour_mask = cv2.bitwise_and(colour_mask, movement)
    mask = cv2.morphologyEx(colour_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(contour) for contour in contours if cv2.contourArea(contour) >= 350]
    return boxes, gray


def load_model(model_path: Path, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(checkpoint["model_state"])
    model.eval().to(device)
    fire_index = checkpoint["class_to_idx"].get("fire_images")
    if fire_index is None:
        raise SystemExit("Model has no 'fire_images' class.")
    return model, fire_index


def predict_fire(model, frame, device, fire_index: int) -> float:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = NORMALISE(Image.fromarray(rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        probability = torch.softmax(model(image), dim=1)[0, fire_index].item()
    return probability


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")
    parser.add_argument("--model", type=Path, default=Path(__file__).parent / "models" / "fire_classifier.pt")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--confirm-frames", type=int, default=6)
    parser.add_argument("--cooldown", type=float, default=15)
    parser.add_argument("--alarm", type=Path, default=Path(__file__).parent / "alerm_audio.wav")
    parser.add_argument("--test-alarm", action="store_true", help="Play the alarm once, then exit.")
    args = parser.parse_args()
    if args.test_alarm:
        alert_sound(args.alarm)
        print(f"Alarm test requested: {args.alarm}")
        return
    if not args.model.exists():
        raise SystemExit("Trained model not found. Run train_model.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, fire_index = load_model(args.model, device)
    capture = cv2.VideoCapture(parse_source(args.source))
    if not capture.isOpened():
        raise SystemExit(f"Cannot open source: {args.source}")
    out_dir = Path(__file__).parent / "detections"
    out_dir.mkdir(exist_ok=True)
    log_path = out_dir / "alerts.csv"
    is_new_log = not log_path.exists()
    recent = deque(maxlen=args.confirm_frames)
    last_alert = 0.0
    previous_gray = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            confidence = predict_fire(model, frame, device, fire_index)
            boxes, previous_gray = localise_fire(frame, previous_gray)
            recent.append(confidence >= args.threshold)
            confirmed = len(recent) == args.confirm_frames and all(recent)
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
            cv2.rectangle(frame, (10, 10), (430, 68), (0, 0, 0), -1)
            cv2.putText(frame, f"{label} | fire: {confidence:.1%}", (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
            if confidence >= args.threshold:
                for x, y, width, height in boxes:
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 2)
                    cv2.putText(frame, "Possible fire", (x, max(y - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
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
