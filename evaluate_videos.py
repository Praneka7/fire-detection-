"""Evaluate fire alerts on labeled fire and hard-negative video clips.

Expected layout: fire_dir/clip.mp4 and non_fire_dir/category/clip.mp4. The script
reports clip-level confusion counts; it does not claim performance on scenes absent
from the supplied evaluation set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from fire_detector import ConsecutiveFireConfirmation, detect_verified_fire
from fire_model import FireModel

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def clips_in(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)


def evaluate_clip(path: Path, label: str, model: FireModel, min_area: int,
                  threshold: float, confirm_frames: int) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {"file": str(path), "label": label, "error": "Could not open video"}

    previous_gray = None
    confirmation = ConsecutiveFireConfirmation(confirm_frames)
    max_confidence = 0.0
    frames = 0
    alerted = False
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames += 1
            _, previous_gray, verified, _ = detect_verified_fire(
                frame, previous_gray, model, min_area, threshold
            )
            confidence = max((score for _, _, score in verified), default=0.0)
            max_confidence = max(max_confidence, confidence)
            _, confirmed = confirmation.update(bool(verified))
            if confirmed:
                alerted = True
    finally:
        capture.release()

    return {
        "file": str(path),
        "label": label,
        "frames": frames,
        "alerted": alerted,
        "max_verified_confidence": round(max_confidence, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fire-alert behavior on labeled clips.")
    parser.add_argument("--fire-dir", required=True, type=Path, help="Directory of fire-positive video clips.")
    parser.add_argument("--non-fire-dir", required=True, type=Path, help="Directory of hard-negative clips (e.g. hands, faces, orange objects, lamps).")
    parser.add_argument("--model", type=Path, default=Path(__file__).parent / "models" / "fire_classifier.pt")
    parser.add_argument("--min-area", type=int, default=350)
    parser.add_argument("--confidence", type=float, default=0.90)
    parser.add_argument("--confirm-frames", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("video_evaluation.json"))
    args = parser.parse_args()

    fire_clips = clips_in(args.fire_dir)
    non_fire_clips = clips_in(args.non_fire_dir)
    if not fire_clips or not non_fire_clips:
        raise SystemExit("Both folders must contain at least one supported video file.")

    model = FireModel(args.model, args.confidence)
    if not model.available:
        raise SystemExit(f"Cannot evaluate without a compatible trained model: {model.error}")

    results = [evaluate_clip(path, "fire", model, args.min_area, args.confidence, args.confirm_frames)
               for path in fire_clips]
    results.extend(evaluate_clip(path, "non_fire", model, args.min_area, args.confidence, args.confirm_frames)
                   for path in non_fire_clips)
    valid = [row for row in results if "error" not in row]
    tp = sum(row["label"] == "fire" and row["alerted"] for row in valid)
    fn = sum(row["label"] == "fire" and not row["alerted"] for row in valid)
    fp = sum(row["label"] == "non_fire" and row["alerted"] for row in valid)
    tn = sum(row["label"] == "non_fire" and not row["alerted"] for row in valid)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    summary = {
        "confidence_threshold": args.confidence,
        "confirm_frames": args.confirm_frames,
        "min_area": args.min_area,
        "clip_counts": {"true_positive": tp, "false_negative": fn, "false_positive": fp, "true_negative": tn},
        "precision": precision,
        "recall": recall,
        "specificity": tn / max(tn + fp, 1),
        "results": results,
        "note": "Clip-level evaluation on only the supplied videos; not a guarantee for unseen camera conditions.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
