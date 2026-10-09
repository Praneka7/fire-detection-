"""Web Command Center Dashboard backend for Fire Camera Detector.

Provides real-time MJPEG camera streaming, live flame telemetry, dynamic sensitivity
tuning, incident snapshot gallery, and REST endpoints.
"""

from __future__ import annotations

import argparse
import os
import platform
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_from_directory

from fire_detector import flame_mask, parse_source


class FireDetectorStreamer:
    def __init__(self, source: int | str = 0, min_area: int = 1200, confirm_frames: int = 8, cooldown: float = 10.0):
        self.source = source
        self.min_area = min_area
        self.confirm_frames = confirm_frames
        self.cooldown = cooldown

        self.output_dir = Path(__file__).parent / "detections"
        self.output_dir.mkdir(exist_ok=True)

        self.lock = threading.Lock()
        self.running = False
        self.thread: threading.Thread | None = None

        # Telemetry state
        self.is_alert = False
        self.candidate_frames = 0
        self.regions_count = 0
        self.total_flame_area = 0
        self.fps = 0.0
        self.total_alerts = len(list(self.output_dir.glob("*.jpg")))
        self.last_saved_time = 0.0
        self.last_alert_timestamp: str | None = None
        self.status_message = "Initializing..."

        # Frame buffers (JPEG bytes)
        self.current_frame_jpeg: bytes | None = None
        self.current_clean_jpeg: bytes | None = None
        self.current_mask_jpeg: bytes | None = None
        self.raw_frame: np.ndarray | None = None

    def clear_snapshots(self) -> int:
        count = 0
        for f in self.output_dir.glob("*.jpg"):
            try:
                f.unlink()
                count += 1
            except Exception:
                pass
        with self.lock:
            self.total_alerts = 0
            self.last_alert_timestamp = None
        return count

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def _play_alert_sound(self) -> None:
        if platform.system() == "Windows":
            try:
                import winsound
                sound_path = Path(__file__).parent / "alerm_audio.wav"
                if sound_path.is_file() and sound_path.stat().st_size > 500:
                    winsound.PlaySound(str(sound_path), winsound.SND_FILENAME | winsound.SND_ASYNC)
                else:
                    winsound.Beep(1100, 250)
            except Exception:
                pass

    def _capture_loop(self) -> None:
        cap = cv2.VideoCapture(parse_source(str(self.source)))
        # Request higher resolution if supported by the camera hardware
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            with self.lock:
                self.status_message = f"Failed to open source {self.source}"
            return

        previous_gray: np.ndarray | None = None
        frame_counter = 0
        fps_timer = time.monotonic()
        last_saved = 0.0

        with self.lock:
            self.status_message = "Active Monitoring"

        while self.running:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            with self.lock:
                current_min_area = self.min_area
                current_confirm = self.confirm_frames
                current_cooldown = self.cooldown

            # Compute flame mask and contours
            mask, previous_gray = flame_mask(frame, previous_gray)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            valid_regions = []
            total_area = 0
            for c in contours:
                area = cv2.contourArea(c)
                if area >= current_min_area:
                    valid_regions.append((cv2.boundingRect(c), int(area)))
                    total_area += int(area)

            # Update alert state
            if valid_regions:
                candidate_frames = self.candidate_frames + 1
            else:
                candidate_frames = 0
            alerted = candidate_frames >= current_confirm

            now = time.monotonic()
            saved_new_alert = False
            if alerted and (now - last_saved >= current_cooldown):
                filename = self.output_dir / f"fire_alert_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                cv2.imwrite(str(filename), frame)
                last_saved = now
                saved_new_alert = True
                self._play_alert_sound()

            # FPS calculation
            frame_counter += 1
            elapsed = now - fps_timer
            current_fps = self.fps
            if elapsed >= 1.0:
                current_fps = round(frame_counter / elapsed, 1)
                frame_counter = 0
                fps_timer = now

            # Clean raw frame (unobstructed feed)
            clean_frame = frame.copy()

            # Draw HUD overlay
            hud_frame = frame.copy()
            for (x, y, w, h), area in valid_regions:
                cv2.rectangle(hud_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                tag = f"FLAME {area}px"
                cv2.putText(hud_frame, tag, (x, max(y - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

            # Mask representation in 3 channels (colored amber for display)
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_colored = np.zeros_like(mask_bgr)
            mask_colored[:, :, 2] = mask  # Red channel
            mask_colored[:, :, 1] = (mask * 0.5).astype(np.uint8)  # Green channel (yields Orange)

            # Compress frames to crisp, high-fidelity JPEG (quality=95 for clean images)
            _, jpeg_frame = cv2.imencode(".jpg", hud_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            _, jpeg_clean = cv2.imencode(".jpg", clean_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            _, jpeg_mask = cv2.imencode(".jpg", mask_colored, [cv2.IMWRITE_JPEG_QUALITY, 85])

            with self.lock:
                self.candidate_frames = candidate_frames
                self.is_alert = alerted
                self.regions_count = len(valid_regions)
                self.total_flame_area = total_area
                self.fps = current_fps
                self.current_frame_jpeg = jpeg_frame.tobytes()
                self.current_clean_jpeg = jpeg_clean.tobytes()
                self.current_mask_jpeg = jpeg_mask.tobytes()
                self.raw_frame = frame
                if saved_new_alert:
                    self.total_alerts += 1
                    self.last_saved_time = now
                    self.last_alert_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            time.sleep(0.01)

        cap.release()

    def get_frame_stream(self, mode: str = "processed"):
        while self.running:
            with self.lock:
                if mode == "mask":
                    frame_bytes = self.current_mask_jpeg
                elif mode == "clean":
                    frame_bytes = self.current_clean_jpeg
                else:
                    frame_bytes = self.current_frame_jpeg

            if frame_bytes is None:
                time.sleep(0.04)
                continue

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            time.sleep(0.03)

    def manual_snapshot(self) -> str | None:
        with self.lock:
            frame = self.raw_frame
        if frame is None:
            return None
        filename = self.output_dir / f"manual_snap_{datetime.now():%Y%m%d_%H%M%S}.jpg"
        cv2.imwrite(str(filename), frame)
        with self.lock:
            self.total_alerts += 1
        return filename.name

    def get_status(self) -> dict:
        with self.lock:
            confirm_percent = min(100, int((self.candidate_frames / max(1, self.confirm_frames)) * 100))
            return {
                "is_alert": self.is_alert,
                "candidate_frames": self.candidate_frames,
                "confirm_frames": self.confirm_frames,
                "confirm_percent": confirm_percent,
                "regions_count": self.regions_count,
                "total_flame_area": self.total_flame_area,
                "min_area": self.min_area,
                "cooldown": self.cooldown,
                "fps": self.fps,
                "total_alerts": self.total_alerts,
                "last_alert_timestamp": self.last_alert_timestamp,
                "status_message": self.status_message,
                "server_time": datetime.now().strftime("%H:%M:%S"),
            }

    def update_config(self, min_area: int | None = None, confirm_frames: int | None = None, cooldown: float | None = None) -> None:
        with self.lock:
            if min_area is not None:
                self.min_area = max(100, int(min_area))
            if confirm_frames is not None:
                self.confirm_frames = max(1, int(confirm_frames))
            if cooldown is not None:
                self.cooldown = max(1.0, float(cooldown))

    def get_snapshots(self) -> list[dict]:
        files = sorted(self.output_dir.glob("*.jpg"), key=os.path.getmtime, reverse=True)
        results = []
        for file in files[:40]:  # return the 40 most recent snapshots
            stats = file.stat()
            results.append({
                "filename": file.name,
                "url": f"/api/snapshot/{file.name}",
                "size_kb": round(stats.st_size / 1024, 1),
                "timestamp": datetime.fromtimestamp(stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "is_manual": file.name.startswith("manual_snap_"),
            })
        return results


def create_app(streamer: FireDetectorStreamer) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/video_feed")
    def video_feed():
        return Response(streamer.get_frame_stream("processed"), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/api/clean_feed")
    def clean_feed():
        return Response(streamer.get_frame_stream("clean"), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/api/mask_feed")
    def mask_feed():
        return Response(streamer.get_frame_stream("mask"), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/api/status")
    def status():
        return jsonify(streamer.get_status())

    @app.route("/api/config", methods=["POST"])
    def update_config():
        data = request.get_json(force=True, silent=True) or {}
        streamer.update_config(
            min_area=data.get("min_area"),
            confirm_frames=data.get("confirm_frames"),
            cooldown=data.get("cooldown"),
        )
        return jsonify({"success": True, "config": streamer.get_status()})

    @app.route("/api/snapshots")
    def list_snapshots():
        return jsonify({"snapshots": streamer.get_snapshots()})

    @app.route("/api/clear_snapshots", methods=["POST"])
    def clear_snapshots():
        count = streamer.clear_snapshots()
        return jsonify({"success": True, "cleared_count": count})

    @app.route("/api/snapshot/<filename>", methods=["GET", "DELETE"])
    def get_snapshot(filename: str):
        if request.method == "DELETE":
            target = streamer.output_dir / filename
            if target.exists() and target.suffix.lower() == ".jpg":
                target.unlink()
                with streamer.lock:
                    streamer.total_alerts = max(0, streamer.total_alerts - 1)
                return jsonify({"success": True})
            return jsonify({"error": "File not found"}), 404
        return send_from_directory(streamer.output_dir, filename)

    @app.route("/api/manual_snapshot", methods=["POST"])
    def manual_snapshot():
        name = streamer.manual_snapshot()
        if name:
            return jsonify({"success": True, "filename": name})
        return jsonify({"success": False, "error": "No frame available"}), 400

    return app


def main():
    parser = argparse.ArgumentParser(description="Fire Camera Detector Web Command Center")
    parser.add_argument("--source", default="0", help="Camera index or video file")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Port (default: 5000)")
    parser.add_argument("--min-area", type=int, default=1200, help="Minimum flame area (px)")
    parser.add_argument("--confirm-frames", type=int, default=8, help="Confirmation frame count")
    parser.add_argument("--cooldown", type=float, default=10.0, help="Snapshot cooldown (seconds)")
    args = parser.parse_args()

    streamer = FireDetectorStreamer(
        source=args.source,
        min_area=args.min_area,
        confirm_frames=args.confirm_frames,
        cooldown=args.cooldown,
    )
    streamer.start()

    app = create_app(streamer)
    print(f"\n=======================================================")
    print(f"🔥 Fire Camera Detector Web Dashboard running at:")
    print(f"   👉 http://{args.host}:{args.port}")
    print(f"=======================================================\n")

    try:
        app.run(host=args.host, port=args.port, threaded=True, debug=False)
    finally:
        streamer.stop()


if __name__ == "__main__":
    main()
