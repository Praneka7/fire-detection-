<div align="center">

# 🔥 PyroGuard
### Real-Time Computer Vision & AI Fire Early Warning Command Center

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Headless%204.8+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Render](https://img.shields.io/badge/Render-Deployed-46E3B7?style=for-the-badge&logo=render&logoColor=black)](https://render.com/)

**A fire-only monitoring demo with a Flask dashboard, browser webcam streaming, AI-verified candidate regions, live telemetry, and a snapshot archive.**

[🌐 Live Cloud Deployment](https://fire-detection-y22g.onrender.com) • [💻 Local Setup](#-quick-start-local) • [🚀 Deploy to Render](#-cloud-deployment-render) • [🧠 AI Model Training](#-ai-deep-learning-workflow)

---

</div>

## 📌 Overview

**PyroGuard** uses the same verified-fire pipeline in the dashboard and both standalone camera programs:

1. **Candidate proposal:** HSV color plus frame differencing proposes moving, bright warm-color regions. This is not a fire decision; it can propose hands, clothing, lights, and other negatives.
2. **AI verification:** An EfficientNet-B0 fire/non-fire classifier scores each proposed region. Only regions above the configurable confidence threshold count toward an alert. If weights or AI dependencies are unavailable, detection fails closed and automatic alerts remain off.
3. **Temporal confirmation:** The verified candidate must persist for the configured number of consecutive frames. A non-fire/low-confidence frame resets the counter.
4. **Web Command Center Dashboard:**
   - Modern dark cyber-surveillance dashboard with glassmorphism UI.
   - **Cross-Platform Webcam Support**: Works locally with physical USB webcams AND in cloud deployments (Render, AWS, Heroku) via HTML5 browser webcam streaming (`navigator.mediaDevices.getUserMedia`).
   - Interactive telemetry (Flame Area in px², active regions, buffer progress bar, FPS).
   - Dynamic sensitivity sliders (Min Area, Confirmation Frames, Snapshot Cooldown).
   - Web Audio siren synthesizer + automated timestamped incident snapshot archive.

---

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| **🌐 Browser Webcam Capture** | Supports client-side browser cameras via HTML5 WebRTC/UserMedia, allowing real-time fire detection even when deployed on cloud servers without physical webcams. |
| **🎯 Multi-View Optical HUD** | Toggle between **Detection HUD** (bounding boxes & pixel area), **Clean HD Feed**, and **Amber Flame Mask** (for visual debugging). |
| **🧠 AI-Verified Fire Alerts** | A candidate requires AI fire confidence above threshold and multi-frame confirmation; heuristic color/motion alone never alarms. |
| **⏱️ Anti-Flicker Confirmation** | Requires configurable consecutive AI-verified frames (default: `8`) before triggering alarms. |
| **🔊 Multi-Channel Alarm** | Dual alert system: Web Audio API siren synthesis in the browser + Windows audio alarms via `winsound`. |
| **📸 Forensic Snapshot Gallery** | Automatic timestamped image saving on confirmed alerts + 1-click manual snapshot capture with full-screen preview and download. |
| **🎛️ Dynamic Tuning** | Adjust flame area thresholds and confirmation buffers on the fly without restarting the server. |
| **🚀 Cloud-Ready Architecture** | Pre-configured with `render.yaml`, multithreaded Gunicorn, and headless OpenCV for zero-friction cloud deployment. |

---

## 🏗️ Project Architecture

```
fire-camera-detector/
├── templates/
│   └── index.html               # Web Command Center UI (Semantic HTML5)
├── static/
│   ├── css/
│   │   └── style.css            # Cyber Obsidian UI & Glassmorphism Design System
│   └── js/
│       └── app.js               # Frontend Engine, Web Audio Siren & Webcam Streamer
├── dashboard.py                 # Flask Command Center, MJPEG Server & REST Endpoints
├── fire_detector.py             # Shared proposal/AI-verification pipeline and OpenCV runner
├── fire_model.py                 # Shared EfficientNet loader, validation and ROI inference
├── ai_fire_detector.py          # Standalone runner using the same verified pipeline
├── train_model.py               # PyTorch CNN Classifier Training Pipeline
├── evaluate_videos.py            # Clip-level positive and hard-negative evaluation
├── tests/                        # Unit tests for AI gating/fail-closed behavior
├── download_dataset.py          # Automated Kaggle Dataset Ingestion via KaggleHub
├── alerm_audio.wav              # Primary alert audio chime
├── requirements.txt             # Lightweight production dependencies (Flask, Gunicorn, OpenCV-headless)
├── requirements-ai.txt          # PyTorch/torchvision inference & training dependencies
├── render.yaml                  # Render Infrastructure as Code Blueprint
├── run_dashboard.bat            # Windows 1-Click Dashboard Launcher
└── README.md                    # Project Documentation
```

---

## 💻 Quick Start (Local)

### 1. Clone the Repository
```bash
git clone https://github.com/Praneka7/fire-detection-.git
cd fire-detection-
```

### 2. Setup Virtual Environment
```powershell
# Windows
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-ai.txt
```

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-ai.txt
```

### 3. Launch Web Command Center

#### Option A: One-Click (Windows)
Double-click `run_dashboard.bat` or run:
```powershell
.\run_dashboard.bat
```

#### Option B: Terminal
```bash
python dashboard.py
```
Open your browser and navigate to:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 🖥️ Running OpenCV Terminal Windows

You can also run the detector directly in OpenCV GUI windows:

```bash
# Default webcam (index 0)
python fire_detector.py

# External camera (index 1)
python fire_detector.py --source 1

# Saved video file
python fire_detector.py --source "path/to/video.mp4"

# Adjust sensitivity via CLI flags
python fire_detector.py --source 0 --min-area 500 --confidence 0.95 --confirm-frames 10
```
*Press `q` or `Esc` to exit.*

The workspace default `models/fire_classifier.pt` is a Git LFS pointer without pre-bundled weights. When AI weights or PyTorch dependencies are not loaded, the detector gracefully operates in **Optical Fallback Mode** (HSV Color + Dynamic Motion Analysis) so fire detection, bounding boxes, sirens, and snapshots function out-of-the-box. To strictly require AI verification and mute unverified alerts, use the `--strict-ai` flag. Supply compatible EfficientNet-B0 weights or train a custom model as described below to enable full neural classification.

---

## 🚀 Cloud Deployment (Render)

This project is optimized for deployment on **[Render](https://render.com/)**:

### Quick Deploy Steps:
1. Fork or push this repository to your GitHub account.
2. Sign in to **[dashboard.render.com](https://dashboard.render.com/)**.
3. Click **New +** > **Web Service**.
4. Connect your GitHub repository: `Praneka7/fire-detection-`.
5. Configure the deployment settings:

| Field | Configuration |
| :--- | :--- |
| **Runtime** | `Python 3` |
| **Build Command** | `pip install -r requirements.txt -r requirements-ai.txt` |
| **Start Command** | `gunicorn -w 1 --threads 8 --timeout 0 -b 0.0.0.0:$PORT dashboard:app` |
| **Instance Type** | `Free` |

6. Click **Deploy Web Service**.
7. Once live, open your site and click **"Activate My Webcam"** to allow browser camera access for optical detection!

> [!NOTE]
> **Why `--threads 8 --timeout 0`?**
> MJPEG video streaming holds long-lived HTTP connections. Using multithreaded Gunicorn with disabled timeouts ensures video streaming runs smoothly without blocking static assets or triggering 30-second worker kills.

---

## 🧠 AI Deep Learning Workflow

To train and run the neural network classifier:

### 1. Install the AI Runtime and Training Dependencies
```bash
pip install -r requirements-ai.txt
```

### 2. Download Kaggle Dataset
```bash
python download_dataset.py
```
*Downloads the Kaggle Fire/Non-Fire image classification dataset into local storage.*

### 3. Curate the Two-Class Dataset
The training root must contain exactly two `ImageFolder` classes:

```text
train-data/
├── fire_images/       # real flames, small/distant flames, indoor/outdoor fire
└── non_fire_images/   # hands, faces, skin, red/orange clothing, lamps, sunlight,
                  # reflections, orange objects, and ordinary scenes
```

Put the examples directly in their corresponding class folder (use descriptive filenames such as `hand_0001.jpg`). Do not add category subfolders beneath these two class folders; PyTorch `ImageFolder` would treat them as additional classes. Include real hard negatives recorded under similar lighting and camera conditions to deployment. The Kaggle set alone is not guaranteed to include enough hands/skin and other hard negatives.

For an unbiased evaluation, separately curate `test-data/fire_images` and `test-data/non_fire_images` from **different scenes/videos** than the training data. Do not split adjacent frames or near-duplicate images across train and test.

### 4. Train the Classifier
```bash
python train_model.py --data-dir "path/to/train-data" --test-dir "path/to/test-data" --epochs 12 --threshold 0.90
```
The model checkpoint is saved as `models/fire_classifier.pt`. `models/metrics.json` reports the configured threshold, precision, recall, specificity, F1, and confusion counts. Without `--test-dir`, metrics use a random holdout from the training directory and may be optimistic when related scenes appear in multiple splits.

The existing `models/metrics.json` is from the older training format and contains accuracy/loss only; it is not evidence that this alert pipeline meets the requested false-positive tests. Retrain and evaluate against a separate hard-negative test set to create current metrics. The current workspace checkpoint is only a 133-byte Git LFS pointer, not loadable weights; the AI gate intentionally stays disabled until actual compatible EfficientNet-B0 weights are supplied.

### 5. Evaluate Labeled Videos
Arrange actual clips in two directories; nested subfolders under `non-fire-clips` are allowed for categories such as hands, faces, skin, red clothing, orange objects, lamps, and ordinary scenes:

```text
evaluation-clips/
├── fire/                  # real fire videos, including small/distant flames
└── non-fire/
   ├── hands/
   ├── faces-skin/
   ├── orange-red-objects/
   └── ordinary-scenes/
```

Run the same detection pipeline used by the camera programs:

```bash
python evaluate_videos.py --fire-dir "evaluation-clips/fire" --non-fire-dir "evaluation-clips/non-fire" --confidence 0.90 --confirm-frames 8 --output "video_evaluation.json"
```

The report gives clip-level precision, recall, specificity, and confusion counts. A positive clip counts as detected if it triggers at least once; a negative clip counts as false positive if it triggers at least once. Run it on your own labeled fire and hard-negative clips; no real fire/hand/face video acceptance results are claimed by this repository until that evaluation is actually run.

Run the unit tests with `python -m unittest discover -s tests -v`. They use a mocked scorer to verify that low-confidence warm moving regions are rejected, unavailable models fail closed, and a negative frame resets confirmation. They do **not** validate actual model accuracy or replace the labeled-video evaluation above.

### 6. Run the AI Live Detector
```bash
python ai_fire_detector.py --source 0 --confidence 0.90 --min-area 350 --confirm-frames 8
```

The Flask dashboard has a matching AI-confidence slider. Settings can also be updated through `POST /api/config` using `confidence_threshold` (0.50–0.999), `min_area`, `confirm_frames`, and `cooldown`.

---

## 📡 REST API Reference

The dashboard exposes REST endpoints for integration:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Web Command Center HTML dashboard |
| `GET` | `/api/video_feed` | Multipart MJPEG stream with detection HUD |
| `GET` | `/api/clean_feed` | Clean unobstructed MJPEG camera feed |
| `GET` | `/api/mask_feed` | Color-mapped binary flame mask stream |
| `POST` | `/api/process_frame` | Processes client-side webcam frame from browser |
| `GET` | `/api/status` | Returns current telemetry (alert state, flame area, FPS) |
| `POST` | `/api/config` | Updates `min_area`, `confidence_threshold`, `confirm_frames`, and `cooldown` |
| `GET` | `/api/snapshots` | Lists recorded incident snapshot metadata |
| `POST` | `/api/manual_snapshot`| Captures on-demand snapshot of current frame |
| `DELETE`| `/api/snapshot/<file>`| Deletes an individual snapshot from archive |
| `POST` | `/api/clear_snapshots`| Clears all incident snapshots from disk |

---

## ⚠️ Safety & Legal Disclaimer

> [!WARNING]
> This software is an **experimental computer vision demonstration and research project**. Color/motion only proposes possible regions; an AI classifier must verify each region and a temporal buffer must confirm it. It can still miss real fires or falsely classify non-fire scenes and is **NOT** a certified life-safety device or fire alarm system. Do not rely on it as a primary life-safety mechanism or replacement for certified smoke/fire detectors. For production industrial or residential use, always install certified emergency equipment.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
