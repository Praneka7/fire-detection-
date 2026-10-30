<div align="center">

# 🔥 PyroGuard
### Intelligent Real-Time Computer Vision & AI Fire Early Warning Command Center

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Headless%204.8+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gunicorn](https://img.shields.io/badge/Gunicorn-21.2+-499848?style=for-the-badge&logo=gunicorn&logoColor=white)](https://gunicorn.org/)
[![Render](https://img.shields.io/badge/Render-Deployed-46E3B7?style=for-the-badge&logo=render&logoColor=black)](https://render.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

**PyroGuard** is an end-to-end, enterprise-grade optical fire surveillance and early warning platform. It combines dual-stage optical candidate extraction (color incandescence, Kovac/Peer skin rejection, Haar face suppression, and inter-frame turbulence differencing) with a high-accuracy **EfficientNet-B0 deep convolutional neural network**, verified through a multi-frame temporal confirmation buffer to eliminate false alarms.

Featuring a **Cyber Obsidian Glassmorphism Command Center**, PyroGuard operates seamlessly across local physical webcams, RTSP IP camera streams, pre-recorded video feeds, and remote cloud deployments using client-side HTML5 browser camera streaming.

[🌐 Live Cloud Deployment](https://fire-detection-y22g.onrender.com) • [🚀 Quick Start](#-quick-start-local) • [🏗️ Architecture](#️-system-architecture) • [🎛️ Command Center UI](#-web-command-center-features) • [🧠 AI Model & Training](#-deep-learning-pipeline-efficientnet-b0) • [📡 API Reference](#-rest-api-reference)

---

</div>

## 📑 Table of Contents

- [📌 Project Overview](#-project-overview)
- [✨ Key Capabilities](#-key-capabilities)
- [🏗️ System Architecture](#️-system-architecture)
  - [End-to-End Pipeline Workflow](#end-to-end-pipeline-workflow)
  - [Candidate Proposal vs. AI Verification](#candidate-proposal-vs-ai-verification)
- [🎛️ Web Command Center Features](#-web-command-center-features)
- [📁 Project Directory Structure](#-project-directory-structure)
- [💻 Quick Start (Local Setup)](#-quick-start-local-setup)
  - [Prerequisites](#prerequisites)
  - [1. Clone Repository](#1-clone-repository)
  - [2. Virtual Environment Setup](#2-virtual-environment-setup)
  - [3. Run the Web Dashboard](#3-run-the-web-dashboard)
- [🖥️ Running Standalone CLI & OpenCV Tools](#️-running-standalone-cli--opencv-tools)
  - [A. Fire Detector (`fire_detector.py`)](#a-fire-detector-fire_detectorpy)
  - [B. AI Fire Detector (`ai_fire_detector.py`)](#b-ai-fire-detector-ai_fire_detectorpy)
  - [C. Video Benchmark Evaluator (`evaluate_videos.py`)](#c-video-benchmark-evaluator-evaluate_videospy)
- [🧠 Deep Learning Pipeline (EfficientNet-B0)](#-deep-learning-pipeline-efficientnet-b0)
  - [Dataset Sourcing & Kaggle Hub](#1-dataset-sourcing--kaggle-hub)
  - [Dataset Layout Structure](#2-dataset-layout-structure)
  - [Model Training & Stratified Validation](#3-model-training--stratified-validation)
  - [Metrics & Performance Reporting](#4-metrics--performance-reporting)
- [☁️ Cloud Deployment (Render / Containers)](#️-cloud-deployment-render--containers)
  - [Deploying to Render via Blueprint](#deploying-to-render-via-blueprint)
  - [Gunicorn Multi-Threading Configuration](#why-gunicorn--w-1---threads-8---timeout-0)
  - [Client-Side Browser Webcam Streaming](#client-side-browser-webcam-streaming)
- [📡 REST API Reference](#-rest-api-reference)
- [🧪 Testing & Quality Assurance](#-testing--quality-assurance)
- [⚙️ Configuration Parameters](#️-configuration-parameters)
- [❓ Frequently Asked Questions (FAQ)](#-frequently-asked-questions-faq)
- [⚠️ Safety & Legal Disclaimer](#️-safety--legal-disclaimer)
- [📄 License](#-license)

---

## 📌 Project Overview

Conventional optical fire detectors rely solely on simplistic chromatic thresholds (such as bright yellow or red pixels) or naive motion differencing. Consequently, they are notorious for triggering costly false alarms whenever human skin, red garments, tungsten lamps, sunlight glints, vehicle headlights, or warm-toned moving objects enter the frame.

**PyroGuard** solves this fundamental challenge using a **hybrid multi-stage verification pipeline**:

1. **High-Speed Optical Region Proposal**: Incoming video frames are analyzed using dynamic HSV incandescence rules, Kovac/Peer skin tone chromatic filtering, Haar-cascade face masking, and adaptive motion differencing. This step acts as a fast candidate proposer, extracting regions of interest (ROIs) that exhibit flame-like spectral and temporal properties.
2. **Deep Convolutional AI Verification**: Every candidate ROI is extracted, normalized, and classified by a custom-trained **EfficientNet-B0** binary neural network. Only ROIs with an inference probability exceeding the configurable threshold (default `90%`) are accepted.
3. **Temporal Anti-Flicker Confirmation**: The system maintains an atomic frame counter (`ConsecutiveFireConfirmation`). A confirmed alert is triggered only when AI-verified fire persists across a uninterrupted streak of consecutive frames (default `15` frames in dashboard, `8` in standalone mode). Any intermediate negative or low-confidence frame instantly resets the counter to zero.
4. **Autonomous Alarm & Forensic Dispatch**: Once confirmed, the system sounds multi-channel alerts (Web Audio API synthesis in browser + OS-level audio chimes) and automatically timestamps and archives high-resolution evidence snapshots.

---

## ✨ Key Capabilities

- **🌐 Universal Cloud & Local Streaming**: Works out-of-the-box with physical USB webcams, RTSP network streams, video files, and **cloud-hosted servers** via client-side WebRTC HTML5 camera streaming (`navigator.mediaDevices.getUserMedia`).
- **🎯 3-Way Optical Telemetry Feeds**:
  - **Detection HUD**: High-resolution feed displaying bounding boxes, real-time AI confidence scores, and flame pixel area ($px^2$).
  - **Clean HD Feed**: Unmodified, pristine camera video feed.
  - **Amber Flame Mask**: Color-mapped binary segmentation mask for visual algorithm debugging.
- **🛡️ Multi-Layer False Positive Suppression**:
  - *Kovac / Peer Skin Filter*: Suppresses hand, arm, and skin chromatic signatures.
  - *Haar Cascade Face Suppression*: Automatically zeros out human faces and heads to prevent false alarms on facial features.
  - *Global Pan Detection*: Suppresses spurious motion triggers caused by camera shakes or rapid panning when $> 35\%$ of the frame changes simultaneously.
- **⚡ Fail-Closed or Graceful Fallback**: By default, dashboard operations feature an optical fallback when weights are missing, or strict fail-closed AI enforcement (`--strict-ai` in CLI) ensuring heuristic-only triggers never sound accidental alarms.
- **📊 Real-Time Telemetry Dashboard**: Live updates for active flame regions, cumulative flame area ($px^2$), alert incident count, confirmation buffer progress meter, and real-time FPS.
- **📸 Forensic Snapshot Gallery**: Automatic timestamped alert captures with interactive modal lightbox, image download, individual snapshot removal, and bulk archive wiping.
- **🎛️ Live Parameter Tuning**: Sliders on the dashboard allow real-time adjustment of Minimum Flame Area, AI Confidence Threshold, Confirmation Frames, and Alert Cooldown without server restarts.

---

## 🏗️ System Architecture

### End-to-End Pipeline Workflow

```mermaid
flowchart TD
    A["Input Video Stream<br/>(Local Webcam / RTSP / Browser WebRTC)"] --> B["Frame Acquisition & Normalization<br/>(1280x720 / 30 FPS)"]
    B --> C["Color & Incandescence Segmentation<br/>(Strict Dual-Range HSV + RGB Thresholds)"]
    
    C --> D["Human Skin & Face Rejection<br/>(Kovac/Peer RGB+YCrCb & Haar Cascades)"]
    D --> E["Inter-Frame Motion Differencing<br/>(cv2.absdiff with Camera Pan Suppression)"]
    E --> F["Morphological Refinement<br/>(Dilation, Opening, Closing Filters)"]
    
    F --> G{"Candidate Contours Found?<br/>(Area >= min_area)"}
    G -- "No" --> H["Reset Consecutive Counter<br/>(Streak = 0)"]
    G -- "Yes" --> I["Extract Bounding Box ROIs"]
    
    I --> J{"AI Model Loaded?<br/>(EfficientNet-B0)"}
    J -- "No (Fallback Mode)" --> K["Accept Heuristic Regions"]
    J -- "Yes (AI Verification)" --> L["Preprocess ROI (224x224, Normalized)<br/>Forward Pass on PyTorch Backbone"]
    
    L --> M{"AI Confidence >= Threshold?<br/>(Default: 90%)"}
    M -- "No (Hard Negative Rejection)" --> H
    M -- "Yes (Verified Fire)" --> N["Increment Temporal Counter<br/>(Streak = Streak + 1)"]
    K --> N
    
    N --> O{"Streak >= Confirm Frames?<br/>(Default: 15 frames)"}
    O -- "Buffering" --> P["Update Telemetry & Progress Bar<br/>(Alert = False)"]
    O -- "Confirmed Fire" --> Q{"Cooldown Elapsed?<br/>(now - last_saved >= cooldown)"}
    
    Q -- "Yes" --> R["TRIGGER CRITICAL ALERT"]
    Q -- "No (Cooldown Active)" --> P
    
    R --> S["🔊 Browser Web Audio Siren Synthesizer"]
    R --> T["🔔 Native OS System Audio Chime"]
    R --> U["📸 Save Forensic Snapshot to /detections"]
    R --> V["📝 Append Record to CSV Incident Log"]
```

### Candidate Proposal vs. AI Verification

| Processing Stage | Mechanism / Algorithm | Primary Purpose |
| :--- | :--- | :--- |
| **Stage 1: Flame Color Filter** | Dual-spectrum HSV: Deep orange/red $(H \in [0,16] \cup [165,180], S \ge 135, V \ge 165, R \ge 190, R-B \ge 80)$ and incandescent core $(H \in [17,21], S \ge 180, V \ge 220, R \ge 220, R-G \ge 40, B \le 60)$. | Isolate pixel clusters exhibiting combustion chromaticity. |
| **Stage 2: Skin Rejection** | Kovac/Peer model: $(R > 95 \land G > 40 \land B > 20 \land R-G > 15 \land R > B) \land (133 \le Cr \le 173 \land 77 \le Cb \le 127)$. | Strip out skin tones from hands, arms, and body reflections. |
| **Stage 3: Face Suppression** | OpenCV Haar Feature-based Cascade (`haarcascade_frontalface_default.xml`) with dynamic padding. | Mask detected human heads and faces to zero. |
| **Stage 4: Motion Differencing** | Absolute grayscale diff (`cv2.absdiff(gray, prev_gray) > 15`) with camera pan suppression (kills motion if $>35\%$ frame changes). | Ensure static warm objects (lamps, heaters, clothing) are discarded. |
| **Stage 5: AI Verification** | Deep Convolutional **EfficientNet-B0** classifier trained on fire/non-fire image sets. | Verify actual combustion texture, turbulent flame tongues, and ambient combustion context. |
| **Stage 6: Temporal Filter** | Strict consecutive streak buffer (`ConsecutiveFireConfirmation`). | Eliminate instantaneous flashes, spark flickers, or momentary model noise. |

---

## 🎛️ Web Command Center Features

The PyroGuard dashboard (`dashboard.py`) delivers a responsive cyber-defense monitoring experience:

- **Cyber Obsidian Glassmorphism UI**: Built with semantic HTML5, modern CSS custom properties, and typography driven by Google Fonts (*Outfit* and *JetBrains Mono*).
- **Client-Side Browser Webcam Ingestion**:
  When hosted in cloud environments (Render, AWS EC2, Heroku) where `/dev/video0` does not exist, the dashboard displays a cloud standby canvas with an animated radar sweep. Users click **"Activate My Webcam"** to grant browser camera access, streaming frames via Base64/Blob directly to `/api/process_frame`.
- **Tri-View Feed Selector**: Switch between `Detection HUD`, `Clean Feed`, and `Flame Mask` without interrupting monitoring.
- **Dynamic Sensitivity Sliders**:
  - **Min Flame Area**: $200\text{ px}$ to $5,000\text{ px}$ (filters out micro-reflections).
  - **AI Fire Confidence**: $50\%$ to $99\%$ (strictness of neural verification).
  - **Consecutive Confirmation Frames**: $2$ to $30\text{ frames}$ (anti-flicker delay).
  - **Alert Cooldown**: $2\text{ s}$ to $60\text{ s}$ (prevents disk saturation during persistent alerts).
- **Forensic Snapshot Archive**:
  - Live gallery rendered directly from `/detections/` directory.
  - Interactive full-resolution modal preview.
  - One-click manual capture button (`Capture Snapshot`).
  - Single-image deletion and bulk wipe functionality.
- **Dual Sound Siren**:
  - Client-side Web Audio API synthesizer generates an emergency rising/falling frequency sweep ($700\text{ Hz} \leftrightarrow 1200\text{ Hz}$).
  - Server-side platform audio via Python `winsound` / system chime.

---

## 📁 Project Directory Structure

```text
fire-detection-/
├── models/
│   ├── fire_classifier.pt       # EfficientNet-B0 model checkpoint (PyTorch)
│   └── metrics.json             # Validation & test evaluation metrics
├── static/
│   ├── css/
│   │   └── style.css            # Cyber Obsidian UI & Glassmorphism Design System
│   └── js/
│       └── app.js               # Frontend Engine, Web Audio Siren & Webcam Streamer
├── templates/
│   └── index.html               # Semantic HTML5 Web Command Center UI
├── tests/
│   └── test_fire_detector.py    # Unit tests for AI gating, fallback & streak resets
├── detections/                  # Output directory for saved incident snapshots & logs
├── ai_fire_detector.py          # Standalone headless/GUI AI-verified detector runner
├── alerm_audio.wav              # Primary alert audio chime (1100 Hz emergency tone)
├── dashboard.py                 # Flask Command Center, MJPEG streaming & REST endpoints
├── download_dataset.py          # Automated Kaggle Dataset Ingestion via KaggleHub
├── evaluate_videos.py           # Clip-level benchmarking tool for positive & negative videos
├── fire_detector.py             # Core optical region proposal, filtering & OpenCV runner
├── fire_model.py                # Fail-closed EfficientNet-B0 loader & ROI inference engine
├── render.yaml                  # Infrastructure-as-Code Blueprint for Render cloud hosting
├── requirements.txt             # Core production dependencies (Flask, Gunicorn, OpenCV)
├── requirements-ai.txt          # PyTorch, Torchvision, KaggleHub, Pillow dependencies
├── run.bat                      # Windows 1-Click OpenCV Detector launcher
├── run_dashboard.bat            # Windows 1-Click Dashboard launcher (auto-opens browser)
├── train_model.py               # Deep learning training pipeline with stratified splits
└── README.md                    # Comprehensive Project Documentation
```

---

## 💻 Quick Start (Local Setup)

### Prerequisites

- **Python**: 3.10 or 3.11 recommended.
- **Hardware**: Any standard webcam (built-in or USB) or video file. CPU inference is fast on modern processors; CUDA GPU supported automatically if available.

### 1. Clone Repository

```bash
git clone https://github.com/Praneka7/fire-detection-.git
cd fire-detection-
```

### 2. Virtual Environment Setup

#### Windows (PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Core requirements (Dashboard & Optical Detector)
pip install -r requirements.txt

# Optional AI dependencies (EfficientNet-B0 inference & training)
pip install -r requirements-ai.txt
```

#### macOS / Linux (Bash)
```bash
python3 -m venv .venv
source .venv/bin/activate

# Core requirements
pip install -r requirements.txt

# Optional AI dependencies
pip install -r requirements-ai.txt
```

### 3. Run the Web Dashboard

#### Option A: One-Click Launcher (Windows)
Double-click `run_dashboard.bat` or run in PowerShell:
```powershell
.\run_dashboard.bat
```
*This starts the Flask server and opens `http://127.0.0.1:5000` automatically.*

#### Option B: Terminal Command
```bash
python dashboard.py
```
Open your browser and navigate to:
👉 **`http://127.0.0.1:5000`**

---

## 🖥️ Running Standalone CLI & OpenCV Tools

In addition to the Web Dashboard, PyroGuard includes dedicated standalone command-line tools for direct OpenCV window streaming, automated CSV logging, and video dataset evaluation.

### A. Fire Detector (`fire_detector.py`)

Launches a native OpenCV desktop window with bounding box overlays, candidate contours, and optical tracking.

```bash
# Default physical webcam (Camera 0)
python fire_detector.py

# External camera device (Camera index 1)
python fire_detector.py --source 1

# Process a pre-recorded video file
python fire_detector.py --source "path/to/test_video.mp4"

# Tune sensitivity via CLI arguments
python fire_detector.py --source 0 --min-area 800 --confidence 0.92 --confirm-frames 12

# Enforce strict AI mode (disables heuristic fallback if AI weights missing)
python fire_detector.py --source 0 --strict-ai
```

*Hotkeys inside OpenCV window:*
- Press `q` or `Esc` to safely quit.

---

### B. AI Fire Detector (`ai_fire_detector.py`)

A production-focused runner that enforces strict AI verification, logs all confirmed incidents to `detections/alerts.csv`, and saves timestamped alert frames.

```bash
# Run AI detector with default settings
python ai_fire_detector.py --source 0 --confidence 0.90 --min-area 350 --confirm-frames 8

# Test the siren audio chime and exit
python ai_fire_detector.py --test-alarm

# Custom cooldown and output configuration
python ai_fire_detector.py --source 0 --cooldown 20 --threshold 0.95
```

---

### C. Video Benchmark Evaluator (`evaluate_videos.py`)

Benchmark the full detection pipeline against a curated suite of positive fire clips and challenging hard-negative video clips (hands, waving arms, red garments, tungsten lamps).

```text
evaluation-clips/
├── fire/
│   ├── campfire.mp4
│   └── stove_flame.mp4
└── non-fire/
    ├── hands/
    │   └── waving_hands.mp4
    ├── clothing/
    │   └── red_jacket_movement.mp4
    └── lighting/
        └── incandescent_lamp.mp4
```

Execute evaluation across all categories:
```bash
python evaluate_videos.py \
  --fire-dir "evaluation-clips/fire" \
  --non-fire-dir "evaluation-clips/non-fire" \
  --confidence 0.90 \
  --confirm-frames 8 \
  --output "video_evaluation.json"
```

The script outputs a comprehensive JSON report containing:
- True Positives, False Positives, True Negatives, False Negatives.
- Clip-level precision, recall, and specificity.
- Peak verified confidence per video clip.

---

## 🧠 Deep Learning Pipeline (EfficientNet-B0)

PyroGuard utilizes **EfficientNet-B0**, a state-of-the-art convolutional architecture optimized for high classification accuracy with low computational latency, making it ideal for real-time edge and server inference.

```
Input Candidate ROI (variable size)
   │
   ▼
Letterbox / Resize to 224x224 RGB
   │
   ▼
ImageNet Normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   │
   ▼
EfficientNet-B0 Feature Extractor (MBConv Blocks)
   │
   ▼
Adaptive Average Pooling (1280 features)
   │
   ▼
Dropout (p=0.2)
   │
   ▼
Linear Classifier (1280 -> 2 classes: non_fire_images, fire_images)
   │
   ▼
Softmax Probability Distribution
```

### 1. Dataset Sourcing & Kaggle Hub

PyroGuard includes a dedicated dataset downloader using `kagglehub`:

```bash
python download_dataset.py
```
This automatically fetches the curated Kaggle Fire/Non-Fire dataset into your local Kaggle cache.

### 2. Dataset Layout Structure

Ensure your dataset directory is organized as a clean two-class PyTorch `ImageFolder` structure:

```text
dataset/
├── fire_images/
│   ├── fire_0001.jpg
│   ├── fire_0002.jpg
│   └── ...
└── non_fire_images/
    ├── hand_gestures_001.jpg
    ├── skin_closeups_002.jpg
    ├── orange_tshirt_003.jpg
    ├── tungsten_lamp_004.jpg
    └── ...
```

> [!IMPORTANT]
> **Hard Negatives are Vital**: Standard open-source fire datasets are frequently imbalanced and lack close-up footage of hands, skin, and warm textiles moving in front of webcams. Augment `non_fire_images` with real camera footage of your intended deployment environment.

### 3. Model Training & Stratified Validation

Run `train_model.py` to train the classifier with balanced class sampling, random augmentations, and evaluation:

```bash
# Standard training with internal 70/15/15 stratified split
python train_model.py --data-dir "path/to/dataset" --epochs 12 --batch-size 32 --lr 0.0003 --threshold 0.90

# Training with a dedicated external test directory
python train_model.py --data-dir "path/to/train_data" --test-dir "path/to/test_data" --epochs 15
```

Key training features:
- **WeightedRandomSampler**: Dynamically balances class representation to prevent bias toward non-fire images.
- **Data Augmentation**: Random horizontal flips, affine rotations ($\pm 10^\circ$), and subtle color jittering.
- **Checkpoint Serialization**: Saves the best model weights, optimizer state, class mappings, and timestamp to `models/fire_classifier.pt`.

### 4. Metrics & Performance Reporting

Training logs and evaluation metrics are written directly to `models/metrics.json`:

```json
{
  "test_loss": 0.0545,
  "test_accuracy": 0.9867,
  "best_validation_accuracy": 0.9933,
  "precision": 0.9821,
  "recall": 0.9910,
  "specificity": 0.9825,
  "f1_score": 0.9865
}
```

---

## ☁️ Cloud Deployment (Render / Containers)

PyroGuard is fully cloud-native and pre-configured for one-click deployment on **[Render](https://render.com/)**.

### Deploying to Render via Blueprint

The repository includes a ready-to-use [`render.yaml`](render.yaml) specification:

1. Push or fork this repository to your GitHub account.
2. Log into the [Render Dashboard](https://dashboard.render.com).
3. Select **New +** > **Blueprint**.
4. Select your `fire-detection-` repository.
5. Render will automatically parse `render.yaml` and configure the web service:
   - **Environment**: Python 3.11.9
   - **Build Command**: `pip install -r requirements.txt -r requirements-ai.txt`
   - **Start Command**: `gunicorn -w 1 --threads 8 --timeout 0 -b 0.0.0.0:$PORT dashboard:app`

### Why Gunicorn `-w 1 --threads 8 --timeout 0`?

- **`--timeout 0`**: MJPEG video feeds (`/api/video_feed`) hold long-lived HTTP multipart connections open indefinitely. Default 30-second timeouts would prematurely kill the stream.
- **`--threads 8`**: Allows multiple simultaneous browser clients to receive the video stream, adjust sliders, and query telemetry without blocking worker I/O.
- **`-w 1`**: Keeps camera state, alert streaks, and thread synchronization coordinated within a single process.

### Client-Side Browser Webcam Streaming

Cloud virtual machines do not possess physical camera hardware. PyroGuard gracefully addresses this:
1. When launched in the cloud, `dashboard.py` detects the absence of `/dev/video0` and enters **Standby Mode** displaying an animated surveillance screen.
2. When a user opens the dashboard in Chrome, Firefox, or Edge, a banner prompts: **"Cloud Standby Mode — Click 'Activate My Webcam'"**.
3. Upon clicking, HTML5 `getUserMedia` acquires the client's local camera.
4. JavaScript captures video frames at 15–20 FPS onto a hidden `<canvas>`, encodes them as JPEG, and posts them to `/api/process_frame`.
5. The cloud backend processes each frame through the full detection and AI pipeline, returning bounding box coordinates and telemetry in real time!

---

## 📡 REST API Reference

The Flask Command Center exposes a RESTful API:

### 1. Video & Stream Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the primary Command Center HTML dashboard. |
| `GET` | `/api/video_feed` | Multipart MJPEG stream with detection bounding boxes & AI confidence HUD. |
| `GET` | `/api/clean_feed` | Unobstructed, clean MJPEG camera stream. |
| `GET` | `/api/mask_feed` | Amber-colorized binary flame segmentation mask stream. |
| `POST`| `/api/process_frame` | Ingests client-side browser webcam frame (Base64 JPEG) and returns JSON telemetry & processed imagery. |

#### Client Frame Ingestion Payload (`POST /api/process_frame`):
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ...",
  "mode": "processed"
}
```
*Response:*
```json
{
  "status": "ok",
  "alert": false,
  "candidate_frames": 3,
  "regions_count": 1,
  "flame_area": 840,
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

---

### 2. Telemetry & Control Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/status` | Returns system telemetry (alert state, flame area, streak buffer, FPS, active mode). |
| `POST`| `/api/config` | Updates sensitivity parameters on the fly without server restart. |

#### Config Update Payload (`POST /api/config`):
```json
{
  "min_area": 1000,
  "confidence_threshold": 0.92,
  "confirm_frames": 12,
  "cooldown": 15.0
}
```

---

### 3. Forensic Snapshot Archive Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/snapshots` | Returns a JSON array of all saved incident snapshots with metadata. |
| `GET` | `/detections/<filename>` | Serves raw high-resolution image file for viewing or downloading. |
| `POST`| `/api/manual_snapshot` | On-demand capture and disk save of the current camera frame. |
| `DELETE`| `/api/snapshot/<filename>`| Permanently deletes an individual snapshot. |
| `POST`| `/api/clear_snapshots` | Wipes all saved incident snapshots from disk. |

---

## 🧪 Testing & Quality Assurance

PyroGuard maintains strict unit test coverage verifying that the AI gating mechanisms, streak buffers, and fail-closed security logic operate correctly:

```bash
# Run the test suite
python -m unittest discover -s tests -v
```

### Verified Test Cases:
- `test_confirmation_requires_consecutive_verified_frames_and_resets`: Validates that temporal confirmation strictly requires consecutive frames and that a single non-fire frame resets the counter.
- `test_warm_moving_candidate_is_rejected_when_ai_says_non_fire`: Asserts that warm, moving objects (e.g. hands or clothing) proposing candidate contours are rejected when the AI scores them below threshold.
- `test_region_is_accepted_only_above_ai_threshold`: Verifies that bounding boxes are accepted only when confidence is $\ge$ the configured threshold.
- `test_unavailable_ai_model_fails_closed`: Confirms that missing or corrupt model files fail safely, preventing unverified alerts in strict mode.
- `test_static_warm_object_is_not_a_motion_candidate`: Asserts that static warm objects without motion differencing do not generate candidates.
- `test_unavailable_ai_model_optical_fallback_detects_fire`: Confirms that heuristic fallback mode allows emergency operation when AI dependencies are deliberately disabled.

---

## ⚙️ Configuration Parameters

All key runtime parameters can be adjusted via CLI flags, environment variables, or the Web Dashboard UI:

| Parameter | Default | Range / Type | Description |
| :--- | :---: | :---: | :--- |
| `source` | `0` | `int` or `str` | Video source: webcam index (`0`, `1`), video file path, or RTSP stream URL. |
| `min_area` | `1200` | $200 - 5000\text{ px}$ | Minimum contour area in pixels required to constitute a flame candidate. |
| `confidence_threshold` | `0.90` | $0.50 - 0.999$ | Minimum EfficientNet-B0 probability score required to verify combustion. |
| `confirm_frames` | `15` | $2 - 30\text{ frames}$ | Consecutive verified frames required before triggering an alert. |
| `cooldown` | `10.0` | $2.0 - 60.0\text{ s}$ | Minimum quiet period between automated snapshot disk writes. |
| `allow_fallback` | `True` | `bool` | Allows optical-only detection if PyTorch or model weights are missing. |
| `FIRE_MODEL_PATH` | `models/fire_classifier.pt` | `Path` | Environment variable pointing to custom PyTorch model weights. |

---

## ❓ Frequently Asked Questions (FAQ)

<details>
<summary><b>1. Why does the model file show as only 133 bytes in Git?</b></summary>
The repository uses Git LFS (Large File Storage) for binary checkpoints. If you cloned without Git LFS installed, run:
```bash
git lfs install
git lfs pull
```
Alternatively, train a customized model using `python train_model.py` or run in optical mode.
</details>

<details>
<summary><b>2. Can PyroGuard run on a Raspberry Pi or low-power edge device?</b></summary>
Yes. In optical mode, PyroGuard runs smoothly at 30+ FPS on a Raspberry Pi 4/5. When running the EfficientNet-B0 AI model, inference speeds of 10–18 FPS can be achieved using PyTorch with CPU optimizations or by exporting the backbone to ONNX / OpenVINO.
</details>

<details>
<summary><b>3. Why did the detector propose a box on my hand or red shirt?</b></summary>
That is Stage 1 (candidate proposal) in action. PyroGuard deliberately casts a wide optical net to avoid missing faint flames. If the AI model is active, the neural network evaluates the region and discards it within milliseconds. You will see the counter remain at zero.
</details>

<details>
<summary><b>4. How do I stream an IP Security Camera (RTSP)?</b></summary>
Pass your camera's RTSP URL into either the CLI or dashboard:
```bash
python dashboard.py --source "rtsp://admin:password@192.168.1.100:554/h264Preview_01_main"
```
</details>

---

## ⚠️ Safety & Legal Disclaimer

> [!CAUTION]
> **EXPERIMENTAL RESEARCH AND DEMONSTRATION SOFTWARE ONLY**
> 
> PyroGuard is an academic and experimental computer vision project. While it employs advanced computer vision algorithms and deep convolutional neural networks, optical detection can experience missed detections or false alarms due to lighting variations, smoke occlusion, or environmental factors.
> 
> **THIS SOFTWARE IS NOT A CERTIFIED LIFE-SAFETY DEVICE, CERTIFIED FIRE ALARM, OR SMOKE DETECTOR.**
> Do **NOT** rely on PyroGuard as a primary fire safety system or replacement for certified smoke, heat, or flame detectors compliant with NFPA, EN 54, or local fire safety regulations. The authors and contributors assume no liability for damages or losses resulting from the use of this software.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for complete details.

---

<div align="center">

**Developed with ❤️ by [Praneka](https://github.com/Praneka7)**

*Star ⭐ this repository if you find it helpful!*

</div>
