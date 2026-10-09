<div align="center">

# 🔥 PyroGuard
### Real-Time Computer Vision & AI Fire Early Warning Command Center

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Headless%204.8+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Render](https://img.shields.io/badge/Render-Deployed-46E3B7?style=for-the-badge&logo=render&logoColor=black)](https://render.com/)

**A dual-engine optical fire surveillance system featuring a web command center, browser webcam streaming, real-time flame telemetry, forensic snapshot archive, and trainable deep learning models.**

[🌐 Live Cloud Deployment](https://fire-detection-y22g.onrender.com) • [💻 Local Setup](#-quick-start-local) • [🚀 Deploy to Render](#-cloud-deployment-render) • [🧠 AI Model Training](#-ai-deep-learning-workflow)

---

</div>

## 📌 Overview

**PyroGuard** combines optical computer vision heuristics with deep learning classification to detect combustion signatures in video feeds in real time:

1. **Optical Motion & Colour Heuristic (`dashboard.py` / `fire_detector.py`)**:
   - Analyzes HSV chromatic spectrum (red, orange, yellow luminance) combined with temporal frame differencing to eliminate static warm objects (lamps, sunsets, bright clothing).
   - High performance: runs at 30+ FPS even on low-power devices and standard CPUs.
   - Built-in multi-frame confirmation buffer to prevent false alarms from transient sparks or light flickers.

2. **Deep Learning Classifier (`ai_fire_detector.py` / `train_model.py`)**:
   - Neural network trained on Kaggle fire/non-fire imagery.
   - Evaluates full-frame features and assigns probability confidence scores.

3. **Web Command Center Dashboard**:
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
| **⏱️ Anti-Flicker Confirmation** | Requires configurable consecutive frames (default: `8`) of sustained combustion before triggering alarms. |
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
├── fire_detector.py             # OpenCV Heuristic Engine (Color + Temporal Differencing)
├── ai_fire_detector.py          # PyTorch Inference Engine with Live Video Feed
├── train_model.py               # PyTorch CNN Classifier Training Pipeline
├── download_dataset.py          # Automated Kaggle Dataset Ingestion via KaggleHub
├── alerm_audio.wav              # Primary alert audio chime
├── requirements.txt             # Lightweight production dependencies (Flask, Gunicorn, OpenCV-headless)
├── requirements-ai.txt          # Optional PyTorch & Kaggle dependencies for model training
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
```

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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
python fire_detector.py --source 0 --min-area 2000 --confirm-frames 10
```
*Press `q` or `Esc` to exit.*

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
| **Build Command** | `pip install -r requirements.txt` |
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

### 1. Install Training Dependencies
```bash
pip install -r requirements-ai.txt
```

### 2. Download Kaggle Dataset
```bash
python download_dataset.py
```
*Downloads the Kaggle Fire/Non-Fire image classification dataset into local storage.*

### 3. Train the Classifier
```bash
python train_model.py --data-dir "path/to/fire_dataset" --epochs 12
```
*Trains the CNN model and exports weights to `models/fire_classifier.pt` with benchmark results in `models/metrics.json`.*

### 4. Run the AI Live Detector
```bash
python ai_fire_detector.py --source 0
```

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
| `POST` | `/api/config` | Updates `min_area`, `confirm_frames`, and `cooldown` |
| `GET` | `/api/snapshots` | Lists recorded incident snapshot metadata |
| `POST` | `/api/manual_snapshot`| Captures on-demand snapshot of current frame |
| `DELETE`| `/api/snapshot/<file>`| Deletes an individual snapshot from archive |
| `POST` | `/api/clear_snapshots`| Clears all incident snapshots from disk |

---

## ⚠️ Safety & Legal Disclaimer

> [!WARNING]
> This software is an **experimental computer vision demonstration and research project**. It uses optical color and temporal motion heuristics and is **NOT** a certified life-safety device or fire alarm system. Do not rely on it as a primary life-safety mechanism or replacement for certified smoke/fire detectors. For production industrial or residential use, always install certified emergency equipment.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
