# Fire camera detector

A local fire-warning project built with Python. It includes a quick OpenCV heuristic and a more accurate trainable AI classifier based on the Kaggle dataset.

> This is a colour-and-motion heuristic, not a certified fire alarm. Test it carefully and do not rely on it for life-safety or emergency response. For production use, replace the heuristic with a validated fire/smoke model and connect it to an approved alarm system.

## Setup

```powershell
cd fire_camera_detector
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```


## Run Web Command Center (Recommended)

Launch the modern browser dashboard with live video stream, telemetry, sensitivity sliders, and snapshot gallery:

```powershell
.\run_dashboard.bat
```

Or run directly with Python:

```powershell
python dashboard.py
```

Then visit [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

## Run Terminal / OpenCV Window

Use the default camera:


```powershell
python fire_detector.py
```

Use another camera index:

```powershell
python fire_detector.py --source 1
```

Analyse a saved video:

```powershell
python fire_detector.py --source "C:\path\to\video.mp4"
```

Press `q` or `Esc` to stop. When an alert is confirmed, a timestamped snapshot is written to `detections/`.

## AI model workflow (recommended)

The dataset is image-classification data: it has `fire_images` and `non-fire_images` folders, but no bounding-box labels. The AI therefore evaluates the whole frame and reports a fire confidence score.

Download the Kaggle dataset (the first download may require agreeing to Kaggle's terms in your account):

```powershell
python download_dataset.py
```

Copy the printed path, then train the model. If the downloaded directory has an extra `fire_dataset` folder, use that folder as the value for `--data-dir`.

```powershell
python train_model.py --data-dir "C:\path\to\fire_dataset" --epochs 12
```

Run the trained live detector:

```powershell
python ai_fire_detector.py --source 0
```

Or run it on a saved video:

```powershell
python ai_fire_detector.py --source "C:\path\to\video.mp4"
```

The AI alert needs six consecutive frames above 85% confidence by default. On Windows it emits an audible beep, saves a snapshot, and adds the event to `detections/alerts.csv`. Increase `--threshold` to reduce false alerts, or increase `--confirm-frames` to require a longer confirmation.

## Adjusting sensitivity

`--min-area` controls the minimum detected flame area (default `1200` pixels). Raise it to reduce false alerts, or lower it for a distant/small flame. `--confirm-frames` controls how many consecutive frames must contain a candidate before an alert (default `8`).

Example:

```powershell
python fire_detector.py --source 0 --min-area 2000 --confirm-frames 10
```
