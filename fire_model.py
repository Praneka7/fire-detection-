"""Shared, fail-closed EfficientNet fire classifier for every live detector path."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class FireModel:
    """Load a trained fire/non-fire EfficientNet-B0 and score image regions.

    Missing dependencies, Git LFS pointer files, incompatible checkpoints, and
    inference errors are reported as unavailable; they never fall back to HSV-only
    fire alerts.
    """

    def __init__(self, model_path: Path, threshold: float = 0.90) -> None:
        self.model_path = Path(model_path)
        self.threshold = float(np.clip(threshold, 0.50, 0.999))
        self.available = False
        self.error: str | None = None
        self.device: Any = None
        self.model: Any = None
        self.fire_index: int | None = None
        self._image_class: Any = None
        self._transform: Any = None
        self._torch: Any = None
        self._inference_lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if not self.model_path.is_file():
            self.error = f"Model file not found: {self.model_path}"
            return
        try:
            with self.model_path.open("rb") as model_file:
                header = model_file.read(128)
            if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise ValueError("Model is a Git LFS pointer, not model weights. Pull/download the LFS file.")

            import torch
            from PIL import Image
            from torchvision import models, transforms

            self._torch = torch
            self._image_class = Image
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            if not isinstance(checkpoint, dict) or "model_state" not in checkpoint or "class_to_idx" not in checkpoint:
                raise ValueError("Checkpoint must contain model_state and class_to_idx.")
            architecture = checkpoint.get("architecture", "efficientnet_b0")
            if architecture != "efficientnet_b0":
                raise ValueError(f"Unsupported checkpoint architecture: {architecture!r} (expected efficientnet_b0).")
            class_to_idx = checkpoint["class_to_idx"]
            if "fire_images" not in class_to_idx or "non_fire_images" not in class_to_idx:
                raise ValueError("Checkpoint classes must include fire_images and non_fire_images.")
            if len(class_to_idx) != 2 or set(class_to_idx.values()) != {0, 1}:
                raise ValueError("Checkpoint must contain exactly two classes indexed 0 and 1.")

            self.model = models.efficientnet_b0(weights=None)
            self.model.classifier[1] = torch.nn.Linear(self.model.classifier[1].in_features, len(class_to_idx))
            self.model.load_state_dict(checkpoint["model_state"], strict=True)
            self.model.eval().to(self.device)
            self.fire_index = int(class_to_idx["fire_images"])
            self._transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            self.available = True
        except Exception as exc:  # fail closed if optional runtime/weights cannot load
            self.available = False
            self.error = f"Fire model unavailable: {exc}"

    def predict_region(self, frame: np.ndarray, box: tuple[int, int, int, int]) -> float | None:
        """Return fire probability for a padded candidate crop, or None on failure."""
        if not self.available:
            return None
        x, y, width, height = box
        pad_x = max(4, int(width * 0.30))
        pad_y = max(4, int(height * 0.30))
        left = max(0, x - pad_x)
        top = max(0, y - pad_y)
        right = min(frame.shape[1], x + width + pad_x)
        bottom = min(frame.shape[0], y + height + pad_y)
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            return None

        try:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = self._transform(self._image_class.fromarray(rgb)).unsqueeze(0).to(self.device)
            with self._inference_lock, self._torch.inference_mode():
                logits = self.model(tensor)
                return float(self._torch.softmax(logits, dim=1)[0, self.fire_index].item())
        except Exception as exc:
            self.error = f"Fire model inference failed: {exc}"
            self.available = False
            return None
