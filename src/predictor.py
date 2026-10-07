"""Find faces in any photo and predict their emotion with the trained CNN."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from src.model import EmotionCNN

ROOT = Path(__file__).resolve().parent.parent
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOJI = {"angry": "😠", "disgust": "🤢", "fear": "😨", "happy": "😄",
         "neutral": "😐", "sad": "😢", "surprise": "😮"}
# Training-set pixel statistics from scripts/preprocess.py (pixels scaled to 0-1).
MEAN, STD = 0.50785, 0.25500
BOX_COLOR = (42, 120, 214)  # RGB


@dataclass
class Face:
    box: tuple  # x, y, w, h
    probs: np.ndarray

    @property
    def emotion(self):
        return EMOTIONS[int(self.probs.argmax())]

    @property
    def confidence(self):
        return float(self.probs.max())


def _overlap(a, b):
    """Intersection area divided by the area of the smaller box (0 = apart, 1 = nested)."""
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return ix * iy / min(a[2] * a[3], b[2] * b[3])


def available_models():
    """Model versions that have trained weights, e.g. ['eD0.1', 'eD0.5'] (oldest first)."""
    versions = [d.name for d in (ROOT / "models").iterdir() if (d / "model.pt").exists()]
    def order(v):  # eD0.1 < eD0.5 < eD0.7 < eDv1.0 < eDv1.1 ...
        numbers = [int(n) for n in v.removeprefix("eD").lstrip("v").split(".") if n.isdigit()]
        return ("v" in v, numbers)
    return sorted(versions, key=order)


def classic_models():
    """eD0.x models: 48x48 FER-framed input, usable with the Haar-cascade pipeline below.
    eDv models need MediaPipe alignment and run in the web app (web/)."""
    import json
    return [v for v in available_models()
            if json.loads((ROOT / "models" / v / "config.json").read_text()).get("input") != "aligned64"]


class EmotionPredictor:
    def __init__(self, version=None, device="cpu"):
        versions = classic_models()
        if not versions:
            raise FileNotFoundError("No trained model in models/ — run scripts/train.py first.")
        self.version = version or versions[-1]  # newest by default
        self.device = torch.device(device)
        self.model = EmotionCNN().to(self.device)
        weights = ROOT / "models" / self.version / "model.pt"
        self.model.load_state_dict(torch.load(weights, map_location=self.device))
        self.model.eval()
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect(self, gray, min_relative_size=0.4, max_overlap=0.2):
        """Find faces, then drop the Haar cascade's typical false positives:
        boxes much smaller than the biggest face (an ear, a fist, a pattern
        in the background) and boxes that overlap a bigger face."""
        min_side = max(40, int(min(gray.shape) * 0.08))
        boxes = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(min_side, min_side))
        boxes = sorted((tuple(int(v) for v in b) for b in boxes), key=lambda b: b[2] * b[3], reverse=True)
        kept = []
        for box in boxes:
            if kept and box[2] < min_relative_size * kept[0][2]:
                continue
            if any(_overlap(box, k) > max_overlap for k in kept):
                continue
            kept.append(box)
        return kept

    @torch.no_grad()
    def classify(self, crops):
        """crops: list of grayscale uint8 face images of any size -> (N, 7) probabilities."""
        x = np.stack([cv2.resize(c, (48, 48), interpolation=cv2.INTER_AREA) for c in crops])
        x = (x.astype(np.float32) / 255.0 - MEAN) / STD
        xb = torch.from_numpy(x).unsqueeze(1).to(self.device)
        # Test-time augmentation: average with the mirrored face.
        probs = (torch.softmax(self.model(xb), 1) + torch.softmax(self.model(torch.flip(xb, dims=[3])), 1)) / 2
        return probs.cpu().numpy()

    def predict(self, rgb):
        """Returns (faces, whole_image_fallback). If no face is found, the whole
        image is treated as one face — handy for already-cropped faces."""
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        boxes = self.detect(gray)
        fallback = not boxes
        if fallback:
            boxes = [(0, 0, gray.shape[1], gray.shape[0])]
        probs = self.classify([gray[y:y + h, x:x + w] for x, y, w, h in boxes])
        return [Face(b, p) for b, p in zip(boxes, probs)], fallback

    @staticmethod
    def annotate(rgb, faces):
        out = rgb.copy()
        scale = max(out.shape[:2]) / 640
        thick = max(2, round(2 * scale))
        for f in faces:
            x, y, w, h = f.box
            cv2.rectangle(out, (x, y), (x + w, y + h), BOX_COLOR, thick)
            label = f"{f.emotion} {f.confidence:.0%}"
            font_scale = 0.7 * scale
            (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
            label_h = th + base + 6
            # Label sits above the box; if there's no room, put it inside the box's top edge.
            top = y - label_h if y >= label_h else y
            cv2.rectangle(out, (x, top), (x + tw + 8, top + label_h), BOX_COLOR, -1)
            cv2.putText(out, label, (x + 4, top + th + 3), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (255, 255, 255), thick, cv2.LINE_AA)
        return out
