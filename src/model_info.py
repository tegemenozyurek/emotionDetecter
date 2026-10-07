"""Everything the app shows about each model version, read from the files that
training and evaluation leave in models/<version>/ — nothing is hard-coded
except the short human description of what changed."""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

# What each version changed, in plain words. Metrics are read from disk.
DESCRIPTIONS = {
    "eD0.1": {
        "tagline": "Baseline",
        "changes": ["VGG-style CNN trained from scratch", "Original FER-2013 labels", "No augmentation · 40 epochs"],
    },
    "eD0.5": {
        "tagline": "Augmented",
        "changes": ["+ flip / rotate / shift / zoom / erase augmentation",
                    "Stays accurate on tilted & covered faces", "Original FER-2013 labels · 60 epochs"],
    },
    "eD0.7": {
        "tagline": "Clean labels",
        "changes": ["+ FER+ labels: 10 people voted on every image",
                    "Same recipe as eD0.5", "Ambiguous & non-face images removed"],
    },
    "eDv1.0": {
        "tagline": "Video-ready",
        "changes": ["+ RAF-DB: 15k real-world faces", "+ aligned faces (same detector as the live app)",
                    "+ webcam-style augmentation", "Learned from 2 teacher models + eD0.7"],
    },
}

EPOCH_LINE = re.compile(r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)%\s+([\d.]+)\s+([\d.]+)%")
SIZE_LINE = re.compile(r"([\d,]+) train / ([\d,]+) val images")
EPOCH_TIME = re.compile(r"%\s+\S+\s+([\d.]+)s")  # "... val_acc%  lr  time s"
DATASETS = {"fer2013": "FER-2013", "ferplus": "FER+", "v1": "FER+ + RAF-DB"}


@dataclass
class ModelInfo:
    version: str
    tagline: str = ""
    changes: list = field(default_factory=list)
    config: dict = field(default_factory=dict)
    history: list = field(default_factory=list)  # [(epoch, train_acc, val_acc)]
    test_original: float | None = None   # test accuracy on original FER-2013 labels
    test_ferplus: float | None = None    # test accuracy on FER+ labels
    distorted: float | None = None       # FER+ test accuracy with all distortions combined
    test_rafdb: float | None = None      # RAF-DB test accuracy (scripts/benchmark.py)
    webcam: float | None = None          # simulated webcam, all effects combined
    train_images: int | None = None      # images the model learned from
    val_images: int | None = None        # images held out to pick the best epoch
    train_seconds: float = 0.0           # sum of the epoch times in the training log

    @property
    def epochs_done(self):
        return len(self.history)

    @property
    def training(self):
        return self.epochs_done < self.config.get("epochs", 0)

    @property
    def train_time(self):
        """Human-readable training time, e.g. '40 min' or '1 h 32 min'."""
        minutes = round(self.train_seconds / 60)
        return f"{minutes // 60} h {minutes % 60} min" if minutes >= 60 else f"{minutes} min"

    @property
    def dataset(self):
        return DATASETS.get(self.config.get("data", "fer2013"), self.config.get("data"))

    @property
    def best_val(self):
        return max((v for _, _, v in self.history), default=None)

    def tooltip(self):
        lines = [f"{self.version} · {self.tagline}", *self.changes, ""]
        if self.train_images:
            lines.append(f"Trained on {self.train_images:,} images ({self.dataset})")
        if self.train_seconds and not self.training:
            lines.append(f"Training time: {self.train_time} on an Apple M4 GPU")
        if self.training:
            lines.append(f"⏳ training: epoch {self.epochs_done}/{self.config.get('epochs')}")
        if self.test_ferplus is not None:
            lines.append(f"FER+ test accuracy: {self.test_ferplus:.1%}")
        if self.test_rafdb is not None:
            lines.append(f"RAF-DB test accuracy: {self.test_rafdb:.1%}")
        if self.webcam is not None:
            lines.append(f"Simulated webcam: {self.webcam:.1%}")
        elif self.distorted is not None:
            lines.append(f"Distorted faces: {self.distorted:.1%}")
        return "\n".join(lines).strip()


def _test_accuracy(path):
    if not path.exists():
        return None
    match = re.search(r"Test accuracy: ([\d.]+)%", path.read_text())
    return float(match.group(1)) / 100 if match else None


def _robustness():
    """{version: accuracy on 'all combined'} — prefers the FER+-labelled run."""
    for name in ("robustness_ferplus.txt", "robustness.txt"):
        path = MODELS / name
        if path.exists():
            lines = path.read_text().splitlines()
            versions = lines[0].split()[2:]
            combined = next(l for l in lines if l.startswith("all combined")).split()[2:]
            return {v: float(c.rstrip("%")) / 100 for v, c in zip(versions, combined)}, name
    return {}, None


def load(version):
    folder = MODELS / version
    desc = DESCRIPTIONS.get(version, {"tagline": "", "changes": []})
    info = ModelInfo(version, desc["tagline"], desc["changes"])
    if (folder / "config.json").exists():
        info.config = json.loads((folder / "config.json").read_text())
    if (folder / "train_log.txt").exists():
        for line in (folder / "train_log.txt").read_text().splitlines():
            size = SIZE_LINE.search(line)
            if size and info.train_images is None:
                info.train_images, info.val_images = (int(g.replace(",", "")) for g in size.groups())
            m = EPOCH_LINE.match(line)
            if m and (t := EPOCH_TIME.search(line)):
                info.train_seconds += float(t.group(1))
            if m:
                info.history.append((int(m.group(1)), float(m.group(3)) / 100, float(m.group(5)) / 100))
    info.test_original = _test_accuracy(folder / "test_report.txt")
    info.test_ferplus = _test_accuracy(folder / "test_report_ferplus.txt")
    v1 = folder / "test_report_v1.txt"
    if v1.exists():
        text = v1.read_text()
        grab = lambda pattern: float(re.search(pattern, text).group(1)) / 100
        info.test_ferplus = grab(r"FER\+ test .*?accuracy ([\d.]+)%")
        info.test_rafdb = grab(r"RAF-DB test .*?accuracy ([\d.]+)%")
        info.webcam = grab(r"all combined\s+([\d.]+)%")
    robust, source = _robustness()
    if source == "robustness_ferplus.txt":
        info.distorted = robust.get(version)
    return info
