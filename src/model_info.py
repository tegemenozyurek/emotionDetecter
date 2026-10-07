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
}

EPOCH_LINE = re.compile(r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)%\s+([\d.]+)\s+([\d.]+)%")
SIZE_LINE = re.compile(r"([\d,]+) train / ([\d,]+) val images")
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
    train_images: int | None = None      # images the model learned from
    val_images: int | None = None        # images held out to pick the best epoch

    @property
    def epochs_done(self):
        return len(self.history)

    @property
    def training(self):
        return self.epochs_done < self.config.get("epochs", 0)

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
        if self.training:
            lines.append(f"⏳ training: epoch {self.epochs_done}/{self.config.get('epochs')}")
        if self.test_ferplus is not None:
            lines.append(f"FER+ test accuracy: {self.test_ferplus:.1%}")
        if self.distorted is not None:
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
            if m:
                info.history.append((int(m.group(1)), float(m.group(3)) / 100, float(m.group(5)) / 100))
    info.test_original = _test_accuracy(folder / "test_report.txt")
    info.test_ferplus = _test_accuracy(folder / "test_report_ferplus.txt")
    robust, source = _robustness()
    if source == "robustness_ferplus.txt":
        info.distorted = robust.get(version)
    return info
