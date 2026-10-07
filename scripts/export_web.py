"""Export models for the browser app (web/): PyTorch -> ONNX, checked against PyTorch.

Each model gets the face framing it was trained on: eDv1.0-family models use the
canonical 64x64 template, eD0.x models the looser FER 48x48 template (see
scripts/align_faces.py). Everything the app needs is written to
web/models/manifest.json.

Usage:
  python scripts/export_web.py                 # eD0.7 and every eDv model
  python scripts/export_web.py eD0.5 eDv1.0
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import model_info  # noqa: E402
from src.model import EmotionCNN, build_model  # noqa: E402
from src.predictor import EMOJI, EMOTIONS, available_models  # noqa: E402

OUT = ROOT / "web" / "models"
FER_MEAN, FER_STD = 0.50785, 0.25500


def load(version):
    folder = ROOT / "models" / version
    config = json.loads((folder / "config.json").read_text())
    templates = json.loads((ROOT / "models" / "alignment.json").read_text())
    if config.get("input") == "aligned64":
        model = build_model(config["arch"], config.get("width", 64))
        spec = {"size": config["size"], "template": templates["canonical"]["points"],
                "mean": config["mean"], "std": config["std"]}
    else:
        model = EmotionCNN()
        spec = {"size": 48, "template": templates["fer"]["points"], "mean": FER_MEAN, "std": FER_STD}
    model.load_state_dict(torch.load(folder / "model.pt", map_location="cpu"))
    return model.eval(), spec


def main():
    versions = sys.argv[1:] or [v for v in available_models() if v == "eD0.7" or v.startswith("eDv")]
    OUT.mkdir(parents=True, exist_ok=True)
    entries = []
    for v in versions:
        model, spec = load(v)
        path = OUT / f"{v}.onnx"
        dummy = torch.zeros(1, 1, spec["size"], spec["size"])
        torch.onnx.export(model, dummy, str(path), dynamo=False, opset_version=17,
                          input_names=["input"], output_names=["logits"],
                          dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})

        # The browser must get exactly what PyTorch computes.
        x = torch.randn(16, 1, spec["size"], spec["size"])
        with torch.no_grad():
            expected = model(x).numpy()
        got = ort.InferenceSession(str(path)).run(None, {"input": x.numpy()})[0]
        diff = float(np.abs(expected - got).max())
        assert diff < 1e-3, f"{v}: ONNX output differs from PyTorch by {diff}"

        info = model_info.load(v)
        entries.append({
            "version": v, "file": f"models/{v}.onnx", **spec,
            "tagline": info.tagline, "changes": info.changes,
            "train_images": info.train_images, "dataset": info.dataset,
            "test_ferplus": info.test_ferplus, "test_rafdb": info.test_rafdb, "webcam": info.webcam,
        })
        print(f"{v:<8} -> {path.relative_to(ROOT)} ({path.stat().st_size / 1e6:.1f} MB), "
              f"input {spec['size']}x{spec['size']}, max diff vs PyTorch {diff:.1e}")

    # "About the models" section: every version (live or not) and the comparison charts.
    about = []
    for v in available_models():
        info = model_info.load(v)
        about.append({"version": v, "tagline": info.tagline, "changes": info.changes,
                      "train_images": info.train_images, "dataset": info.dataset,
                      "train_time": info.train_time if info.train_seconds else None,
                      "test_ferplus": info.test_ferplus, "test_rafdb": info.test_rafdb, "webcam": info.webcam,
                      "live": v in versions})
    charts = []
    for src, caption in [("assets/v1/benchmark.png", "Accuracy on FER+, RAF-DB and a simulated webcam"),
                         ("assets/v1/webcam_conditions.png", "Simulated webcam: accuracy per condition"),
                         ("assets/v1/alignment.png", "Every face is aligned the same way in training and live")]:
        if (ROOT / src).exists():
            (ROOT / "web" / "assets").mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / src, ROOT / "web" / "assets" / Path(src).name)
            charts.append({"file": f"assets/{Path(src).name}", "caption": caption})

    manifest = {"emotions": EMOTIONS, "emoji": [EMOJI[e] for e in EMOTIONS],
                "default": versions[-1], "models": entries, "about": about, "charts": charts}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved {(OUT / 'manifest.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
