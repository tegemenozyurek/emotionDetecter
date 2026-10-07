"""Benchmark every model on the same faces: FER+ test, RAF-DB test and a simulated webcam.

Each model gets the framing it was trained on (eD0.x: FER framing at 48x48,
eDv: canonical alignment at 64x64), exactly as in the web app.

Simulated webcam: the test faces are degraded in ways a real camera does —
dim light, motion blur, low resolution, sensor noise and a jittery face
detector — with a fixed random seed so every model sees identical images.

Usage:
  python scripts/benchmark.py                                  # all model versions
  python scripts/benchmark.py --extra _teachers/T1 _teachers/T2

Outputs:
  models/<version>/test_report_v1.txt      per-dataset accuracy, F1 and per-emotion recall
  models/benchmark.txt                     comparison table
  assets/v1/benchmark.png                  FER+ / RAF-DB / webcam accuracy per model
  assets/v1/webcam_conditions.png          accuracy under each webcam condition
"""
import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.model import get_device  # noqa: E402
from src.predictor import available_models  # noqa: E402

_spec = importlib.util.spec_from_file_location("train_v1", ROOT / "scripts" / "train_v1.py")
tv1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tv1)

DATA = ROOT / "data" / "processed" / "v1.npz"
ASSETS = ROOT / "assets" / "v1"
TEST = 2
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
TEXT, TEXT_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


# ---------------------------------------------------------------- webcam conditions (deterministic)

def dim_light(x, g):
    return ((x - 0.5) * 0.6 + 0.4).clamp(1e-4, 1) ** 1.8


def motion_blur(x, g):
    n, _, h, w = x.shape
    k = max(3, round(7 * h / 64) | 1)  # same blur relative to the face size at 48 and 64 px
    r = torch.arange(k) - k // 2
    ys, xs = torch.meshgrid(r.float(), r.float(), indexing="ij")
    angle = torch.rand(n, 1, 1, generator=g) * math.pi
    across = -xs * torch.sin(angle) + ys * torch.cos(angle)
    kern = (across.abs() <= 0.5).float()
    kern = (kern / kern.sum(dim=(1, 2), keepdim=True)).unsqueeze(1)
    padded = F.pad(x.view(1, n, h, w), (k // 2,) * 4, mode="replicate")
    return F.conv2d(padded, kern, groups=n).view(n, 1, h, w)


def low_resolution(x, g):
    h = x.shape[-1]
    small = F.interpolate(x, size=(round(h * 0.4),) * 2, mode="bilinear", align_corners=False, antialias=True)
    return F.interpolate(small, size=(h, h), mode="bilinear", align_corners=False)


def sensor_noise(x, g):
    return (x + torch.randn(x.shape, generator=g) * 0.05).clamp(0, 1)


def detector_jitter(x, g):
    n = x.shape[0]
    angle = (torch.rand(n, generator=g) * 2 - 1) * math.radians(8)
    scale = 1 + (torch.rand(n, generator=g) * 2 - 1) * 0.08
    tx, ty = (torch.rand(2, n, generator=g) * 2 - 1) * 0.05 * 2
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    return F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False),
                         padding_mode="border", align_corners=False)


CONDITIONS = {
    "clean": [],
    "dim light": [dim_light],
    "motion blur": [motion_blur],
    "low resolution": [low_resolution],
    "sensor noise": [sensor_noise],
    "detector jitter": [detector_jitter],
    "all combined": [detector_jitter, dim_light, motion_blur, low_resolution, sensor_noise],
}


# ---------------------------------------------------------------- evaluation

@torch.no_grad()
def logits_for(model, config, images, steps, device, batch_size=512):
    aligned = config.get("input") == "aligned64"
    mean, std = (config["mean"], config["std"]) if aligned else (tv1.FER_MEAN, tv1.FER_STD)
    g = torch.Generator().manual_seed(0)  # identical degradations for every model
    out = []
    for start in range(0, len(images), batch_size):
        x = torch.from_numpy(images[start:start + batch_size]).float().div(255).unsqueeze(1)
        for step in steps:
            x = step(x, g)
        out.append(model(((x - mean) / std).to(device)).cpu())
    return torch.cat(out)


def macro_f1(y, pred, n=7):
    f1 = []
    for c in range(n):
        tp = np.sum((pred == c) & (y == c))
        p = tp / max(1, np.sum(pred == c))
        r = tp / max(1, np.sum(y == c))
        f1.append(0 if p + r == 0 else 2 * p * r / (p + r))
    return float(np.mean(f1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra", nargs="*", default=[], help="more models, e.g. _teachers/T1")
    args = parser.parse_args()

    data = dict(np.load(DATA))
    emotions = [str(e) for e in data["emotions"]]
    test = np.flatnonzero(data["split"] == TEST)
    y, src = data["hard"][test], data["source"][test]
    device = get_device()
    names = available_models() + args.extra

    results = {}
    for name in names:
        model, config = tv1.load_model(name, device)
        aligned = config.get("input") == "aligned64"
        images = (data["X"] if aligned else data["X48"])[test]
        model = model.to(device)
        r = {}
        for cond, steps in CONDITIONS.items():
            pred = logits_for(model, config, images, steps, device).argmax(1).numpy()
            r[cond] = float((pred == y).mean())
            if cond == "clean":
                clean_pred = pred
        fer, raf = src == 0, src == 1
        r["fer"] = float((clean_pred[fer] == y[fer]).mean())
        r["raf"] = float((clean_pred[raf] == y[raf]).mean())
        r["fer_f1"], r["raf_f1"] = macro_f1(y[fer], clean_pred[fer]), macro_f1(y[raf], clean_pred[raf])
        r["recall"] = {e: float((clean_pred[y == i] == i).mean()) for i, e in enumerate(emotions)}
        results[name] = r

        report = [f"Model: {name} | no TTA | each model on its own framing ({'aligned 64x64' if aligned else 'FER 48x48'})",
                  f"FER+ test   ({fer.sum():,} faces): accuracy {r['fer']:.1%}, macro F1 {r['fer_f1']:.1%}",
                  f"RAF-DB test ({raf.sum():,} faces): accuracy {r['raf']:.1%}, macro F1 {r['raf_f1']:.1%}",
                  "", "Simulated webcam (FER+ + RAF-DB test):"]
        report += [f"  {c:<16} {r[c]:.1%}" for c in CONDITIONS]
        report += ["", "Recall per emotion (both test sets):"]
        report += [f"  {e:<9} {v:.1%}" for e, v in r["recall"].items()]
        if not name.startswith("_"):
            (ROOT / "models" / name / "test_report_v1.txt").write_text("\n".join(report) + "\n")
        print(f"{name:<14} FER+ {r['fer']:.1%}  RAF-DB {r['raf']:.1%}  webcam(all) {r['all combined']:.1%}")

    # comparison table
    cols = ["fer", "raf"] + list(CONDITIONS)
    header = f"{'model':<14}" + "".join(f"{c[:12]:>13}" for c in ["FER+ test", "RAF-DB test"] + list(CONDITIONS))
    lines = [header] + [f"{n:<14}" + "".join(f"{results[n][c]:>13.1%}" for c in cols) for n in names]
    (ROOT / "models" / "benchmark.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))

    shown = [n for n in names if not n.startswith("_")]
    plot_groups(results, shown, [("FER+ test", "fer"), ("RAF-DB test", "raf"), ("webcam, all effects", "all combined")],
                "Test accuracy per model", ASSETS / "benchmark.png")
    plot_groups(results, shown, [(c, c) for c in CONDITIONS], "Simulated webcam: accuracy per condition",
                ASSETS / "webcam_conditions.png")
    print(f"\nSaved models/benchmark.txt, {ASSETS.relative_to(ROOT)}/benchmark.png, webcam_conditions.png")


def plot_groups(results, models, groups, title, path):
    ASSETS.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(groups))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(groups) + 3), 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for i, m in enumerate(models):
        vals = [results[m][key] for _, key in groups]
        bars = ax.bar(x - 0.4 + width * (i + 0.5), vals, width - 0.02, color=SERIES[i % len(SERIES)], label=m)
        for b in bars:
            ax.annotate(f"{b.get_height():.0%}", (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8, color=TEXT_MUTED)
    lo = min(results[m][key] for m in models for _, key in groups)
    ax.set_ylim(max(0, lo - 0.1), 1.0)
    ax.set_xticks(x, [g for g, _ in groups])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.legend(frameon=False, ncols=len(models), loc="upper right")
    ax.set_title(title, loc="left", fontsize=14, color=TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
