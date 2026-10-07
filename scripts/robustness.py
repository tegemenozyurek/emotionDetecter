"""Step 8: Compare model versions on clean vs. perturbed test faces.

Real photos are rarely as neatly aligned as FER-2013: heads are tilted, faces
are off-centre, hands cover part of the face. Here we distort the TEST set in
controlled ways and measure how much each model's accuracy drops.

Usage:
  python scripts/robustness.py                         # all versions, original labels
  python scripts/robustness.py eD0.1 eD0.5
  python scripts/robustness.py --data ferplus          # FER+ test labels

Outputs:
  assets/robustness[_ferplus].png   — accuracy per condition, one bar per model
  models/robustness[_ferplus].txt   — the printed table
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.augment import augment  # noqa: E402
from src.model import EmotionCNN, get_device  # noqa: E402
from src.predictor import available_models  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
NO_CHANGE = dict(max_rotate=0, max_shift=0, scale_range=(1, 1), erase_prob=0)
CONDITIONS = {
    "clean": None,
    "rotated ±15°": {**NO_CHANGE, "max_rotate": 15},
    "shifted ±10%": {**NO_CHANGE, "max_shift": 0.1},
    "zoomed ±15%": {**NO_CHANGE, "scale_range": (0.85, 1.15)},
    "partly covered": {**NO_CHANGE, "erase_prob": 1.0, "erase_size": 14},
    "all combined": dict(max_rotate=15, max_shift=0.1, scale_range=(0.85, 1.15), erase_prob=0.5),
}

SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


@torch.no_grad()
def accuracy(model, X, y, device, distortion, batch_size=512):
    torch.manual_seed(0)  # identical distortions for every model
    correct = 0
    for start in range(0, len(X), batch_size):
        xb = X[start:start + batch_size].to(device)
        if distortion:
            xb = augment(xb, **distortion)
        correct += (model(xb).argmax(1).cpu() == y[start:start + batch_size]).sum().item()
    return correct / len(X)


def plot(results, versions, out_path, labels):
    conditions = list(CONDITIONS)
    x = np.arange(len(conditions))
    width = 0.8 / len(versions)
    fig, ax = plt.subplots(figsize=(11, 4.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for i, v in enumerate(versions):
        offsets = x - 0.4 + width * (i + 0.5)
        bars = ax.bar(offsets, [results[v][c] for c in conditions], width - 0.02,
                      color=SERIES_COLORS[i], label=v)
        for bar in bars:
            ax.annotate(f"{bar.get_height():.0%}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, color=TEXT_MUTED)
    ax.set_xticks(x, conditions)
    ax.set_ylim(0.4, max(0.72, max(max(r.values()) for r in results.values()) + 0.05))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.legend(frameon=False, loc="upper right", ncols=len(versions))
    ax.set_title(f"Test accuracy on clean vs. distorted faces ({labels} labels)", loc="left", fontsize=14, color=TEXT)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = sys.argv[1:]
    data_name = "fer2013"
    if "--data" in args:
        i = args.index("--data")
        data_name = args[i + 1]
        del args[i:i + 2]
    versions = args or available_models()
    sfx = "" if data_name == "fer2013" else f"_{data_name}"
    data = np.load(PROCESSED / f"{data_name}.npz")
    X = torch.from_numpy(data["X_test"]).unsqueeze(1)
    y = torch.from_numpy(data["y_test"])
    device = get_device()

    results = {}
    for v in versions:
        model = EmotionCNN().to(device)
        model.load_state_dict(torch.load(ROOT / "models" / v / "model.pt", map_location=device))
        model.eval()
        results[v] = {name: accuracy(model, X, y, device, d) for name, d in CONDITIONS.items()}

    lines = [f"{'test condition':<16}" + "".join(f"{v:>9}" for v in versions)]
    for name in CONDITIONS:
        lines.append(f"{name:<16}" + "".join(f"{results[v][name]:>9.1%}" for v in versions))
    table = "\n".join(lines)
    print(table)
    (ROOT / "models" / f"robustness{sfx}.txt").write_text(table + "\n")
    plot(results, versions, ROOT / "assets" / f"robustness{sfx}.png",
         "FER+" if data_name == "ferplus" else "original FER-2013")
    print(f"\nSaved assets/robustness{sfx}.png, models/robustness{sfx}.txt")


if __name__ == "__main__":
    main()
