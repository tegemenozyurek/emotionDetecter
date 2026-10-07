"""Step 10: Build the README header image — one test face per emotion with the
model's probability for every emotion underneath.

Usage:
  python scripts/make_showcase.py              # newest model
  python scripts/make_showcase.py eD0.1
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.model import EmotionCNN, get_device  # noqa: E402
from src.predictor import available_models  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
TOP_COLOR = "#2a78d6"
OTHER_COLOR = "#c9d9ef"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
SURFACE = "#fcfcfb"


def main():
    version = sys.argv[1] if len(sys.argv) > 1 else available_models()[-1]
    config = json.loads((ROOT / "models" / version / "config.json").read_text())
    data_name = config.get("data", "fer2013")  # show faces with the labels the model learned
    data = np.load(PROCESSED / f"{data_name}.npz")
    emotions = [str(e) for e in data["emotions"]]
    X, y = data["X_test"], data["y_test"]
    mean, std = float(data["mean"]), float(data["std"])

    device = get_device()
    model = EmotionCNN().to(device)
    model.load_state_dict(torch.load(ROOT / "models" / version / "model.pt", map_location=device))
    model.eval()
    with torch.no_grad():
        probs = np.concatenate([
            torch.softmax(model(torch.from_numpy(X[i:i + 512]).unsqueeze(1).to(device)), 1).cpu().numpy()
            for i in range(0, len(X), 512)
        ])

    # For each emotion pick a random face the model gets right with 70-97% confidence:
    # a clear example, but one that still shows the runner-up emotions.
    rng = np.random.default_rng(3)
    picks = []
    for label in range(len(emotions)):
        ok = np.flatnonzero((y == label) & (probs.argmax(1) == label)
                            & (probs.max(1) > 0.70) & (probs.max(1) < 0.97))
        picks.append(rng.choice(ok))

    n = len(emotions)
    fig = plt.figure(figsize=(n * 1.7 + 0.8, 4.2), facecolor=SURFACE)
    grid = fig.add_gridspec(2, n, height_ratios=[1.0, 0.85], hspace=0.12, wspace=0.12)
    for col, idx in enumerate(picks):
        ax_img = fig.add_subplot(grid[0, col])
        ax_img.imshow(X[idx] * std + mean, cmap="gray", vmin=0, vmax=1)
        ax_img.set_title(emotions[y[idx]], fontsize=12, color=TEXT, pad=6)
        ax_img.axis("off")

        ax_bar = fig.add_subplot(grid[1, col])
        p = probs[idx]
        colors = [TOP_COLOR if i == p.argmax() else OTHER_COLOR for i in range(n)]
        ax_bar.barh(range(n), p, color=colors, height=0.72)
        # Emotion names only on the first column; the other columns share the same order.
        ax_bar.set_yticks(range(n), emotions if col == 0 else [""] * n, fontsize=8)
        ax_bar.invert_yaxis()
        ax_bar.set_xlim(0, 1)
        ax_bar.set_xticks([])
        ax_bar.tick_params(axis="y", length=0, colors=TEXT_MUTED, pad=2)
        for spine in ax_bar.spines.values():
            spine.set_visible(False)
        ax_bar.text(p.max() + 0.03 if p.max() < 0.75 else p.max() - 0.03, p.argmax(), f"{p.max():.0%}",
                    va="center", ha="left" if p.max() < 0.75 else "right", fontsize=8,
                    color=TEXT if p.max() < 0.75 else "white")
        ax_bar.set_facecolor(SURFACE)

    labels = "FER+" if data_name == "ferplus" else "FER-2013"
    fig.suptitle(f"Test-set faces the model has never seen, with its predicted probabilities ({version}, {labels} labels)",
                 x=0.01, ha="left", fontsize=12, color=TEXT_MUTED)
    out = ROOT / "assets" / "showcase.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"Saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
