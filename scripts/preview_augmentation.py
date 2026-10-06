"""Step 8: Show what data augmentation does to a few training faces."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.augment import augment  # noqa: E402

DATA = ROOT / "data" / "processed" / "fer2013.npz"
TEXT = "#0b0b0b"
SURFACE = "#fcfcfb"


def main(rows=4, versions=7):
    torch.manual_seed(0)
    data = np.load(DATA)
    mean, std = float(data["mean"]), float(data["std"])
    faces = torch.from_numpy(data["X_train"][[3, 11, 25, 40][:rows]]).unsqueeze(1)

    fig, axes = plt.subplots(rows, versions + 1, figsize=((versions + 1) * 1.3, rows * 1.45), facecolor=SURFACE)
    for r in range(rows):
        batch = faces[r:r + 1].repeat(versions, 1, 1, 1)
        variants = [faces[r]] + list(augment(batch))
        for c, img in enumerate(variants):
            ax = axes[r, c]
            ax.imshow(img[0].numpy() * std + mean, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if r == 0:
                ax.set_title("original" if c == 0 else f"aug #{c}", fontsize=9, color=TEXT)
    fig.suptitle("Data augmentation: flip, rotate ±10°, shift ±10%, zoom ±10%, random erase",
                 x=0.01, ha="left", fontsize=12, color=TEXT)
    fig.tight_layout()
    out = ROOT / "assets" / "augmentation.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
