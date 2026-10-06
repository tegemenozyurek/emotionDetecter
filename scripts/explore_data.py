"""Step 3: Explore FER-2013 — class balance, sample faces, and the "average face".

Outputs (saved to assets/):
  class_distribution.png  — how many images per emotion (train vs test)
  sample_faces.png        — random examples of each emotion
  average_faces.png       — pixel-wise mean face of each emotion
"""
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "fer2013"
ASSETS = ROOT / "assets"
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Chart styling
TRAIN_COLOR = "#2a78d6"
TEST_COLOR = "#eb6834"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "text.color": TEXT,
    "axes.labelcolor": TEXT_MUTED,
    "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED,
    "axes.edgecolor": GRID,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
})


def image_paths(split, emotion):
    return sorted((DATA / split / emotion).glob("*.jpg"))


def load_gray(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def plot_class_distribution():
    train = [len(image_paths("train", e)) for e in EMOTIONS]
    test = [len(image_paths("test", e)) for e in EMOTIONS]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(EMOTIONS))
    width = 0.38
    bars_train = ax.bar(x - width / 2 - 0.01, train, width, color=TRAIN_COLOR, label="train")
    bars_test = ax.bar(x + width / 2 + 0.01, test, width, color=TEST_COLOR, label="test")

    for bars in (bars_train, bars_test):
        for bar in bars:
            ax.annotate(f"{int(bar.get_height()):,}",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, color=TEXT_MUTED)

    ax.set_xticks(x, EMOTIONS)
    ax.set_ylabel("images")
    ax.set_title("FER-2013: images per emotion", loc="left", fontsize=14, color=TEXT)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="both", length=0)
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(ASSETS / "class_distribution.png", dpi=150)
    plt.close(fig)

    print("Images per emotion (train / test):")
    for e, tr, te in zip(EMOTIONS, train, test):
        print(f"  {e:<9} {tr:>5} / {te:>4}   {tr / sum(train):6.1%} of train")
    ratio = max(train) / min(train)
    print(f"  -> largest class is {ratio:.1f}x the smallest (imbalanced!)")


def plot_sample_faces(per_class=8, seed=42):
    rng = random.Random(seed)
    fig, axes = plt.subplots(len(EMOTIONS), per_class,
                             figsize=(per_class * 1.2, len(EMOTIONS) * 1.3))
    for row, emotion in enumerate(EMOTIONS):
        samples = rng.sample(image_paths("train", emotion), per_class)
        for col, path in enumerate(samples):
            ax = axes[row, col]
            ax.imshow(load_gray(path), cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[row, 0].set_ylabel(emotion, rotation=0, ha="right", va="center",
                                fontsize=12, color=TEXT, labelpad=10)
    fig.suptitle("Random training samples (48×48 grayscale)", x=0.02, ha="left",
                 fontsize=14, color=TEXT)
    fig.tight_layout()
    fig.savefig(ASSETS / "sample_faces.png", dpi=150)
    plt.close(fig)


def plot_average_faces():
    fig, axes = plt.subplots(1, len(EMOTIONS), figsize=(len(EMOTIONS) * 1.6, 2.2))
    for ax, emotion in zip(axes, EMOTIONS):
        stack = np.stack([load_gray(p) for p in image_paths("train", emotion)])
        ax.imshow(stack.mean(axis=0), cmap="gray")
        ax.set_title(emotion, fontsize=11, color=TEXT)
        ax.axis("off")
    fig.suptitle("Average face per emotion", x=0.02, ha="left", fontsize=14, color=TEXT)
    fig.tight_layout()
    fig.savefig(ASSETS / "average_faces.png", dpi=150)
    plt.close(fig)


def main():
    ASSETS.mkdir(exist_ok=True)

    sample = load_gray(image_paths("train", "happy")[0])
    print(f"One image: shape={sample.shape}, pixel range={sample.min():.0f}-{sample.max():.0f}\n")

    plot_class_distribution()
    plot_sample_faces()
    plot_average_faces()
    print(f"\nSaved charts to {ASSETS.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
