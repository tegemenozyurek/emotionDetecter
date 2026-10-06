"""Step 4: Preprocess FER-2013 into ready-to-train numpy arrays.

What happens here:
  1. Load every image as a 48x48 grayscale array.
  2. Hold out 10% of the training set as a *validation* set (stratified, so
     every emotion keeps its share). The test set stays untouched until step 7.
  3. Scale pixels from 0-255 to 0-1, then standardize with the TRAIN mean/std
     (never compute statistics on val/test — that would leak information).
  4. Compute class weights so rare emotions (disgust!) count more in the loss.

Outputs:
  data/processed/fer2013.npz       — X_train, y_train, X_val, y_val, X_test, y_test, ...
  assets/preprocessing.png         — pixel distribution before vs after standardizing
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "fer2013"
OUT = ROOT / "data" / "processed" / "fer2013.npz"
ASSETS = ROOT / "assets"
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
VAL_FRACTION = 0.10
SEED = 42

BAR_COLOR = "#2a78d6"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def load_split(split):
    images, labels = [], []
    for label, emotion in enumerate(EMOTIONS):
        for path in sorted((DATA / split / emotion).glob("*.jpg")):
            images.append(np.asarray(Image.open(path).convert("L"), dtype=np.uint8))
            labels.append(label)
    return np.stack(images), np.array(labels, dtype=np.int64)


def stratified_split(y, fraction, seed):
    """Return (keep_idx, holdout_idx) so each class loses the same fraction."""
    rng = np.random.default_rng(seed)
    keep, holdout = [], []
    for label in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == label))
        n_hold = round(len(idx) * fraction)
        holdout.append(idx[:n_hold])
        keep.append(idx[n_hold:])
    return rng.permutation(np.concatenate(keep)), rng.permutation(np.concatenate(holdout))


def plot_distribution(raw, standardized):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), facecolor=SURFACE)
    panels = [
        (raw.ravel(), "Raw pixels (0–255)"),
        (standardized.ravel(), "After standardizing (mean 0, std 1)"),
    ]
    for ax, (values, title) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.hist(values, bins=64, color=BAR_COLOR, edgecolor=SURFACE, linewidth=0.5)
        ax.set_title(title, loc="left", fontsize=12, color=TEXT)
        ax.set_yticks([])
        ax.tick_params(axis="x", colors=TEXT_MUTED, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    fig.suptitle("Pixel value distribution (training set)", x=0.01, ha="left",
                 fontsize=14, color=TEXT)
    fig.tight_layout()
    fig.savefig(ASSETS / "preprocessing.png", dpi=150)
    plt.close(fig)


def main():
    print("Loading images...")
    X_full, y_full = load_split("train")
    X_test, y_test = load_split("test")
    print(f"  train (full): {X_full.shape}   test: {X_test.shape}")

    train_idx, val_idx = stratified_split(y_full, VAL_FRACTION, SEED)
    X_train, y_train = X_full[train_idx], y_full[train_idx]
    X_val, y_val = X_full[val_idx], y_full[val_idx]

    # Scale to 0-1, then standardize using TRAIN statistics only.
    to_float = lambda x: x.astype(np.float32) / 255.0
    X_train_f, X_val_f, X_test_f = map(to_float, (X_train, X_val, X_test))
    mean, std = X_train_f.mean(), X_train_f.std()
    X_train_f = (X_train_f - mean) / std
    X_val_f = (X_val_f - mean) / std
    X_test_f = (X_test_f - mean) / std

    # Inverse-frequency class weights: rare classes get larger weights.
    counts = np.bincount(y_train, minlength=len(EMOTIONS))
    class_weights = (counts.sum() / (len(EMOTIONS) * counts)).astype(np.float32)

    print(f"\nSplit sizes:  train {len(y_train):,} | val {len(y_val):,} | test {len(y_test):,}")
    print(f"\nNormalization (from train): mean={mean:.4f}  std={std:.4f}")
    print(f"  train after: mean={X_train_f.mean():+.3f}  std={X_train_f.std():.3f}")
    print(f"  val   after: mean={X_val_f.mean():+.3f}  std={X_val_f.std():.3f}")

    print(f"\n{'emotion':<9} {'train':>6} {'val':>5} {'test':>5}   weight")
    for i, e in enumerate(EMOTIONS):
        print(f"{e:<9} {counts[i]:>6} {np.sum(y_val == i):>5} {np.sum(y_test == i):>5}   {class_weights[i]:.2f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        X_train=X_train_f, y_train=y_train,
        X_val=X_val_f, y_val=y_val,
        X_test=X_test_f, y_test=y_test,
        mean=np.float32(mean), std=np.float32(std),
        class_weights=class_weights,
        emotions=np.array(EMOTIONS),
    )
    print(f"\nSaved {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.0f} MB)")

    ASSETS.mkdir(exist_ok=True)
    plot_distribution(X_train, X_train_f)
    print(f"Saved {(ASSETS / 'preprocessing.png').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
