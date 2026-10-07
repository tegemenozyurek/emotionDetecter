"""Relabel our FER-2013 images with the cleaner FER+ labels (Microsoft).

FER+ had every FER-2013 image re-labeled by 10 people. Its label file follows
the row order of the original fer2013.csv, while our images (Kaggle folder
version) have random file names — so we match every image to its csv row by
pixel similarity, check that the old labels agree, then take the FER+ vote.

Inputs:
  data/fer2013/                     our images (scripts/download_data.py)
  data/ferplus/fer2013.csv          original FER-2013 pixels + labels
  data/ferplus/fer2013new.csv       FER+ votes (github.com/microsoft/FERPlus)

Label rule (the "majority" mode of the FER+ paper):
  1. ignore emotions that got only 1 vote (one tagger's outlier)
  2. keep the image only if the top emotion has more than half of the votes
  3. drop it if the winner is contempt (not in our 7 classes), unknown, or not-a-face

Outputs:
  data/processed/ferplus.npz              same layout as fer2013.npz, with FER+ labels
  assets/ferplus/label_changes.png        old label -> new label
  assets/ferplus/relabeled_examples.png   faces whose label changed, with the votes
"""
import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "data" / "fer2013"
FERPLUS = ROOT / "data" / "ferplus"
OUT = ROOT / "data" / "processed" / "ferplus.npz"
ASSETS = ROOT / "assets" / "ferplus"

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
FER_CODES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]  # fer2013.csv label ids
VOTE_COLUMNS = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt", "unknown", "NF"]
VOTE_TO_EMOTION = {"neutral": "neutral", "happiness": "happy", "surprise": "surprise", "sadness": "sad",
                   "anger": "angry", "disgust": "disgust", "fear": "fear"}
MEAN, STD = 0.50785, 0.25500  # same normalization as fer2013.npz, so the app needs no change
VAL_FRACTION, SEED = 0.10, 42

TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
SURFACE = "#fcfcfb"
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#f3f7fd", "#2a78d6", "#0d2f5c"])


def load_csv_rows():
    pixels, old_labels, usage = [], [], []
    with open(FERPLUS / "fer2013.csv") as f:
        for row in csv.DictReader(f):
            pixels.append(np.array(row["pixels"].split(), dtype=np.uint8))
            old_labels.append(EMOTIONS.index(FER_CODES[int(row["emotion"])]))
            usage.append(row["Usage"])
    with open(FERPLUS / "fer2013new.csv") as f:
        votes = np.array([[int(row[c]) for c in VOTE_COLUMNS] for row in csv.DictReader(f)])
    return np.stack(pixels).astype(np.float32), np.array(old_labels), np.array(usage), votes


def load_our_images(split, prefix):
    images, labels, names = [], [], []
    for label, emotion in enumerate(EMOTIONS):
        for path in sorted((IMAGES / split / emotion).glob(f"{prefix}_*.jpg")):
            images.append(np.asarray(Image.open(path).convert("L"), dtype=np.uint8).ravel())
            labels.append(label)
            names.append(path.name)
    return np.stack(images), np.array(labels), names


def nearest_rows(ours, theirs, device, chunk=2048):
    """For every image in `ours`, index of the closest csv row (mean abs pixel diff)."""
    a = torch.from_numpy(ours.astype(np.float32)).to(device)
    b = torch.from_numpy(theirs).to(device)
    b_sq = (b ** 2).sum(1)
    idx, dist = [], []
    for start in range(0, len(a), chunk):
        x = a[start:start + chunk]
        d = (x ** 2).sum(1, keepdim=True) - 2 * x @ b.T + b_sq  # squared L2, all pairs
        best = d.argmin(1)
        idx.append(best.cpu())
        dist.append((x - b[best]).abs().mean(1).cpu())
    return torch.cat(idx).numpy(), torch.cat(dist).numpy()


def ferplus_label(v):
    """FER+ majority vote -> (label index or None, reason)."""
    v = np.where(v <= 1, 0, v)  # drop single-vote outliers
    if v.sum() == 0:
        return None, "no agreement"
    top = int(v.argmax())
    if v[top] <= v.sum() / 2:
        return None, "no majority"
    name = VOTE_COLUMNS[top]
    if name not in VOTE_TO_EMOTION:
        return None, name
    return EMOTIONS.index(VOTE_TO_EMOTION[name]), "ok"


def stratified_split(y, fraction, seed):
    rng = np.random.default_rng(seed)
    keep, hold = [], []
    for label in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == label))
        n = round(len(idx) * fraction)
        hold.append(idx[:n])
        keep.append(idx[n:])
    return rng.permutation(np.concatenate(keep)), rng.permutation(np.concatenate(hold))


def plot_label_changes(old, new):
    cm = np.zeros((7, 7), dtype=np.int64)
    np.add.at(cm, (old, new), 1)
    norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7.5, 6.5), facecolor=SURFACE)
    im = ax.imshow(norm, cmap=SEQUENTIAL, vmin=0, vmax=1)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{norm[i, j]:.0%}", ha="center", va="center", fontsize=10,
                    color="white" if norm[i, j] > 0.45 else TEXT)
    ax.set_xticks(range(7), EMOTIONS, rotation=35, ha="right")
    ax.set_yticks(range(7), EMOTIONS)
    ax.set_xlabel("FER+ label (10 taggers)", color=TEXT_MUTED)
    ax.set_ylabel("original FER-2013 label", color=TEXT_MUTED)
    ax.tick_params(colors=TEXT_MUTED, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Where the original labels move under FER+ (row-normalized)", loc="left", fontsize=12, color=TEXT)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format=plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=TEXT_MUTED, length=0)
    fig.tight_layout()
    fig.savefig(ASSETS / "label_changes.png", dpi=150)
    plt.close(fig)


def plot_examples(images, old, new, votes, n=16, seed=1):
    changed = np.flatnonzero(old != new)
    rng = np.random.default_rng(seed)
    picks = rng.choice(changed, n, replace=False)
    cols = 8
    fig, axes = plt.subplots(n // cols, cols, figsize=(cols * 1.55, n // cols * 2.35), facecolor=SURFACE)
    for ax, i in zip(axes.ravel(), picks):
        ax.imshow(images[i].reshape(48, 48), cmap="gray", vmin=0, vmax=255)
        winner_votes = votes[i][VOTE_COLUMNS.index(
            next(k for k, e in VOTE_TO_EMOTION.items() if e == EMOTIONS[new[i]]))]
        ax.set_title(f"was: {EMOTIONS[old[i]]}\nFER+: {EMOTIONS[new[i]]} ({winner_votes}/10)",
                     fontsize=8.5, color=TEXT)
        ax.axis("off")
    fig.suptitle("Images whose label changed (original → FER+, with the winning vote count)",
                 x=0.01, ha="left", fontsize=12, color=TEXT)
    fig.tight_layout()
    fig.savefig(ASSETS / "relabeled_examples.png", dpi=150)
    plt.close(fig)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Loading fer2013.csv + fer2013new.csv ...")
    csv_pixels, csv_old, csv_usage, votes = load_csv_rows()

    groups = [("train", "Training"), ("test", "PublicTest"), ("test", "PrivateTest")]
    matched = {}
    print("\nMatching our images to csv rows:")
    for split, usage in groups:
        imgs, folder_labels, _ = load_our_images(split, usage)
        rows = np.flatnonzero(csv_usage == usage)
        best, dist = nearest_rows(imgs, csv_pixels[rows], device)
        csv_idx = rows[best]
        agree = (csv_old[csv_idx] == folder_labels).mean()
        unique = len(np.unique(csv_idx)) / len(csv_idx)
        print(f"  {usage:<11} {len(imgs):>6} images | mean pixel diff {dist.mean():.2f} (max {dist.max():.2f}) | "
              f"old labels agree {agree:.2%} | unique rows {unique:.2%}")
        matched[usage] = (imgs, folder_labels, csv_idx, dist)

    # Sanity check: matching must be near-exact (JPEG noise only) and labels must agree.
    all_dist = np.concatenate([m[3] for m in matched.values()])
    all_agree = np.concatenate([csv_old[m[2]] == m[1] for m in matched.values()])
    assert all_dist.max() < 10 and all_agree.mean() > 0.99, "matching failed — do not trust these labels"

    def relabel(usage_names):
        imgs = np.concatenate([matched[u][0] for u in usage_names])
        old = np.concatenate([matched[u][1] for u in usage_names])
        rows = np.concatenate([matched[u][2] for u in usage_names])
        results = [ferplus_label(votes[r]) for r in rows]
        reasons = Counter(reason for _, reason in results)
        keep = np.array([lab is not None for lab, _ in results])
        new = np.array([lab if lab is not None else -1 for lab, _ in results])
        return imgs[keep], old[keep], new[keep], votes[rows][keep], reasons, len(rows)

    print("\nApplying FER+ majority vote:")
    out = {}
    for name, usages in [("train", ["Training"]), ("test", ["PublicTest", "PrivateTest"])]:
        imgs, old, new, v, reasons, total = relabel(usages)
        dropped = {k: c for k, c in reasons.items() if k != "ok"}
        print(f"  {name:<5} kept {len(new):,} / {total:,} | dropped: "
              + ", ".join(f"{k} {c}" for k, c in sorted(dropped.items(), key=lambda kv: -kv[1])))
        print(f"        label changed for {(old != new).mean():.1%} of kept images")
        out[name] = (imgs, old, new, v)

    ASSETS.mkdir(parents=True, exist_ok=True)
    tr_imgs, tr_old, tr_new, tr_votes = out["train"]
    te_imgs, te_old, te_new, te_votes = out["test"]
    plot_label_changes(np.concatenate([tr_old, te_old]), np.concatenate([tr_new, te_new]))
    plot_examples(tr_imgs, tr_old, tr_new, tr_votes)

    # Same pipeline as scripts/preprocess.py: stratified val split, fixed normalization, class weights.
    train_idx, val_idx = stratified_split(tr_new, VAL_FRACTION, SEED)
    norm = lambda a: ((a.reshape(-1, 48, 48).astype(np.float32) / 255.0) - MEAN) / STD
    counts = np.bincount(tr_new[train_idx], minlength=7)
    class_weights = (counts.sum() / (7 * counts)).astype(np.float32)

    print(f"\n{'emotion':<9} {'old train':>9} {'FER+ train':>10} {'FER+ test':>9}   weight")
    old_counts = np.bincount(tr_old, minlength=7)
    for i, e in enumerate(EMOTIONS):
        print(f"{e:<9} {old_counts[i]:>9} {np.sum(tr_new == i):>10} {np.sum(te_new == i):>9}   {class_weights[i]:.2f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        X_train=norm(tr_imgs[train_idx]), y_train=tr_new[train_idx],
        X_val=norm(tr_imgs[val_idx]), y_val=tr_new[val_idx],
        X_test=norm(te_imgs), y_test=te_new, y_test_old=te_old,
        mean=np.float32(MEAN), std=np.float32(STD),
        class_weights=class_weights, emotions=np.array(EMOTIONS),
    )
    print(f"\nSaved {OUT.relative_to(ROOT)}: train {len(train_idx):,} / val {len(val_idx):,} / test {len(te_new):,}")
    print(f"Saved assets/ferplus/label_changes.png, assets/ferplus/relabeled_examples.png")


if __name__ == "__main__":
    main()
