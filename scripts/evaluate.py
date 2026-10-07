"""Step 7: Evaluate the trained model on the held-out TEST set (first time we touch it).

Usage:
  python scripts/evaluate.py --model eD0.5
  python scripts/evaluate.py --model eD0.7 --data ferplus   # outputs get a _ferplus suffix

Prints overall accuracy plus precision / recall / F1 for each emotion, and saves:
  assets/<model>/confusion_matrix.png   — which emotions get confused with which
  assets/<model>/predictions.png        — random test faces with true vs predicted label
  models/<model>/test_report.txt        — the printed report

Test-time augmentation (averaging with the mirrored face) is on by default,
the same as in the app; pass --no-tta to turn it off.
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.model import EmotionCNN, get_device  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
ASSETS = ROOT / "assets"

TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
SURFACE = "#fcfcfb"
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#f3f7fd", "#2a78d6", "#0d2f5c"])


@torch.no_grad()
def predict(model, X, device, batch_size=512, tta=False):
    """Class probabilities. With tta, also look at the mirrored face and average."""
    probs = []
    for start in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[start:start + batch_size]).unsqueeze(1).to(device)
        p = torch.softmax(model(xb), dim=1)
        if tta:
            p = (p + torch.softmax(model(torch.flip(xb, dims=[3])), dim=1)) / 2
        probs.append(p.cpu().numpy())
    return np.concatenate(probs)


def confusion_matrix(y_true, y_pred, n):
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def build_report(cm, emotions):
    tp = np.diag(cm)
    precision = tp / np.maximum(cm.sum(axis=0), 1)
    recall = tp / np.maximum(cm.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    support = cm.sum(axis=1)

    lines = [f"Test accuracy: {tp.sum() / cm.sum():.1%}  ({tp.sum():,} / {cm.sum():,} correct)", "",
             f"{'emotion':<9} {'precision':>9} {'recall':>7} {'f1':>6} {'support':>8}"]
    for i, e in enumerate(emotions):
        lines.append(f"{e:<9} {precision[i]:>9.1%} {recall[i]:>7.1%} {f1[i]:>6.1%} {support[i]:>8}")
    lines.append(f"{'macro avg':<9} {precision.mean():>9.1%} {recall.mean():>7.1%} {f1.mean():>6.1%} {support.sum():>8}")

    off = cm.copy()
    np.fill_diagonal(off, 0)
    lines += ["", "Most common mistakes (true -> predicted):"]
    for flat in np.argsort(off.ravel())[::-1][:5]:
        t, p = divmod(flat, len(emotions))
        lines.append(f"  {emotions[t]:<9} -> {emotions[p]:<9} {off[t, p]:>4}  ({off[t, p] / support[t]:.0%} of {emotions[t]})")
    return "\n".join(lines)


def plot_confusion(cm, emotions, out_path):
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7.5, 6.5), facecolor=SURFACE)
    im = ax.imshow(norm, cmap=SEQUENTIAL, vmin=0, vmax=1)
    for i in range(len(emotions)):
        for j in range(len(emotions)):
            ax.text(j, i, f"{norm[i, j]:.0%}", ha="center", va="center", fontsize=10,
                    color="white" if norm[i, j] > 0.45 else TEXT)
    ax.set_xticks(range(len(emotions)), emotions, rotation=35, ha="right")
    ax.set_yticks(range(len(emotions)), emotions)
    ax.set_xlabel("predicted", color=TEXT_MUTED)
    ax.set_ylabel("true", color=TEXT_MUTED)
    ax.tick_params(colors=TEXT_MUTED, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Confusion matrix (test set, row-normalized)", loc="left", fontsize=13, color=TEXT)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format=plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=TEXT_MUTED, length=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_predictions(X, y, probs, emotions, mean, std, out_path, n=24, seed=7):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), n, replace=False)
    cols = 8
    fig, axes = plt.subplots(n // cols, cols, figsize=(cols * 1.5, n // cols * 1.95), facecolor=SURFACE)
    for ax, i in zip(axes.ravel(), idx):
        pred = probs[i].argmax()
        ok = pred == y[i]
        ax.imshow(X[i] * std + mean, cmap="gray", vmin=0, vmax=1)  # undo standardization
        ax.set_title(f"{'✓' if ok else '✗'} {emotions[pred]} {probs[i, pred]:.0%}\ntrue: {emotions[y[i]]}",
                     fontsize=8.5, color=TEXT if ok else "#c0392b")
        ax.axis("off")
    fig.suptitle("Test predictions (✓ correct, ✗ wrong — predicted label, confidence, true label)",
                 x=0.01, ha="left", fontsize=12, color=TEXT)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="model version, e.g. eD0.5 -> models/eD0.5/model.pt")
    parser.add_argument("--data", default="fer2013", choices=["fer2013", "ferplus"],
                        help="test labels: original FER-2013 or FER+")
    parser.add_argument("--no-tta", dest="tta", action="store_false", help="disable test-time augmentation")
    args = parser.parse_args()
    model_dir = MODELS / args.model
    plot_dir = ASSETS / args.model
    plot_dir.mkdir(parents=True, exist_ok=True)
    sfx = "" if args.data == "fer2013" else f"_{args.data}"

    data = np.load(PROCESSED / f"{args.data}.npz")
    emotions = [str(e) for e in data["emotions"]]
    X_test, y_test = data["X_test"], data["y_test"]

    device = get_device()
    model = EmotionCNN().to(device)
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location=device))
    model.eval()

    probs = predict(model, X_test, device, tta=args.tta)
    cm = confusion_matrix(y_test, probs.argmax(1), len(emotions))

    report = f"Model: {args.model} | TTA: {args.tta} | labels: {args.data}\n" + build_report(cm, emotions)
    print(report)
    (model_dir / f"test_report{sfx}.txt").write_text(report + "\n")

    plot_confusion(cm, emotions, plot_dir / f"confusion_matrix{sfx}.png")
    plot_predictions(X_test, y_test, probs, emotions, float(data["mean"]), float(data["std"]),
                     plot_dir / f"predictions{sfx}.png")
    print(f"\nSaved {plot_dir.relative_to(ROOT)}/confusion_matrix{sfx}.png, {plot_dir.relative_to(ROOT)}/predictions{sfx}.png, "
          f"{model_dir.relative_to(ROOT)}/test_report{sfx}.txt")


if __name__ == "__main__":
    main()
