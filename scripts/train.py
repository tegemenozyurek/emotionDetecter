"""Step 6: Train the CNN on FER-2013.

Each epoch the model sees every training image once, in shuffled mini-batches.
For every batch: predict -> measure how wrong (loss) -> backpropagate -> nudge
all 4.8M weights a little in the direction that reduces the loss.
After each epoch we check accuracy on the validation set and keep the best model.

Outputs:
  models/best.pt                  — weights of the epoch with the best val accuracy
  models/history.csv              — loss / accuracy per epoch
  assets/training_curves.png      — loss & accuracy curves (train vs val)
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.model import EmotionCNN, get_device  # noqa: E402

DATA = ROOT / "data" / "processed" / "fer2013.npz"
MODELS = ROOT / "models"
ASSETS = ROOT / "assets"

TRAIN_COLOR = "#2a78d6"
VAL_COLOR = "#eb6834"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def run_epoch(model, X, y, loss_fn, device, batch_size, optimizer=None):
    """One pass over (X, y). Trains if an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)
    order = torch.randperm(len(X), device=device) if training else torch.arange(len(X), device=device)
    total_loss, correct = 0.0, 0
    with torch.set_grad_enabled(training):
        for start in range(0, len(X), batch_size):
            idx = order[start:start + batch_size]
            xb, yb = X[idx], y[idx]
            logits = model(xb)
            loss = loss_fn(logits, yb)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(idx)
            correct += (logits.argmax(1) == yb).sum().item()
    return total_loss / len(X), correct / len(X)


def plot_history(history, best_epoch):
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=SURFACE)
    panels = [("loss", "Loss (lower is better)"), ("acc", "Accuracy (higher is better)")]
    for ax, (key, title) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.plot(epochs, [h[f"train_{key}"] for h in history], color=TRAIN_COLOR, lw=2, label="train")
        ax.plot(epochs, [h[f"val_{key}"] for h in history], color=VAL_COLOR, lw=2, label="validation")
        ax.axvline(best_epoch, color=TEXT_MUTED, lw=1, ls="--")
        ax.text(best_epoch, ax.get_ylim()[1], f" best epoch {best_epoch}", va="top",
                fontsize=9, color=TEXT_MUTED)
        if key == "acc":
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_title(title, loc="left", fontsize=12, color=TEXT)
        ax.set_xlabel("epoch", color=TEXT_MUTED)
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=TEXT_MUTED, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.legend(frameon=False)
    fig.suptitle("Training curves", x=0.01, ha="left", fontsize=14, color=TEXT)
    fig.tight_layout()
    fig.savefig(ASSETS / "training_curves.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = get_device()
    data = np.load(DATA)

    # The whole dataset (~250 MB) fits in GPU memory, so load it once.
    to_x = lambda a: torch.from_numpy(a).unsqueeze(1).to(device)
    to_y = lambda a: torch.from_numpy(a).to(device)
    X_train, y_train = to_x(data["X_train"]), to_y(data["y_train"])
    X_val, y_val = to_x(data["X_val"]), to_y(data["y_val"])
    class_weights = torch.from_numpy(data["class_weights"]).to(device)

    model = EmotionCNN().to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Halve the learning rate when val accuracy stops improving for 3 epochs.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    MODELS.mkdir(exist_ok=True)
    ASSETS.mkdir(exist_ok=True)
    history, best_acc, best_epoch = [], 0.0, 0
    print(f"Training on {device} | {len(X_train):,} train / {len(X_val):,} val images | "
          f"{args.epochs} epochs, batch {args.batch_size}, lr {args.lr}\n")
    print(f"{'epoch':>5} {'train_loss':>10} {'train_acc':>9} {'val_loss':>9} {'val_acc':>8} {'lr':>8} {'time':>6}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, X_train, y_train, loss_fn, device, args.batch_size, optimizer)
        val_loss, val_acc = run_epoch(model, X_val, y_val, loss_fn, device, args.batch_size)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_acc)

        marker = ""
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), MODELS / "best.pt")
            marker = "  * saved"
        history.append(dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                            val_loss=val_loss, val_acc=val_acc, lr=lr))
        print(f"{epoch:>5} {train_loss:>10.4f} {train_acc:>9.1%} {val_loss:>9.4f} {val_acc:>8.1%} "
              f"{lr:>8.1e} {time.time() - t0:>5.1f}s{marker}", flush=True)

    with open(MODELS / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    plot_history(history, best_epoch)

    print(f"\nBest validation accuracy: {best_acc:.1%} at epoch {best_epoch}")
    print(f"Saved models/best.pt, models/history.csv, assets/training_curves.png")


if __name__ == "__main__":
    main()
