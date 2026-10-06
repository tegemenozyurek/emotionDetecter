"""Step 5: Build the CNN, print a layer-by-layer summary, and run one
forward pass on real faces to check that everything is wired correctly.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.model import EmotionCNN, get_device  # noqa: E402

DATA = ROOT / "data" / "processed" / "fer2013.npz"


def print_summary(model, input_shape=(1, 1, 48, 48)):
    rows = []

    def hook(module, _inputs, output):
        params = sum(p.numel() for p in module.parameters(recurse=False))
        rows.append((module.__class__.__name__, tuple(output.shape[1:]), params))

    handles = [m.register_forward_hook(hook) for m in model.modules()
               if not list(m.children())]  # leaf layers only
    model.eval()
    with torch.no_grad():
        model(torch.zeros(input_shape))
    for h in handles:
        h.remove()

    print(f"{'#':>2}  {'layer':<18} {'output shape':<16} {'params':>10}")
    print("-" * 50)
    for i, (name, shape, params) in enumerate(rows, 1):
        print(f"{i:>2}  {name:<18} {str(shape):<16} {params:>10,}")
    print("-" * 50)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {total:,}  (~{total * 4 / 1e6:.1f} MB as float32)")


def main():
    model = EmotionCNN()
    print_summary(model)

    data = np.load(DATA)
    emotions = list(data["emotions"])
    X = torch.from_numpy(data["X_val"][:8]).unsqueeze(1)  # (8, 1, 48, 48)
    y = data["y_val"][:8]

    device = get_device()
    model = model.to(device).eval()
    with torch.no_grad():
        probs = torch.softmax(model(X.to(device)), dim=1).cpu().numpy()

    print(f"\nDevice: {device}")
    print(f"Input batch: {tuple(X.shape)}  ->  output: {probs.shape}")
    print("\nPredictions of the UNTRAINED model on 8 validation faces:")
    print(f"  {'true':<9} {'predicted':<9} confidence")
    for true, p in zip(y, probs):
        print(f"  {emotions[true]:<9} {emotions[p.argmax()]:<9} {p.max():.1%}")
    print(f"\n(Random guessing = {1 / len(emotions):.1%} per class — "
          "the model hasn't learned anything yet.)")


if __name__ == "__main__":
    main()
