"""Train an eDv1.0-family model on aligned 64x64 faces (FER+ + RAF-DB).

What is new compared to scripts/train.py:
  - soft labels (FER+ vote shares) instead of a single label per face
  - webcam-style augmentation (blur, low resolution, lighting, noise, occlusion)
  - warmup + cosine learning-rate schedule, and an exponential moving average (EMA)
    of the weights, which is what gets evaluated and saved
  - optional knowledge distillation: the model also learns from an ensemble of
    teacher models. Teacher weights are picked on the validation set.

Usage:
  python scripts/train_v1.py --name _teachers/T1 --arch resnet
  python scripts/train_v1.py --name _teachers/T2 --arch vgg
  python scripts/train_v1.py --name eDv1.0 --arch resnet --teachers _teachers/T1,_teachers/T2,eD0.7

Outputs: models/<name>/{model.pt, config.json, history.csv, train_log.txt}
         assets/<name>/training_curves.png
"""
import argparse
import csv
import importlib.util
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.augment import augment_video  # noqa: E402
from src.model import EmotionCNN, build_model, get_device  # noqa: E402

DATA = ROOT / "data" / "processed" / "v1.npz"
MODELS = ROOT / "models"
ASSETS = ROOT / "assets"
FER_MEAN, FER_STD = 0.50785, 0.25500  # normalization of the eD0.x models
TRAIN, VAL = 0, 1

_spec = importlib.util.spec_from_file_location("train_v0", ROOT / "scripts" / "train.py")
_train_v0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train_v0)
plot_history = _train_v0.plot_history


def load_model(name, device):
    """Load any trained version. Returns (model, config)."""
    folder = MODELS / name
    config = json.loads((folder / "config.json").read_text())
    model = build_model(config["arch"], config.get("width", 64)) if config.get("input") == "aligned64" else EmotionCNN()
    model.load_state_dict(torch.load(folder / "model.pt", map_location=device))
    return model.to(device).eval(), config


@torch.no_grad()
def predict_logits(model, config, data, idx, device, batch_size=512):
    """Logits for data[idx], each model getting the framing it was trained on, flip-averaged."""
    aligned = config.get("input") == "aligned64"
    X = data["X"] if aligned else data["X48"]
    mean, std = (config["mean"], config["std"]) if aligned else (FER_MEAN, FER_STD)
    out = []
    for start in range(0, len(idx), batch_size):
        xb = torch.from_numpy(X[idx[start:start + batch_size]]).float().div(255).unsqueeze(1).to(device)
        xb = (xb - mean) / std
        out.append(((model(xb) + model(torch.flip(xb, dims=[3]))) / 2).cpu())
    return torch.cat(out)


def pick_teacher_weights(val_logits, val_labels, temperature):
    """Grid-search ensemble weights (steps of 0.1) that maximize validation accuracy."""
    probs = [F.softmax(z / temperature, 1) for z in val_logits]
    best = (-1.0, None)
    steps = [i / 10 for i in range(11)]
    for w in itertools.product(steps, repeat=len(probs)):
        if abs(sum(w) - 1) > 1e-6:
            continue
        acc = (sum(wi * p for wi, p in zip(w, probs)).argmax(1) == val_labels).float().mean().item()
        if acc > best[0] + 1e-9:
            best = (acc, w)
    return best


class EMA:
    """Exponential moving average of the weights; usually generalizes better than the raw weights."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.steps = 0

    @torch.no_grad()
    def update(self, model):
        self.steps += 1
        d = min(self.decay, (1 + self.steps) / (10 + self.steps))
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1 - d)
            else:
                self.shadow[k].copy_(v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--arch", default="resnet", choices=["resnet", "vgg"])
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup", type=int, default=3, help="warmup epochs")
    parser.add_argument("--ema", type=float, default=0.999)
    parser.add_argument("--teachers", default="", help="comma-separated model names to distill from")
    parser.add_argument("--alpha", type=float, default=0.5, help="share of the loss coming from the teachers")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    data = dict(np.load(DATA))
    split = data["split"]
    tr, va = np.flatnonzero(split == TRAIN), np.flatnonzero(split == VAL)
    mean, std = float(data["mean"]), float(data["std"])

    out_dir, plot_dir = MODELS / args.name, ASSETS / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(out_dir / "train_log.txt", "w")

    def log(line=""):
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    # Data on the GPU: uint8 images, soft labels, per-sample class-balance weights.
    X_tr = torch.from_numpy(data["X"][tr]).to(device)
    X_va = torch.from_numpy(data["X"][va]).to(device)
    soft_tr = torch.from_numpy(data["soft"][tr]).to(device)
    soft_va = torch.from_numpy(data["soft"][va]).to(device)
    hard_va = torch.from_numpy(data["hard"][va]).to(device)
    src_va = data["source"][va]
    counts = soft_tr.sum(0)
    class_w = (counts.sum() / (7 * counts)).sqrt()  # sqrt-inverse-frequency: helps rare emotions, gently
    sample_w = soft_tr @ class_w
    sample_w /= sample_w.mean()

    # Optional teachers -> one soft target per training image.
    teacher_tr, teacher_info = None, []
    if args.teachers:
        names = [t.strip() for t in args.teachers.split(",") if t.strip()]
        log(f"Teachers: {', '.join(names)}")
        val_logits, tr_logits = [], []
        for name in names:
            model, config = load_model(name, device)
            val_logits.append(predict_logits(model, config, data, va, device))
            tr_logits.append(predict_logits(model, config, data, tr, device))
            acc = (val_logits[-1].argmax(1) == hard_va.cpu()).float().mean().item()
            log(f"  {name:<14} val acc {acc:.1%}")
            teacher_info.append({"name": name, "val_acc": acc})
        acc, weights = pick_teacher_weights(val_logits, hard_va.cpu(), args.temperature)
        log(f"  ensemble weights {dict(zip(names, weights))} -> val acc {acc:.1%}\n")
        for info, w in zip(teacher_info, weights):
            info["weight"] = w
        teacher_tr = sum(w * F.softmax(z / args.temperature, 1) for w, z in zip(weights, tr_logits)).to(device)

    model = build_model(args.arch, args.width).to(device)
    ema = EMA(model, args.ema)
    eval_model = build_model(args.arch, args.width).to(device).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(tr) / args.batch_size)
    total, warmup = args.epochs * steps_per_epoch, args.warmup * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warmup if s < warmup else
                                              0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup))))

    config = {**vars(args), "input": "aligned64", "size": 64, "template": "canonical",
              "mean": mean, "std": std, "data": "v1", "teachers_used": teacher_info}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    params = sum(p.numel() for p in model.parameters())
    log(f"Model {args.name} | {args.arch} width {args.width} ({params / 1e6:.1f}M params) | training on {device} | "
        f"{len(tr):,} train / {len(va):,} val images | {args.epochs} epochs, batch {args.batch_size}, "
        f"lr {args.lr}, data=v1 (FER+ + RAF-DB, aligned 64x64)\n")
    log(f"{'epoch':>5} {'train_loss':>10} {'train_acc':>9} {'val_loss':>9} {'val_acc':>8} {'lr':>8} {'time':>6}"
        f" {'FER+val':>8} {'RAFval':>7}")

    history, best_acc, best_epoch = [], 0.0, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(tr), device=device)
        loss_sum, correct = 0.0, 0
        for start in range(0, len(tr), args.batch_size):
            idx = perm[start:start + args.batch_size]
            x = augment_video(X_tr[idx].float().div(255).unsqueeze(1))
            logits = model((x - mean) / std)
            logp = F.log_softmax(logits, 1)
            loss = ((-(soft_tr[idx] * logp).sum(1)) * sample_w[idx]).mean()
            if teacher_tr is not None:
                t = teacher_tr[idx]
                kd = (t * (t.clamp_min(1e-8).log() - F.log_softmax(logits / args.temperature, 1))).sum(1)
                loss = (1 - args.alpha) * loss + args.alpha * (kd * sample_w[idx]).mean() * args.temperature ** 2
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            ema.update(model)
            loss_sum += loss.item() * len(idx)
            correct += (logits.argmax(1) == soft_tr[idx].argmax(1)).sum().item()

        # Evaluate the EMA weights on validation.
        eval_model.load_state_dict(ema.shadow)
        with torch.no_grad():
            val_logits = torch.cat([eval_model((X_va[s:s + 512].float().div(255).unsqueeze(1) - mean) / std)
                                    for s in range(0, len(va), 512)])
        val_loss = (-(soft_va * F.log_softmax(val_logits, 1)).sum(1)).mean().item()
        hit = (val_logits.argmax(1) == hard_va).cpu().numpy()
        val_acc, fer_acc, raf_acc = hit.mean(), hit[src_va == 0].mean(), hit[src_va == 1].mean()
        lr = opt.param_groups[0]["lr"]

        marker = ""
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save(ema.shadow, out_dir / "model.pt")
            marker = "  * saved"
        train_loss, train_acc = loss_sum / len(tr), correct / len(tr)
        history.append(dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc, val_loss=val_loss,
                            val_acc=val_acc, lr=lr, fer_val=fer_acc, raf_val=raf_acc))
        log(f"{epoch:>5} {train_loss:>10.4f} {train_acc:>9.1%} {val_loss:>9.4f} {val_acc:>8.1%} {lr:>8.1e} "
            f"{time.time() - t0:>5.1f}s {fer_acc:>8.1%} {raf_acc:>7.1%}{marker}")

    with open(out_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    plot_history(history, best_epoch, plot_dir / "training_curves.png")
    log(f"\nBest validation accuracy: {best_acc:.1%} at epoch {best_epoch}")
    log(f"Saved {out_dir.relative_to(ROOT)}/ and {plot_dir.relative_to(ROOT)}/training_curves.png")
    log_file.close()


if __name__ == "__main__":
    main()
