"""Train the face gate: a tiny CNN that says whether an aligned 64x64 crop is a human face.

The threshold is chosen on validation so that 99.5% of human faces pass: losing a real
person is worse than letting an occasional look-alike through.

Outputs: models/_gate/{model.pt, config.json, report.txt}, web/models/gate.onnx
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.augment import augment_video  # noqa: E402
from src.model import EmotionResNet, get_device  # noqa: E402

KEEP_HUMANS = 0.995
EPOCHS, BATCH = 8, 256


def main():
    torch.manual_seed(0)
    device = get_device()
    d = np.load(ROOT / "data" / "processed" / "gate.npz")
    groups = [str(g) for g in d["groups"]]
    tr, va = np.flatnonzero(d["split"] == 0), np.flatnonzero(d["split"] == 1)
    X = torch.from_numpy(d["X"]).to(device)
    y = torch.from_numpy(d["y"]).float().to(device)
    mean, std = 0.5, 0.25

    # balanced batches: humans and non-humans equally often
    w = np.where(d["y"][tr] == 1, 1 / (d["y"][tr] == 1).sum(), 1 / (d["y"][tr] == 0).sum())
    model = EmotionResNet(num_classes=1, width=16, dropout=0.2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.05)
    steps = EPOCHS * (len(tr) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    print(f"gate: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, {len(tr):,} train / {len(va):,} val")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        idx_all = torch.from_numpy(np.random.choice(tr, size=(len(tr) // BATCH) * BATCH, p=w / w.sum())).to(device)
        for idx in idx_all.view(-1, BATCH):
            x = augment_video(X[idx].float().div(255).unsqueeze(1))
            loss = F.binary_cross_entropy_with_logits(model((x - mean) / std).squeeze(1), y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        p_va = predict(model, X, va, mean, std)
        acc = ((p_va > 0.5) == d["y"][va]).mean()
        print(f"epoch {epoch}  loss {loss.item():.4f}  val acc {acc:.2%}  {time.time() - t0:.0f}s", flush=True)

    p_va = predict(model, X, va, mean, std)
    human = d["y"][va] == 1
    threshold = float(np.quantile(p_va[human], 1 - KEEP_HUMANS))
    lines = [f"Face gate | threshold {threshold:.3f} (keeps {KEEP_HUMANS:.1%} of human faces on validation)", ""]
    for g, name in enumerate(groups):
        for only_detected in (False, True):
            m = (d["group"][va] == g) & (d["detected"][va] | (not only_detected))
            if not m.any():
                continue
            passed = (p_va[m] >= threshold).mean()
            what = "kept" if g == 0 else "wrongly kept"
            lines.append(f"  {name:<8} {'(detector fired)' if only_detected else '(all)':<17} {m.sum():>5}  {what} {passed:.2%}")
    report = "\n".join(lines)
    print(report)

    out = ROOT / "models" / "_gate"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    (out / "report.txt").write_text(report + "\n")
    (out / "config.json").write_text(json.dumps({"size": 64, "mean": mean, "std": std, "threshold": threshold,
                                                 "width": 16, "template": "canonical"}, indent=2) + "\n")

    # export for the web app, checked against PyTorch
    model = model.cpu().eval()
    path = ROOT / "web" / "models" / "gate.onnx"
    torch.onnx.export(model, torch.zeros(1, 1, 64, 64), str(path), dynamo=False, opset_version=17,
                      input_names=["input"], output_names=["logit"],
                      dynamic_axes={"input": {0: "batch"}, "logit": {0: "batch"}})
    xt = torch.randn(8, 1, 64, 64)
    diff = np.abs(model(xt).detach().numpy() - ort.InferenceSession(str(path)).run(None, {"input": xt.numpy()})[0]).max()
    print(f"Saved models/_gate/ and web/models/gate.onnx ({path.stat().st_size / 1e6:.1f} MB, max diff {diff:.1e})")


@torch.no_grad()
def predict(model, X, idx, mean, std):
    model.eval()
    out = []
    for s in range(0, len(idx), 1024):
        x = X[torch.from_numpy(idx[s:s + 1024]).to(X.device)].float().div(255).unsqueeze(1)
        out.append(torch.sigmoid(model((x - mean) / std).squeeze(1)).cpu())
    return torch.cat(out).numpy()


if __name__ == "__main__":
    main()
