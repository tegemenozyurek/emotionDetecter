"""eDv1.0 step 2: align every FER-2013 and RAF-DB face to one template.

Input: data/aligned/keypoints.jsonl, written by tools/landmarks.html, which runs
MediaPipe's face detector (the same one the web app uses) on every image.

Two templates are derived from the data itself:
  canonical (64x64)  median keypoints of RAF-DB, whose images are already
                     eye-aligned. eDv1.0 is trained and run on this framing.
  FER (48x48)        median keypoints of FER-2013. The older eD0.x models
                     expect this looser framing, so RAF-DB faces (and webcam
                     faces in the app) are warped to it before reaching them.

Outputs:
  data/aligned/faces.npz        aligned images + keypoint status per image
  models/alignment.json         both templates, used by the web app
  assets/v1/alignment.png       before / after examples
"""
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.align import KEYPOINTS, fit_similarity, plausible  # noqa: E402

KP_FILE = ROOT / "data" / "aligned" / "keypoints.jsonl"
OUT = ROOT / "data" / "aligned" / "faces.npz"
TEMPLATES = ROOT / "models" / "alignment.json"
ASSETS = ROOT / "assets" / "v1"
SIZE, FER_SIZE = 64, 48

TEXT = "#0b0b0b"
SURFACE = "#fcfcfb"
DOT = "#eb6834"


def load_keypoints():
    rows = {}
    with open(KP_FILE) as f:
        for line in f:
            row = json.loads(line)
            rows[row["path"]] = row  # last write wins if the tool ran twice
    return rows


def main():
    rows = load_keypoints()
    paths = sorted(rows)
    print(f"{len(paths):,} images with detector output")

    images, kps, good = [], [], []
    for p in paths:
        img = cv2.imread(str(ROOT / p), cv2.IMREAD_GRAYSCALE)
        row = rows[p]
        kp = np.array(row["kp"], np.float64) if row["ok"] else None
        ok = kp is not None and plausible(kp, img.shape[1], img.shape[0])
        images.append(img)
        kps.append(kp if ok else np.full((6, 2), np.nan))
        good.append(ok)
    kps, good = np.stack(kps), np.array(good)
    is_raf = np.array([p.startswith("data/rafdb/") for p in paths])

    # Templates = median keypoint positions of each dataset.
    raf_template = np.median(kps[good & is_raf][:, KEYPOINTS], axis=0)          # in 100x100 px
    fer_template = np.median(kps[good & ~is_raf][:, KEYPOINTS], axis=0)         # in 48x48 px
    canonical = raf_template * SIZE / 100.0
    print("canonical template (64x64):", np.round(canonical, 1).tolist())
    print("FER template (48x48):      ", np.round(fer_template, 1).tolist())

    aligned = np.zeros((len(paths), SIZE, SIZE), np.uint8)
    fer_framed = np.zeros((len(paths), FER_SIZE, FER_SIZE), np.uint8)
    for i, (img, kp) in enumerate(zip(images, kps)):
        # Faces the detector missed fall back to their dataset's average framing.
        src = kp[KEYPOINTS] if good[i] else (raf_template if is_raf[i] else fer_template)
        aligned[i] = cv2.warpAffine(img, fit_similarity(src, canonical), (SIZE, SIZE),
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        if is_raf[i]:
            fer_framed[i] = cv2.warpAffine(img, fit_similarity(src, fer_template), (FER_SIZE, FER_SIZE),
                                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        else:
            fer_framed[i] = img  # FER images already have the FER framing

    for name, mask in [("FER-2013", ~is_raf), ("RAF-DB", is_raf)]:
        print(f"  {name:<8} {mask.sum():>6,} images, keypoints usable for {good[mask].mean():.1%}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, paths=np.array(paths), aligned=aligned, fer_framed=fer_framed,
                        keypoints_ok=good, is_raf=is_raf)
    TEMPLATES.write_text(json.dumps({
        "keypoints": ["right eye", "left eye", "nose tip", "mouth center"],
        "canonical": {"size": SIZE, "points": np.round(canonical, 3).tolist()},
        "fer": {"size": FER_SIZE, "points": np.round(fer_template, 3).tolist()},
    }, indent=2) + "\n")
    print(f"Saved {OUT.relative_to(ROOT)} and {TEMPLATES.relative_to(ROOT)}")

    # Before / after examples: original with keypoints, then the aligned face.
    ASSETS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(4)
    picks = list(rng.choice(np.flatnonzero(good & ~is_raf), 6, replace=False)) + \
        list(rng.choice(np.flatnonzero(good & is_raf), 4, replace=False))
    fig, axes = plt.subplots(2, len(picks), figsize=(len(picks) * 1.45, 3.3), facecolor=SURFACE)
    for col, i in enumerate(picks):
        top, bottom = axes[0, col], axes[1, col]
        top.imshow(images[i], cmap="gray", vmin=0, vmax=255)
        top.scatter(kps[i][KEYPOINTS, 0], kps[i][KEYPOINTS, 1], s=9, c=DOT)
        top.set_title("RAF-DB" if is_raf[i] else "FER-2013", fontsize=8, color=TEXT)
        bottom.imshow(aligned[i], cmap="gray", vmin=0, vmax=255)
        bottom.scatter(canonical[:, 0], canonical[:, 1], s=9, c=DOT)
        for ax in (top, bottom):
            ax.axis("off")
    axes[0, 0].text(-0.15, 0.5, "original", transform=axes[0, 0].transAxes, rotation=90,
                    va="center", ha="right", fontsize=10, color=TEXT)
    axes[1, 0].text(-0.15, 0.5, "aligned", transform=axes[1, 0].transAxes, rotation=90,
                    va="center", ha="right", fontsize=10, color=TEXT)
    fig.suptitle("Face alignment: eyes, nose and mouth moved onto one template", x=0.01, ha="left",
                 fontsize=12, color=TEXT)
    fig.tight_layout()
    fig.savefig(ASSETS / "alignment.png", dpi=150)
    plt.close(fig)
    print(f"Saved {(ASSETS / 'alignment.png').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
