"""Face gate data: is this aligned crop a human face, or an animal / cartoon / emoji?

The face detector also fires on cats, dogs, cartoon characters and emoji. To learn
to reject those, every non-human image goes through the same MediaPipe detector
(tools/landmarks.html) and the same alignment as the human faces, so the gate sees
exactly the crops it will see live. Images where the detector found nothing are
cut out assuming the face is centred (AFHQ and Cartoon Set are face-centred).

Everything is grayscale: FER-2013 is grayscale, and colour would let the gate learn
"gray = human" instead of what a human face looks like.

Inputs:  data/aligned/faces.npz             human faces (FER-2013 + RAF-DB), aligned 64x64
         data/aligned/neg_keypoints.jsonl   detector output on data/negatives/{afhq,cartoon,emoji}
Output:  data/processed/gate.npz
"""
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.align import KEYPOINTS, fit_similarity, plausible  # noqa: E402

GROUPS = ["human", "animal", "cartoon", "emoji"]
SIZE = 64


def main():
    templates = json.loads((ROOT / "models" / "alignment.json").read_text())
    canonical = np.array(templates["canonical"]["points"])
    raf_template = canonical * 100 / 64  # RAF-DB framing, in 100 px coordinates

    images, groups, detected = [], [], []
    rows = [json.loads(line) for line in open(ROOT / "data" / "aligned" / "neg_keypoints.jsonl")]
    for row in rows:
        img = cv2.imread(str(ROOT / row["path"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        kp = np.array(row["kp"]) if row["ok"] else None
        ok = kp is not None and plausible(kp, img.shape[1], img.shape[0])
        src = kp[KEYPOINTS] if ok else raf_template * min(img.shape) / 100 + (np.array(img.shape[::-1]) - min(img.shape)) / 2
        images.append(cv2.warpAffine(img, fit_similarity(src, canonical), (SIZE, SIZE),
                                     flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE))
        group = row["path"].split("/")[2]
        groups.append({"afhq": 1, "cartoon": 2, "emoji": 3}[group])
        detected.append(ok)

    faces = np.load(ROOT / "data" / "aligned" / "faces.npz")
    keep = faces["keypoints_ok"]  # human crops with real keypoints, like live
    human = faces["aligned"][keep]

    X = np.concatenate([human, np.stack(images)])
    y = np.concatenate([np.ones(len(human), np.uint8), np.zeros(len(images), np.uint8)])
    group = np.concatenate([np.zeros(len(human), np.uint8), np.array(groups, np.uint8)])
    det = np.concatenate([np.ones(len(human), bool), np.array(detected)])
    split = (np.random.default_rng(0).random(len(X)) < 0.1).astype(np.uint8)  # 1 = validation

    print(f"{len(X):,} crops")
    for g, name in enumerate(GROUPS):
        m = group == g
        print(f"  {name:<8} {m.sum():>6,}   detector fired on {det[m].mean():.0%}")
    out = ROOT / "data" / "processed" / "gate.npz"
    np.savez_compressed(out, X=X, y=y, group=group, detected=det, split=split, groups=np.array(GROUPS))
    print(f"Saved {out.relative_to(ROOT)} | validation: {Counter(split.tolist())[1]:,}")

    # a quick look at what the gate will see
    rng = np.random.default_rng(1)
    rows_img = []
    for g in range(4):
        idx = rng.choice(np.flatnonzero((group == g) & det), 10, replace=False)
        rows_img.append(np.concatenate(list(X[idx]), axis=1))
    cv2.imwrite(str(ROOT / "assets" / "v1" / "gate_examples.png"),
                cv2.resize(np.concatenate(rows_img, axis=0), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))


if __name__ == "__main__":
    main()
