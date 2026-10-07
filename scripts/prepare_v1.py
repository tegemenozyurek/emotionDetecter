"""eDv1.0 step 3: build the training set from FER+ and RAF-DB.

Labels
  FER+     soft labels: the share of the 10 annotators that picked each emotion.
           Images without a clear majority are no longer dropped — "6 neutral,
           4 sad" is useful information. Only images where most votes went to
           contempt / unknown / not-a-face are left out.
  RAF-DB   one label per image (agreed by ~40 annotators), lightly smoothed.

Splits
  train    FER+ Training + 90% of RAF-DB train
  val      eD0.7's exact FER+ validation images + 10% of RAF-DB train
           (kept identical so eD0.7 can be weighed fairly inside the teacher ensemble)
  test     FER+ test faces with a clear majority (the same 6,323 as before)
           + the official RAF-DB test set (3,068)

Output: data/processed/v1.npz
"""
import argparse
import importlib.util
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("prepare_ferplus", ROOT / "scripts" / "prepare_ferplus.py")
pf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pf)

FACES = ROOT / "data" / "aligned" / "faces.npz"
OUT = ROOT / "data" / "processed" / "v1.npz"
EMOTIONS = pf.EMOTIONS
RAF_CLASSES = {1: "surprise", 2: "fear", 3: "disgust", 4: "happy", 5: "sad", 6: "angry", 7: "neutral"}
VOTE_ORDER = [EMOTIONS.index(pf.VOTE_TO_EMOTION[c]) for c in pf.VOTE_COLUMNS[:7]]  # vote column -> our index
SMOOTHING = 0.1
TRAIN, VAL, TEST = 0, 1, 2
FER, RAF = 0, 1


def fer_soft_label(v):
    """FER+ votes (10 columns) -> 7-class distribution, or None if the image is not usable."""
    emotion_votes = np.zeros(7)
    emotion_votes[VOTE_ORDER] = v[:7]
    if emotion_votes.sum() < v.sum() / 2:  # most annotators said contempt / unknown / not a face
        return None
    cleaned = np.where(emotion_votes <= 1, 0, emotion_votes)  # one-vote outliers
    if cleaned.sum() > 0:
        emotion_votes = cleaned
    return emotion_votes / emotion_votes.sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=64, help="face size; 96 reads faces_96.npz, writes v1_96.npz")
    size = parser.parse_args().size
    faces_path = FACES if size == 64 else FACES.with_name(f"faces_{size}.npz")
    out = OUT if size == 64 else OUT.with_name(f"v1_{size}.npz")
    faces = np.load(faces_path)
    index = {p: i for i, p in enumerate(faces["paths"])}
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # --- FER+: match images to csv rows (same procedure as scripts/prepare_ferplus.py)
    print("Matching FER images to FER+ votes ...")
    csv_pixels, _, csv_usage, votes = pf.load_csv_rows()
    fer_rows = {}
    for split, usage in [("train", "Training"), ("test", "PublicTest"), ("test", "PrivateTest")]:
        imgs, folder_labels, names = pf.load_our_images(split, usage)
        rows = np.flatnonzero(csv_usage == usage)
        best, _ = pf.nearest_rows(imgs, csv_pixels[rows], device)
        paths = [f"data/fer2013/{split}/{EMOTIONS[l]}/{n}" for l, n in zip(folder_labels, names)]
        fer_rows[usage] = (paths, rows[best])

    # eD0.7's validation images: rebuild its split exactly.
    paths, rows = fer_rows["Training"]
    majority = [pf.ferplus_label(votes[r])[0] for r in rows]
    kept = [i for i, lab in enumerate(majority) if lab is not None]
    _, val_idx = pf.stratified_split(np.array([majority[i] for i in kept]), pf.VAL_FRACTION, pf.SEED)
    fer_val = {paths[kept[i]] for i in val_idx}

    items = []  # (face index, soft label, hard label or -1, source, split)
    dropped = Counter()
    for usage, (paths, rows) in fer_rows.items():
        for path, r in zip(paths, rows):
            soft = fer_soft_label(votes[r])
            hard, _ = pf.ferplus_label(votes[r])
            if soft is None and hard is not None:  # keep val/test faces that do have a majority
                soft = np.eye(7)[hard]
            if usage == "Training":
                if path in fer_val:
                    items.append((index[path], soft, hard, FER, VAL))
                elif soft is not None:
                    items.append((index[path], soft, -1 if hard is None else hard, FER, TRAIN))
                else:
                    dropped["FER+ train: not a usable face"] += 1
            elif hard is not None:  # test: only faces with a clear majority, as before
                items.append((index[path], soft, hard, FER, TEST))
            else:
                dropped["FER+ test: no clear majority"] += 1

    # --- RAF-DB
    raf_train = []
    for split in ["train", "test"]:
        for code, emotion in RAF_CLASSES.items():
            label = EMOTIONS.index(emotion)
            for p in sorted((ROOT / "data" / "rafdb" / "DATASET" / split / str(code)).glob("*.jpg")):
                soft = np.full(7, SMOOTHING / 6)
                soft[label] = 1 - SMOOTHING
                rel = str(p.relative_to(ROOT))
                if split == "train":
                    raf_train.append((index[rel], soft, label))
                else:
                    items.append((index[rel], soft, label, RAF, TEST))
    labels = np.array([lab for _, _, lab in raf_train])
    train_idx, val_idx = pf.stratified_split(labels, pf.VAL_FRACTION, pf.SEED)
    for i in train_idx:
        items.append((*raf_train[i], RAF, TRAIN))
    for i in val_idx:
        items.append((*raf_train[i], RAF, VAL))

    idx = np.array([it[0] for it in items])
    soft = np.stack([it[1] for it in items]).astype(np.float32)
    hard = np.array([it[2] for it in items], dtype=np.int64)
    source = np.array([it[3] for it in items], dtype=np.uint8)
    split = np.array([it[4] for it in items], dtype=np.uint8)
    X = faces["aligned"][idx]
    X48 = faces["fer_framed"][idx]
    pixels = X[split == TRAIN].astype(np.float32) / 255.0
    mean, std = float(pixels.mean()), float(pixels.std())

    print("\nImages per split (FER+ / RAF-DB):")
    for s, name in [(TRAIN, "train"), (VAL, "val"), (TEST, "test")]:
        f, r = np.sum((split == s) & (source == FER)), np.sum((split == s) & (source == RAF))
        print(f"  {name:<5} {f:>6,} + {r:>6,} = {f + r:>6,}")
    ambiguous = np.sum((split == TRAIN) & (source == FER) & (hard == -1))
    print(f"  (train includes {ambiguous:,} FER+ faces without a clear majority, as soft labels)")
    for k, v in dropped.items():
        print(f"  dropped — {k}: {v:,}")

    print(f"\n{'emotion':<9} {'train':>7} {'eD0.7 had':>9}")
    train_votes = soft[split == TRAIN].sum(0)  # soft count: fractional images per emotion
    before = [2100, 119, 532, 7287, 8740, 3014, 3149]  # eD0.7's FER+ train counts (prepare_ferplus.py)
    for i, e in enumerate(EMOTIONS):
        print(f"{e:<9} {train_votes[i]:>7,.0f} {before[i]:>9,}")
    print(f"\nPixel mean/std on aligned train faces: {mean:.4f} / {std:.4f}")

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, X48=X48, soft=soft, hard=hard, source=source, split=split,
                        paths=faces["paths"][idx], mean=np.float32(mean), std=np.float32(std),
                        emotions=np.array(EMOTIONS))
    print(f"Saved {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
