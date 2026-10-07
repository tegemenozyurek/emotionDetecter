# 😄 emotionDetecter

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/demo-Gradio-F97316?logo=gradio&logoColor=white)](app.py)
[![Dataset](https://img.shields.io/badge/data-FER--2013%20%2B%20FER%2B-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/msambare/fer2013)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Real-time facial emotion recognition with a CNN trained from scratch.** Upload a photo or turn on your webcam, and the model finds every face and tells you whether it looks angry, disgusted, afraid, happy, neutral, sad or surprised.

![Showcase](assets/showcase.png)

## Highlights

- **80.1% test accuracy** with `eD0.7`, measured on the cleaner [FER+](https://github.com/microsoft/FERPlus) labels (10 annotators per image). The baseline scores 66.8% on the same test.
- **Fixing the labels beat fixing the model.** A third of FER-2013's labels change under FER+. Retraining the same network on the corrected labels added **+6.7 points**.
- **Robust to real-world faces:** 77.5% on tilted, shifted, zoomed and partly covered faces, where the baseline drops to 58.9%.
- **Interactive demo:** a Gradio app with photo upload, live webcam, a model picker, and an *About models* tab that compares every version.
- **Small and fast:** 4.8M parameters (19 MB), runs on a laptop CPU. Each version trains in about an hour on an Apple M4.
- **Built step by step:** every stage (data → preprocessing → model → training → evaluation → improvement) is its own script with its own plots, documented [below](#how-it-was-built).

## Quickstart

The trained weights are included, so you can run the demo right after cloning. No dataset or training needed.

```bash
git clone https://github.com/tegemenozyurek/emotionDetecter.git
cd emotionDetecter
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py        # open http://127.0.0.1:7860
```

The app ([`app.py`](app.py)) finds faces with OpenCV's Haar cascade, crops each to 48×48, classifies it, and draws the result on the image with a probability chart. The **Live webcam** tab updates in real time, and **About models** compares the versions (hover a model name to see what changed).

## Results

All versions are compared on the same **FER+ test set** (6,323 faces with a clear majority label):

| version | what changed | FER+ test | FER+ test, distorted faces | original FER-2013 test |
|---|---|---:|---:|---:|
| `eD0.1` | baseline CNN, 40 epochs, no augmentation | 66.8% | 58.9% | 65.8% |
| `eD0.5` | + data augmentation, 60 epochs | 73.4% | 69.8% | 65.8% |
| `eD0.7` | + trained on FER+ labels | **80.1%** | **77.5%** | 57.1%* |

\* `eD0.7` scores lower on the original labels because it learned the corrected ones: most of the faces FER-2013 calls *fear* or *sad* are *neutral* according to FER+.

![Robustness](assets/robustness_ferplus.png)

Every version lives in its own folder with its weights, settings, logs and test report:

```
models/<version>/   model.pt · config.json · history.csv · train_log.txt · test_report[_ferplus].txt
assets/<version>/   training_curves.png · confusion_matrix[_ferplus].png · predictions[_ferplus].png
```

## Model versions: what's different

All three versions use **the same network** ([`src/model.py`](src/model.py), 4.8M parameters) and the same optimizer (AdamW, lr 1e-3, batch 128, class-weighted loss). They differ only in **what data the network sees during training**:

| | `eD0.1` | `eD0.5` | `eD0.7` |
|---|---|---|---|
| Training labels | FER-2013 (original) | FER-2013 (original) | **FER+** (10 annotators) |
| Training images | 25,837 | 25,837 | 22,447 (ambiguous faces removed) |
| Data augmentation | no | **flip, rotate, shift, zoom, erase** | flip, rotate, shift, zoom, erase |
| Epochs | 40 | 60 | 60 |
| Best epoch | 36 | 60 | 47 |
| Train / val gap at the end | 23 pts (memorizing) | 2 pts | 0 pts |
| **FER+ test accuracy** | 66.8% | 73.4% | **80.1%** |

**`eD0.1`: baseline.** Trained on the raw dataset with no tricks. It learned fast, but after ~20 epochs it started memorizing the training faces (87% train vs 64% validation), and it falls apart on tilted or off-centre faces (58.9%).

**`eD0.5`: fixes overfitting.** Every training image is randomly mirrored, rotated, shifted, zoomed and partly erased, so the network never sees the same picture twice and has to learn the expression itself. Overfitting disappears, and accuracy on distorted faces jumps from 58.9% to 69.8%. This only became visible after switching to cleaner test labels: on the original FER-2013 test set `eD0.1` and `eD0.5` both score 65.8%.

**`eD0.7`: fixes the labels.** Same recipe as `eD0.5`, but trained on FER+ labels, where about a third of FER-2013's labels are corrected. The model stops learning "this calm face is *fear*" and gains another **+6.7 points**. This is the default model in the app.

Per-emotion F1 on the FER+ test set shows where each step helped:

| emotion | `eD0.1` | `eD0.5` | `eD0.7` | test faces |
|---|---:|---:|---:|---:|
| neutral | 62.2% | 72.4% | **82.1%** | 2,272 |
| happy | 87.7% | **91.2%** | 90.1% | 1,758 |
| surprise | 81.3% | 84.0% | **85.9%** | 811 |
| sad | 46.0% | 51.7% | **56.2%** | 735 |
| angry | 61.1% | 67.2% | **74.2%** | 560 |
| fear | 28.2% | 33.7% | **54.2%** | 145 |
| disgust | 41.4% | 34.9% | **41.8%** | 42 |

- **Neutral** made the biggest difference: recall went 49% → 62% → 83%. The older models were trained on labels where many neutral faces were called *sad* or *fear*, so they kept refusing to say *neutral*. Neutral is 36% of the test set, which explains most of the overall gain.
- **Fear** precision more than doubled (18% → 44%). The old models called many sad or neutral faces *fear*, copying the label noise.
- **Disgust** barely moved. FER+ keeps only 119 training images of it, so it stays the weakest class in every version.

## Project structure

```
emotionDetecter/
├── app.py                        # Gradio demo (photo, live webcam, model picker, About models)
├── src/
│   ├── model.py                  # EmotionCNN — 4 conv blocks + classifier head
│   ├── augment.py                # on-GPU augmentation (flip/rotate/shift/zoom/erase)
│   ├── predictor.py              # face detection + preprocessing + inference
│   └── model_info.py             # per-version metrics, read from logs & reports
├── scripts/
│   ├── download_data.py          # 1. fetch FER-2013 from Kaggle
│   ├── explore_data.py           # 2. class balance, samples, average faces
│   ├── preprocess.py             # 3. split, normalize, class weights → .npz
│   ├── model_summary.py          # 4. layer-by-layer summary
│   ├── train.py                  # 5. train a model version
│   ├── evaluate.py               # 6. test-set metrics + confusion matrix
│   ├── preview_augmentation.py   # 7. visualize augmentation
│   ├── robustness.py             # 8. clean vs. distorted comparison
│   ├── prepare_ferplus.py        # 9. relabel the images with FER+ votes
│   └── make_showcase.py          #    README header image
├── models/                       # one folder per trained version
└── assets/                       # all plots
```

## Retrain from scratch

```bash
# 1. Kaggle token (https://www.kaggle.com/settings → API → Create New Token), stored outside the repo:
mkdir -p ~/.kaggle && echo YOUR_TOKEN > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token

# 2. Data
python scripts/download_data.py
python scripts/preprocess.py

# 3. Optional: FER+ labels. Needs data/ferplus/fer2013.csv (original FER-2013 csv) and
#    data/ferplus/fer2013new.csv from https://github.com/microsoft/FERPlus
python scripts/prepare_ferplus.py

# 4. Train + evaluate a new version
python scripts/train.py --name eD0.8 --augment --epochs 80 --patience 5 --data ferplus
python scripts/evaluate.py --model eD0.8 --data ferplus
python scripts/robustness.py --data ferplus
```

The datasets themselves are not in this repo. Images download to `data/fer2013/{train,test}/<emotion>/`.

## How it was built

### Data at a glance

```bash
python scripts/explore_data.py
```

35,887 face images (48×48 grayscale) across 7 emotions. The classes are **heavily imbalanced** — `happy` has ~16× more training images than `disgust`:

![Class distribution](assets/class_distribution.png)

Random samples from each class:

![Sample faces](assets/sample_faces.png)

Averaging every training image per emotion already reveals the signal a model has to learn — the open mouth of *surprise*, the smile of *happy*:

![Average faces](assets/average_faces.png)

### Preprocessing

```bash
python scripts/preprocess.py
```

- **Split:** 10% of the training set is held out as a stratified *validation* set → train 25,837 / val 2,872 / test 7,178. The test set is not touched until final evaluation.
- **Normalize:** pixels are scaled to 0–1, then standardized with the *training* mean/std (0.508 / 0.255) so no information leaks from val/test.
- **Class weights:** inverse-frequency weights counter the imbalance — `disgust` gets 9.4×, `happy` 0.57×.

![Pixel distribution](assets/preprocessing.png)

### Model

```bash
python scripts/model_summary.py
```

A VGG-style CNN in [`src/model.py`](src/model.py) — **4.8M parameters (~19 MB)**. Each block is `conv3×3 → BatchNorm → ReLU` ×2 → `MaxPool` → `Dropout`, doubling channels while halving resolution:

```
1×48×48 → 64×24×24 → 128×12×12 → 256×6×6 → 512×3×3 → global avg pool → 256 → 7
```

Trains on the Apple Silicon GPU (MPS) when available, falling back to CUDA or CPU.

### Training

```bash
python scripts/train.py --name eD0.1      # ~40 min on an Apple M4 (MPS)
```

AdamW (lr 1e-3, weight decay 1e-4), class-weighted cross-entropy, batch 128, 40 epochs. The learning rate is halved when validation accuracy plateaus, and the best checkpoint is kept.

**Baseline model `eD0.1`: 64.9% validation accuracy** (epoch 36) — already around human-level on FER-2013 (~65%).

![Training curves](assets/eD0.1/training_curves.png)

The gap between the curves is classic **overfitting**: training accuracy keeps climbing to 87% while validation stalls around 64% and validation loss rises after epoch ~22. The model is starting to memorize the training faces — step 8 tackles this with data augmentation.

### Evaluation

```bash
python scripts/evaluate.py --model eD0.1 --no-tta
```

`eD0.1` on the held-out **test set (7,178 images): 65.8% accuracy**, macro F1 64.9%.

| emotion | precision | recall | F1 |
|---|---:|---:|---:|
| angry | 58.1% | 58.7% | 58.4% |
| disgust | 71.3% | 69.4% | 70.3% |
| fear | 54.7% | 37.8% | 44.7% |
| happy | 86.4% | 83.0% | 84.7% |
| neutral | 58.1% | 66.7% | 62.1% |
| sad | 51.2% | 58.5% | 54.6% |
| surprise | 78.7% | 80.5% | 79.6% |

![Confusion matrix](assets/eD0.1/confusion_matrix.png)

- **Happy** and **surprise** are easy — a smile and an open mouth are strong, unambiguous signals.
- **Fear** is the hardest: only 38% recall, often mistaken for *sad* (25%) or *angry* (13%).
- **Sad** is the model's "catch-all" for low-energy negative faces — fear, neutral and angry all leak into it.
- Class weighting paid off for **disgust**: 69% recall despite only 436 training images.

![Predictions](assets/eD0.1/predictions.png)

Some "mistakes" are arguably label noise — FER-2013 is known to have mislabeled images.

### Improvement: data augmentation (`eD0.5`)

`eD0.1` memorized its training set (87% train vs 64% val). For `eD0.5`, every training batch is randomly distorted on the GPU ([`src/augment.py`](src/augment.py)): mirrored, rotated ±10°, shifted ±10%, zoomed ±10%, and a random patch is erased. The model never sees the exact same image twice.

```bash
python scripts/preview_augmentation.py
python scripts/train.py --name eD0.5 --augment --epochs 60 --patience 5   # ~55 min
python scripts/evaluate.py --model eD0.5 --no-tta
python scripts/robustness.py
```

![Augmentation](assets/augmentation.png)

Overfitting is gone. Train and validation accuracy now move together, and the best epoch was the last one, so the model was still improving when training stopped:

![eD0.5 training curves](assets/eD0.5/training_curves.png)

On the clean test set both versions score **65.8%**. The difference shows up when the test faces are tilted, off-centre or partly covered, which is how faces look in real photos and webcam frames. There `eD0.1` drops by up to 10 points and `eD0.5` stays close to its clean accuracy (see the [Results](#results) chart).

`eD0.5` is the default model in the app.

### Cleaner labels: FER+ (`eD0.7`)

Looking at the mistakes in step 7, many "wrong" predictions looked right: smiling faces labelled *angry*, calm faces labelled *fear*. Microsoft's [FER+](https://github.com/microsoft/FERPlus) had every FER-2013 image re-labelled by 10 people. Its label file follows the row order of the original `fer2013.csv`, while the Kaggle images have random file names, so [`scripts/prepare_ferplus.py`](scripts/prepare_ferplus.py) matches every image to its row by pixel distance on the GPU. All 35,887 matched, and the old labels agreed 99.8% of the time, which confirms the matching.

Each image gets the FER+ majority vote. Images without a clear majority, non-faces and *contempt* (not one of our 7 classes) are dropped, which leaves 31,264 images. **32% of the remaining labels change.** Most original *fear* and *sad* faces are *neutral* according to the annotators:

![FER+ label changes](assets/ferplus/label_changes.png)

![Relabelled examples](assets/ferplus/relabeled_examples.png)

`eD0.7` uses exactly the `eD0.5` recipe, but trained on FER+ labels:

```bash
python scripts/train.py --name eD0.7 --augment --epochs 60 --patience 5 --data ferplus
python scripts/evaluate.py --model eD0.7 --data ferplus
```

![eD0.7 training curves](assets/eD0.7/training_curves.png)

On the FER+ test set it reaches **80.1%**, against 73.4% for `eD0.5`. Cleaner test labels already lift the older models (`eD0.5`: 65.8% → 73.4%), because many of their "mistakes" were correct answers to wrong labels. Cleaner training labels add another 6.7 points on top.

![eD0.7 confusion matrix](assets/eD0.7/confusion_matrix_ferplus.png)

## Limitations

- **FER-2013 is noisy.** About a third of its labels disagree with the FER+ re-annotation. `eD0.7` trains on FER+, but FER+ still drops ~13% of images that annotators could not agree on, so those harder faces are never evaluated.
- **Rare emotions are still weak.** Under FER+ only 119 training images are *disgust* and 532 *fear*. `eD0.7` reaches 55% and 71% recall on them, but with low precision (34% and 44%).
- **The face detector is the weak link.** The Haar cascade only finds roughly frontal faces and occasionally fires on ears or background. False positives are filtered, but a modern detector (e.g. YuNet) would be more reliable.
- **Dataset bias.** FER-2013 was scraped from the web and is not balanced across age, ethnicity or lighting conditions, so accuracy will vary between people. This is a learning project, not a tool for making decisions about anyone.

## Ideas for next versions

- Swap the Haar cascade for a DNN face detector (YuNet).
- Try transfer learning from a pretrained backbone (e.g. ResNet-18).
- Deploy the demo to Hugging Face Spaces.

## Acknowledgements

FER-2013 was introduced in *Challenges in Representation Learning: A report on three machine learning contests* (Goodfellow et al., 2013) for the ICML 2013 workshop. The FER+ labels come from *Training Deep Networks for Facial Expression Recognition with Crowd-Sourced Label Distribution* (Barsoum et al., 2016), released by Microsoft under the MIT license. The images and labels belong to their original creators and are not redistributed here.

## License

Code released under the [MIT License](LICENSE).
