# emotionDetecter

Facial emotion recognition trained on the [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) dataset — 48×48 grayscale faces, 7 emotions: angry, disgust, fear, happy, sad, surprise, neutral.

> 🚧 Work in progress — built step by step.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

The dataset is **not** included in this repo. To download it:

1. Create a Kaggle API token at <https://www.kaggle.com/settings> → *API* → *Create New Token*.
2. Save it outside the repo:
   ```bash
   mkdir -p ~/.kaggle && echo YOUR_TOKEN > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
   ```
3. Run:
   ```bash
   python scripts/download_data.py
   ```

The images land in `data/fer2013/{train,test}/<emotion>/`.

## Roadmap

- [x] 1. Project skeleton
- [ ] 2. Download dataset
- [ ] 3. Explore the data
- [ ] 4. Preprocessing
- [ ] 5. CNN model
- [ ] 6. Training
- [ ] 7. Evaluation
- [ ] 8. Improvements (augmentation, tuning)
- [ ] 9. Live webcam demo
- [ ] 10. Polish & results
