# CNN-Model-For-Emotion-Classification

# 🎭 Facial Emotion Recognition CNN

A **CNN** model built with PyTorch that recognizes 7 core emotions from a face image —
trained on FER2013 and tested on 7 external datasets to measure real-world generalization,
with targeted fine-tuning for each one.

`Angry` · `Disgust` · `Fear` · `Happy` · `Neutral` · `Sad` · `Surprise`

---

## 📖 Overview

This project is part of a larger system that monitors a student's mood while studying via
webcam, and engages them with questions from the lesson content based on how they're feeling
(focused, confused, frustrated...). This repo contains **only the training and evaluation
code** — the code that connects the model to the web app lives in a separate project.

## 🧠 Model Architecture

An 8-layer convolutional network (32 → 1024 channels) with Batch Normalization after every
layer, Average Pooling for progressive downsampling, and Adaptive Pooling + Dropout before
the final classification layer. See `CNNModel` in the notebook for a fully commented,
layer-by-layer breakdown.

## 📊 Data

| Type | Dataset | Purpose |
|---|---|---|
| Base training | [FER2013](https://www.kaggle.com/datasets/msambare/fer2013) | Training the model from scratch (40 epochs) |
| Generalization testing | CK+, FEMI-DB, EmoRec-DB, RAF-DB, SFEW, AffectNet, JAFFE | Measuring performance on unseen data distributions |
| Fine-tuning | FEMI-DB, EmoRec-DB, RAF-DB, SFEW | Targeted adaptation (freezing most layers) per dataset |

## 📈 Results Summary

| Dataset | Accuracy (Base) | Accuracy (After Fine-tuning) |
|---|---|---|
| FER2013 (test set) | **68.46%** | — |
| CK+ | 76.38% | — |
| FEMI-DB | 77.68% | 76.84% |
| EmoRec-DB* | 67.56% | ~88% |
| RAF-DB | 65.78% | **81.03%** |
| SFEW | 38.52% | — |
| JAFFE | 39.44% | — |

\* EmoRec-DB does not include the Disgust and Fear classes. Full details and confusion
matrices are in the notebook.

## 🚀 Getting Started

```bash
pip install torch torchvision kagglehub scikit-learn matplotlib seaborn tqdm
jupyter notebook Emotion_Recognition_CNN.ipynb
```

Run the cells in order from top to bottom. Base training (Section 5) requires a GPU and
takes a long time — lower `NUM_EPOCHS` for a quick sanity check before running the full
training.

## 📁 Repo Structure

```
.
├── Emotion_Recognition_CNN.ipynb   # Full notebook (training + evaluation + fine-tuning)
├── README.md
└── model/                          # (optional) put trained .pth weights here
```

## 🔌 Using the Trained Model

The model is saved as a `.pth` file (PyTorch state_dict) — use it directly by loading the
same `CNNModel` architecture defined in the notebook:

```python
model = CNNModel(num_classes=7)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()
```

## 📝 Notes

- Input images are **224×224 grayscale**, normalized with `mean=0.5, std=0.5`
- The 7 output classes follow alphabetical order: `Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise`
- Fine-tuning freezes all layers except `conv6` and `fc` — adjustable via `prepare_for_finetuning()`

## 📄 License

Add your project's license here (e.g. MIT) before publishing.
