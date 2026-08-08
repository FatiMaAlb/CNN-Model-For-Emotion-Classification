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

## 🔍 Detailed Performance & Generalization Analysis

The model was trained **only** on FER2013, then evaluated as-is (no additional training) on
7 external datasets to see how well it holds up on faces it has never encountered — different
cameras, lighting, ethnicities, and image quality. Here's what that testing revealed.

### Per-class breakdown

| Dataset | Angry | Disgust | Fear | Happy | Neutral | Sad | Surprise |
|---|---|---|---|---|---|---|---|
| **CK+** (base) | 71.8% | 65.1% | 47.2% | 100.0% | — | 65.8% | 95.1% |
| **FEMI-DB** (base) | 70.0% | 81.0% | 64.7% | 94.0% | 82.0% | 68.0% | 86.3% |
| **FEMI-DB** (fine-tuned) | 70.3% | 77.0% | 66.0% | 93.0% | 76.3% | 68.3% | 87.0% |
| **EmoRec-DB** (base) | 78.8% | n/a* | n/a* | 95.1% | 67.4% | 47.6% | 36.5% |
| **EmoRec-DB** (fine-tuned) | 94.8% | 96.6% | 84.6% | 81.3% | 92.2% | — | — |
| **RAF-DB** (base) | 72.8% | 3.8% | 24.3% | 79.0% | 61.8% | 71.3% | 54.4% |
| **RAF-DB** (fine-tuned) | 75.3% | 41.9% | 47.3% | 91.6% | 84.0% | 74.3% | 76.3% |
| **SFEW** (base) | 40.3% | 0.0% | 13.0% | 76.4% | 27.4% | 58.9% | 14.3% |
| **AffectNet** (fine-tuned) | 63.5% | 47.7% | 55.7% | 87.2% | 61.6% | 54.7% | 71.6% |
| **JAFFE** (base) | 0.0% | 0.0% | 21.9% | 100.0% | 30.0% | 58.1% | 63.3% |

\* EmoRec-DB's source data doesn't include Disgust or Fear samples, so those cells show no
score for the base run.

### Key patterns

- **Happy is consistently the easiest class** — it scores highest across every single
  dataset (76–100%), because a smile is the most visually distinct facial expression and
  the most represented class in most training data.
- **Disgust and Fear are consistently the weakest classes** — often near 0% on harder
  datasets (RAF-DB, SFEW, JAFFE). These expressions are subtle, less represented in FER2013,
  and easily confused with Anger or Neutral.
- **Sad ↔ Neutral is the model's most common confusion** on the original FER2013 test set,
  which carries over to most external datasets — both expressions can look visually similar
  in a static image without motion context.
- **Domain gap matters more than dataset size.** CK+ (controlled studio lighting, posed
  expressions) generalizes very well (76%) despite zero fine-tuning, while SFEW and JAFFE
  (uncontrolled lighting, film stills, low-resolution scans) perform poorly (~39%) even
  though they're conceptually similar face-emotion datasets.

### Impact of fine-tuning

Freezing all layers except `conv6` and `fc`, then training briefly on each dataset's own
data, had very different effects depending on the dataset:

| Dataset | Base → Fine-tuned | Verdict |
|---|---|---|
| RAF-DB | 65.8% → **81.0%** (+15.2 pts) | Large, diverse dataset — fine-tuning helped a lot |
| AffectNet | 46.0%* → **64.6%** (+18.6 pts) | Same pattern — big gain from adapting to a large, varied dataset |
| EmoRec-DB | 67.6% → **~88%** (+20 pts) | Biggest jump, but dataset lacks 2 of the 7 classes, so treat this number cautiously |
| FEMI-DB | 77.7% → 76.8% (−0.9 pts) | No real improvement — the base model was already strong here, so there was little room to gain |

\* AffectNet's "before fine-tuning" number reflects the base model on AffectNet's YOLO-format
validation split, evaluated separately from the numbers in the summary table above.

**Takeaway:** fine-tuning is most valuable when (a) the target dataset has a meaningfully
different visual distribution than FER2013, and (b) there's enough data to adapt without
overfitting. When the base model already generalizes well to a dataset, fine-tuning offers
little to no benefit and can even slightly hurt performance on some classes (see FEMI-DB's
Neutral score dropping from 82% to 76.3%).

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

⚠️ Copyright © 2026 Fatimah Mohammed Al-Buraiki. All rights reserved.
This project is provided for research and demonstration purposes only. Unauthorized use, reproduction, modification, or redistribution is not permitted.

