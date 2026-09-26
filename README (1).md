# CodeAlpha_EmotionRecognition

Emotion Recognition from Speech — CodeAlpha Machine Learning Internship (Task 2)

## 📌 Objective

Recognize human emotions (angry, calm, disgust, fearful, happy, neutral, sad, surprised) from speech audio using deep learning and speech signal processing techniques, as required by the CodeAlpha ML internship task list.

## 📊 Dataset

**RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song) — speech subset only.

- 24 actors (`Actor_01` to `Actor_24`)
- 1440 audio files total
- 8 emotion classes: neutral, calm, happy, sad, angry, fearful, disgust, surprised
- Note: "neutral" has half the samples (96) of the other classes (192 each), since RAVDESS only recorded neutral at normal intensity

## 🧠 Approach

Speech signal processing + deep learning, as suggested by the task:

1. **Feature extraction**: MFCC (Mel-Frequency Cepstral Coefficients), plus delta and delta-delta MFCCs, extracted per audio clip and padded/trimmed to a fixed length. Features are normalized (mean/std) before training.
2. **Modeling**: Two models were built and compared —
   - **LSTM** (baseline): a stacked LSTM network trained directly on raw MFCC sequences.
   - **CNN-LSTM (final model)**: Conv1D layers extract local spectral patterns from the MFCC/delta features, followed by an LSTM layer to model how those patterns evolve over time. This combines two of the architectures suggested by the task (CNN and LSTM) into a single model.
3. **Class imbalance**: handled with computed class weights during training, to account for "neutral" being underrepresented.
4. **Evaluation**: accuracy, precision, recall, F1-score (weighted), full classification report, and a confusion matrix.

## 📈 Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| LSTM (baseline) | 61.81% | 64.66% | 61.81% | 61.79% |
| **CNN-LSTM (final)** | **70.83%** | **71.48%** | **70.83%** | **70.76%** |

The CNN-LSTM model, combined with feature normalization, delta/delta-delta MFCCs, and class weighting, improved accuracy by roughly 9 percentage points over the plain LSTM baseline.

Per-class performance (CNN-LSTM, on held-out test set of 216 samples):

```
              precision    recall  f1-score   support

       angry       0.68      0.72      0.70        29
        calm       0.92      0.76      0.83        29
     disgust       0.71      0.83      0.76        29
     fearful       0.78      0.62      0.69        29
       happy       0.62      0.55      0.58        29
     neutral       0.71      0.86      0.77        14
         sad       0.60      0.62      0.61        29
   surprised       0.71      0.79      0.75        28
```

Confusion matrix and training/validation curves are saved to `results/confusion_matrix.png` and `results/training_curves.png` after running the evaluation script.

## 📁 Project Structure

```
CodeAlpha_EmotionRecognition/
├── data/
│   └── archive (5)/          # RAVDESS dataset (Actor_01 ... Actor_24)
├── features/
│   ├── X.npy                 # extracted feature sequences
│   ├── y.npy                 # emotion labels
│   ├── X_test.npy            # held-out test features
│   ├── y_test.npy            # held-out test labels
│   ├── mfcc_mean.npy         # normalization mean
│   └── mfcc_std.npy          # normalization std
├── models/
│   ├── emotion_model.h5      # trained CNN-LSTM model
│   └── label_encoder.pkl     # label encoder
├── results/
│   ├── training_history.pkl
│   ├── confusion_matrix.png
│   └── training_curves.png
├── scripts/
│   ├── extract_features.py
│   ├── train_model.py
│   └── evaluate_model.py
├── requirements.txt
└── README.md
```

## ▶️ How to Run

From the `scripts/` folder, run in order:

```bash
python extract_features.py
python train_model.py
python evaluate_model.py
```

If features have already been extracted (`X.npy` / `y.npy` exist), `extract_features.py` will skip re-extraction — delete those two files first if you want to re-run extraction (e.g. after changing feature settings).

## 🛠 Tech Stack

- Python
- TensorFlow / Keras
- Librosa (audio feature extraction)
- scikit-learn (metrics, preprocessing, train/test split)
- NumPy
- Matplotlib / Seaborn (visualizations)

## 🎓 Internship

This project was completed as part of the **CodeAlpha Machine Learning Internship**.

- 🔗 GitHub: [your-github-link-here]
- 🔗 LinkedIn post: [your-linkedin-post-link-here]

## 📞 Contact (CodeAlpha)

- Website: [www.codealpha.tech](https://www.codealpha.tech)
- Email: services@codealpha.tech
