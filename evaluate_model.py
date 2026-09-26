import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import load_model
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# -------------------------------
# Config
# -------------------------------

MODEL_PATH = "../models/emotion_model.h5"
ENCODER_PATH = "../models/label_encoder.pkl"
HISTORY_PATH = "../results/training_history.pkl"

X_TEST_PATH = "../features/X_test.npy"
Y_TEST_PATH = "../features/y_test.npy"


# -------------------------------
# Check everything exists
# -------------------------------

required_files = [MODEL_PATH, ENCODER_PATH, X_TEST_PATH, Y_TEST_PATH]

for f in required_files:
    if not os.path.exists(f):
        print("missing file:", f)
        print("make sure you've run train_model.py first")
        exit()


# -------------------------------
# Load model, encoder, and test data
# -------------------------------

model = load_model(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

X_test = np.load(X_TEST_PATH)
y_test = np.load(Y_TEST_PATH)   # one-hot encoded

print("model loaded successfully")
print("X_test shape:", X_test.shape)


# -------------------------------
# Make predictions
# -------------------------------

y_pred_probs = model.predict(X_test)

y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

emotion_labels = encoder.classes_


# -------------------------------
# Calculate evaluation metrics
# -------------------------------

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average="weighted")
rec = recall_score(y_true, y_pred, average="weighted")
f1 = f1_score(y_true, y_pred, average="weighted")

print("\nmodel: CNN-LSTM")
print(f"accuracy:  {acc:.2%}")
print(f"precision: {prec:.2%}")
print(f"recall:    {rec:.2%}")
print(f"f1 score:  {f1:.2%}")

print("\nclassification report:")
print(classification_report(y_true, y_pred, target_names=emotion_labels))


# -------------------------------
# Confusion matrix
# -------------------------------

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=emotion_labels,
    yticklabels=emotion_labels
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - CNN-LSTM Emotion Recognition")
plt.tight_layout()

os.makedirs("../results", exist_ok=True)
plt.savefig("../results/confusion_matrix.png")
plt.close()

print("\nsaved ../results/confusion_matrix.png")


# -------------------------------
# Training / validation curves
# -------------------------------

if os.path.exists(HISTORY_PATH):
    history = joblib.load(HISTORY_PATH)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history["accuracy"], label="Train Accuracy")
    plt.plot(history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()

    plt.tight_layout()
    plt.savefig("../results/training_curves.png")
    plt.close()

    print("saved ../results/training_curves.png")
else:
    print("training history not found, skipping training curves plot")