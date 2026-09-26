import os
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from keras.models import Sequential
from keras.layers import (
    Conv1D, BatchNormalization, MaxPooling1D,
    LSTM, Dense, Dropout
)
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping


# -------------------------------
# Config
# -------------------------------

X_PATH = "../features/X.npy"
Y_PATH = "../features/y.npy"

MODEL_OUT = "../models/emotion_model.h5"
ENCODER_OUT = "../models/label_encoder.pkl"


# -------------------------------
# Load extracted features
# -------------------------------

if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
    print("features not found. run extract_features.py first")
    exit()

X = np.load(X_PATH)
y = np.load(Y_PATH)

print("X shape:", X.shape)   # (num_files, 130, 120) -> 40 MFCC * 3 (raw, delta, delta2)
print("y shape:", y.shape)   # (num_files,)

# Warn if a trained model already exists (this run will overwrite it)
if os.path.exists(MODEL_OUT):
    print("\nnote: a trained model already exists at", MODEL_OUT)
    print("continuing will overwrite it\n")


# -------------------------------
# Encode labels (text -> numbers -> one-hot)
# -------------------------------

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

print("classes:", list(encoder.classes_))


# -------------------------------
# Train / validation / test split (3-way, no leakage)
# -------------------------------
# 70% train, 15% val (used for early stopping), 15% test (untouched until evaluate_model.py)

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y_categorical,
    test_size=0.3,
    random_state=42,
    stratify=y_categorical
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

print("training samples:", X_train.shape[0])
print("validation samples:", X_val.shape[0])
print("testing samples:", X_test.shape[0])

# Save the untouched test split so evaluate_model.py uses the exact same data
os.makedirs("../features", exist_ok=True)
np.save("../features/X_test.npy", X_test)
np.save("../features/y_test.npy", y_test)
print("saved ../features/X_test.npy and y_test.npy for evaluation")


# -------------------------------
# Class weights (handles RAVDESS having half as many "neutral" samples)
# -------------------------------

y_train_labels = np.argmax(y_train, axis=1)
class_weights_arr = compute_class_weight(
    "balanced",
    classes=np.unique(y_train_labels),
    y=y_train_labels
)
class_weights = dict(enumerate(class_weights_arr))
print("class weights:", class_weights)


# -------------------------------
# Build CNN-LSTM hybrid model
# -------------------------------
# CNN layers pick up local spectral patterns in the MFCC/delta features,
# LSTM layer models how those patterns evolve over time. This satisfies
# the task's suggested approach (CNN, RNN, or LSTM) by combining CNN + LSTM.

num_classes = y_categorical.shape[1]
timesteps = X.shape[1]     # 130
n_features = X.shape[2]    # 120 (40 MFCC * 3)

model = Sequential([
    Conv1D(64, kernel_size=5, activation="relu", input_shape=(timesteps, n_features)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Conv1D(128, kernel_size=5, activation="relu"),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    LSTM(64, return_sequences=False),
    Dropout(0.3),

    Dense(32, activation="relu"),
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# -------------------------------
# Train
# -------------------------------

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[early_stop]
)

print("model trained successfully")


# -------------------------------
# Save model, encoder, and training history
# -------------------------------

os.makedirs("../models", exist_ok=True)
os.makedirs("../results", exist_ok=True)

model.save(MODEL_OUT)
joblib.dump(encoder, ENCODER_OUT)
joblib.dump(history.history, "../results/training_history.pkl")

print("saved", MODEL_OUT)
print("saved", ENCODER_OUT)
print("saved ../results/training_history.pkl")