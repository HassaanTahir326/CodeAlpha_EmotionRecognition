import os
import numpy as np
import librosa

# -------------------------------
# Config
# -------------------------------

DATA_PATH = "../data/archive (5)"      # folder containing Actor_01, Actor_02, ... Actor_24
OUTPUT_X = "../features/X.npy"         # extracted MFCC sequences
OUTPUT_Y = "../features/y.npy"         # emotion labels
OUTPUT_MEAN = "../features/mfcc_mean.npy"
OUTPUT_STD = "../features/mfcc_std.npy"

N_MFCC = 40          # number of MFCC coefficients per frame
MAX_LEN = 130        # fixed number of time-frames per clip (pad/trim to this)

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


# -------------------------------
# Extract MFCC + delta + delta-delta sequence for one file
# -------------------------------

def extract_mfcc_sequence(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # stack -> shape becomes (N_MFCC*3, time_frames), then transpose
    features = np.vstack([mfcc, delta, delta2]).T  # (time_frames, N_MFCC*3)

    # Pad with zeros if too short, trim if too long
    if features.shape[0] < MAX_LEN:
        pad_width = MAX_LEN - features.shape[0]
        features = np.pad(features, ((0, pad_width), (0, 0)), mode="constant")
    else:
        features = features[:MAX_LEN, :]

    return features


# -------------------------------
# Walk dataset and build X, y
# -------------------------------

if not os.path.isdir(DATA_PATH):
    print("dataset folder not found at", DATA_PATH)
    print("make sure you've unzipped the archive and set DATA_PATH correctly")
    exit()

# Skip extraction if it was already done before
if os.path.exists(OUTPUT_X) and os.path.exists(OUTPUT_Y):
    print("features already extracted at", OUTPUT_X, "and", OUTPUT_Y)
    print("delete these files first if you want to re-run extraction")
    exit()

X = []
y_labels = []
skipped = 0

for actor_folder in sorted(os.listdir(DATA_PATH)):
    actor_path = os.path.join(DATA_PATH, actor_folder)

    if not os.path.isdir(actor_path) or not actor_folder.startswith("Actor_"):
        continue

    for filename in sorted(os.listdir(actor_path)):
        if not filename.endswith(".wav"):
            continue

        parts = filename.replace(".wav", "").split("-")

        if len(parts) != 7:
            skipped += 1
            continue

        emotion_code = parts[2]
        emotion = EMOTION_MAP.get(emotion_code)

        if emotion is None:
            skipped += 1
            continue

        file_path = os.path.join(actor_path, filename)

        try:
            feat_seq = extract_mfcc_sequence(file_path)
        except Exception as e:
            print("failed on", file_path, "-", e)
            skipped += 1
            continue

        X.append(feat_seq)
        y_labels.append(emotion)

    print("processed", actor_folder)

print("\ntotal files processed:", len(X))
print("skipped:", skipped)


# -------------------------------
# Normalize features (fit on the full extracted set, saved for reuse)
# -------------------------------

X = np.array(X)              # shape: (num_files, MAX_LEN, N_MFCC*3)
y_labels = np.array(y_labels)

mean = X.mean(axis=(0, 1), keepdims=True)
std = X.std(axis=(0, 1), keepdims=True) + 1e-8
X = (X - mean) / std


# -------------------------------
# Save as numpy arrays
# -------------------------------

os.makedirs(os.path.dirname(OUTPUT_X), exist_ok=True)

np.save(OUTPUT_X, X)
np.save(OUTPUT_Y, y_labels)
np.save(OUTPUT_MEAN, mean)
np.save(OUTPUT_STD, std)

print("\nsaved", OUTPUT_X, "with shape", X.shape)
print("saved", OUTPUT_Y, "with shape", y_labels.shape)
print("saved", OUTPUT_MEAN, "and", OUTPUT_STD, "for reuse on new audio")

print("\nemotion counts:")
unique, counts = np.unique(y_labels, return_counts=True)
for emotion, count in zip(unique, counts):
    print(emotion, ":", count)