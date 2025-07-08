import os
import numpy as np
import librosa

def load_audio(file_path, sample_rate=22050):
    y, sr = librosa.load(file_path, sr=sample_rate)
    return y, sr

def extract_features(y, sr, n_mfcc=40):
    try:
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs_scaled = np.mean(mfccs.T, axis=0)

        # Clean up bad values
        mfccs_scaled = np.nan_to_num(mfccs_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # Skip all-zero feature vectors
        if np.all(mfccs_scaled == 0):
            return None

        return mfccs_scaled
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def process_dataset(directory_path):
    features = []
    labels = []

    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)

                # Load and extract features
                y, sr = load_audio(file_path)
                mfccs = extract_features(y, sr)

                features.append(mfccs)

                # Extract label from folder name or filename
                label = os.path.basename(root)  # Assuming folder = label
                labels.append(label)

    return np.array(features), np.array(labels)
