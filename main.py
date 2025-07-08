import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import unique_labels
import joblib

from utils.audio_preprocessing import process_dataset

# Load and process dataset
data_path = "data/audio"
X, y = process_dataset(data_path)

# Sanity check
print(f"✅ Total samples: {len(X)}")
print(f"✅ Feature shape: {X.shape}")
print(f"✅ Unique labels: {set(y)}")
print(f"✅ Any NaNs: {np.isnan(X).any()}, Any Infs: {np.isinf(X).any()}")

# Encode emotion labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Initialize and train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)

# Only evaluate labels that are present in test predictions
present_labels = unique_labels(y_test, y_pred)
present_names = encoder.inverse_transform(present_labels)

print("\n🎯 Classification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=present_labels,
    target_names=present_names,
    zero_division=0
))

# Save the model
os.makedirs("audio_model", exist_ok=True)
joblib.dump(clf, "audio_model/emotion_audio_model.pkl")
joblib.dump(encoder, "audio_model/label_encoder.pkl")

print("✅ Model and label encoder saved in 'audio_model/' folder.")
