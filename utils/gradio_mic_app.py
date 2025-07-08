import os
import gradio as gr
import joblib
import numpy as np
import librosa

# Load model and encoder using correct relative paths
base_path = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(base_path, "audio_model", "emotion_audio_model.pkl")
encoder_path = os.path.join(base_path, "audio_model", "label_encoder.pkl")

model = joblib.load(model_path)
encoder = joblib.load(encoder_path)

# 🧠 Feature extraction function
def extract_features(file_path, sr=22050):
    try:
        audio, sr = librosa.load(file_path, sr=sr)
        if len(audio) == 0:
            raise ValueError("Audio file is empty or invalid.")
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return mfccs_processed
    except Exception as e:
        print(f"❌ Error extracting features: {e}")
        return None

# 🎯 Main prediction function
def predict_emotion_mic(audio_path):
    features = extract_features(audio_path)
    if features is None:
        return "❌ Could not process audio. Please try again with a valid voice input."
    try:
        features = features.reshape(1, -1)
        prediction = model.predict(features)
        emotion = encoder.inverse_transform(prediction)[0]
        return f"🎙️ Predicted Emotion: {emotion}"
    except Exception as e:
        return f"❌ Prediction error: {str(e)}"

# 🖥️ Gradio UI
gr.Interface(
    fn=predict_emotion_mic,
    inputs=gr.Audio(source="microphone", type="filepath", label="🎤 Record Your Voice"),
    outputs=gr.Text(label="Predicted Emotion"),
    title="🎧 Real-Time Emotion Detection",
    description="Speak into the microphone. This app will detect your emotion from voice.",
    allow_flagging="never"
).launch()
