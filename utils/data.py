import os
import shutil

# Your paths
source_dir = "/Users/danishkhan/Development/PersonalProjects/python/CREMA-D/AudioWAV"
target_dir = "/Users/danishkhan/Development/PersonalProjects/emotion-detector/data/audio"

# Mapping 3-letter codes to your 2-digit emotion_map
code_map = {
    "NEU": "neutral",
    "CAL": "calm",
    "HAP": "happy",
    "SAD": "sad",
    "ANG": "angry",
    "FEA": "fearful",
    "DIS": "disgust",
    "SUR": "surprised"
}

# Create folders
for emotion in code_map.values():
    os.makedirs(os.path.join(target_dir, emotion), exist_ok=True)

# Move files
moved = 0
for file in os.listdir(source_dir):
    if file.endswith(".wav"):
        parts = file.split("_")
        if len(parts) >= 3:
            emo_code = parts[2]
            emotion = code_map.get(emo_code)
            if emotion:
                src = os.path.join(source_dir, file)
                dst = os.path.join(target_dir, emotion, file)
                shutil.copy2(src, dst)
                moved += 1

print(f"✅ Finished. Total files moved: {moved}")
