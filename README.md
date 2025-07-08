# 🎧 Emotion Detection from Speech Audio using Machine Learning

This project detects emotions from `.wav` audio files using machine learning techniques. It uses the [RAVDESS dataset](https://zenodo.org/record/1188976) to train a model that classifies speech into 8 emotional states.

---

## 📌 Features

- ✅ Trained on 1,440 labeled `.wav` files from RAVDESS  
- ✅ Supports 8 emotions: `neutral`, `calm`, `happy`, `sad`, `angry`, `fearful`, `disgust`, `surprised`  
- ✅ Uses MFCC features extracted via `librosa`  
- ✅ Trained with `RandomForestClassifier`  
- ✅ Saves model and label encoder for future use  
- ✅ Optional Gradio UI for audio testing  

---

## 📁 Project Structure

```
emotion-detector/
├── main.py                      # Main script to train and evaluate the model
├── organize_ravdess.py          # Script to organize RAVDESS data by emotion
├── utils/
│   └── audio_preprocessing.py   # Feature extraction code
├── data/
│   └── audio/                   # Sorted folders with .wav files per emotion
├── audio_model/                 # Trained model and label encoder
├── README.md
└── requirements.txt
```

---

## 🧠 Emotions Covered

| Code | Emotion     |
|------|-------------|
| 01   | Neutral     |
| 02   | Calm        |
| 03   | Happy       |
| 04   | Sad         |
| 05   | Angry       |
| 06   | Fearful     |
| 07   | Disgust     |
| 08   | Surprised   |

---

## 🚀 Getting Started


### 1. Set Up Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📥 Dataset Setup

1. Download the RAVDESS dataset from [Zenodo](https://zenodo.org/record/1188976)  
2. Extract it to:  
   ```
   emotion-detector/Audio_Speech_Actors_01-24/
   ```

3. Organize audio by emotion:
```bash
python organize_ravdess.py
```

---

## 🏋️‍♂️ Training the Model

```bash
python main.py
```

Model and label encoder will be saved in the `audio_model/` folder.

---

## 📈 Sample Output (Classification Report)

```
              precision    recall  f1-score   support
       angry       0.56      0.53      0.54
        calm       0.54      0.84      0.66
     disgust       0.56      0.58      0.57
     fearful       0.77      0.77      0.77
       happy       0.52      0.38      0.44
     neutral       0.60      0.32      0.41
         sad       0.64      0.47      0.55
   surprised       0.44      0.54      0.48

Accuracy: ~57%
```

---

## 👨‍💻 Author

**Danish Khan**  

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
