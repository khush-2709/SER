# scripts/web_infer.py

import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from scripts.data_preprocess import extract_feature
from sklearn.preprocessing import LabelEncoder
from pydub import AudioSegment

# Emotion labels (same order as training)
emotion_labels = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
le = LabelEncoder()
le.fit(emotion_labels)

def convert_to_wav(file_path):
    """
    Convert audio file to .wav if needed.
    Returns the path to the .wav file.
    """
    if file_path.endswith(".wav"):
        return file_path

    sound = AudioSegment.from_file(file_path)
    wav_path = os.path.splitext(file_path)[0] + ".wav"
    sound.export(wav_path, format="wav")
    return wav_path

def predict_emotion_web(file_path, model_path="model/ser.keras"):
    """
    Predict emotion from audio for web app use.
    Returns predicted label and class probabilities.
    """
    wav_path = convert_to_wav(file_path)
    data, sr = librosa.load(wav_path)
    features = extract_feature(data, sr, mfcc=True, chroma=True, mel=True)
    features = features.reshape(1, -1, 1)

    model = load_model(model_path)
    predictions = model.predict(features)
    predicted_label = le.inverse_transform([np.argmax(predictions)])

    return predicted_label[0], predictions[0].tolist()
