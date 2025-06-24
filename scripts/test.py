import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from data_preprocess import extract_feature
from sklearn.preprocessing import LabelEncoder
from pydub import AudioSegment

# Define labels
emotion_labels = ['angry', 'calm', 'disgust', 'fearful','happy','neutral','sad','surprised']
le = LabelEncoder()
le.fit(emotion_labels)

def convert_to_wav(file_path):
    if file_path.endswith(".wav"):
        return file_path
    sound = AudioSegment.from_file(file_path)
    wav_path = os.path.splitext(file_path)[0] + ".wav"
    sound.export(wav_path, format="wav")
    return wav_path

def predict_emotion(file_path, model_path="model/ser.keras", return_all=False):
    wav_path = convert_to_wav(file_path)
    data, sr = librosa.load(wav_path)
    
    features = extract_feature(data, sr, mfcc=True, chroma=True, mel=True)
    features = features.reshape(1, -1, 1)

    model = load_model(model_path)
    predictions = model.predict(features)
    predicted_class = le.inverse_transform([np.argmax(predictions)])[0]

    if return_all:
        return predicted_class, predictions[0]
    
    
    print(f"\nEmotion Predicted: {predicted_class}")
    print("\nClass Probabilities:")
    for i, prob in enumerate(predictions[0]):
        print(f"{emotion_labels[i]:<10s}: {prob:.4f}")
    
    return predicted_class  
