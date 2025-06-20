

import numpy as np
import librosa

def extract_feature(data, sr, mfcc=True, chroma=True, mel=True):
    """
    Extract features from audio files into a numpy array.
    """
    result = np.array([])

    if chroma or mel:
        stft = np.abs(librosa.stft(y=data))

    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))

    if chroma:
        chroma_feat = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
        result = np.hstack((result, chroma_feat))

    if mel:
        mel_feat = np.mean(librosa.feature.melspectrogram(y=data, sr=sr).T, axis=0)
        result = np.hstack((result, mel_feat))

    return result

def preprocess_audio_file(file_path):
    """
    Preprocess a .wav file to match model input requirements.

    Parameters
    ----------
    file_path : str - path to .wav file

    Returns
    -------
    np.ndarray - reshaped 3D input ready for model prediction
    """
    data, sr = librosa.load(file_path)

    # Extract features
    features = extract_feature(data, sr, mfcc=True, chroma=True, mel=True)

    # Reshape to match CNN model input shape
    features = features.reshape(1, -1, 1)

    return features
