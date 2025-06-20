# app.py

import streamlit as st
import os
from scripts.web import predict_emotion_web
from pydub import AudioSegment
import tempfile

st.title("🎙️ Speech Emotion Recognizer")
st.write("Upload a `.wav` or `.mp3` file to detect the emotion.")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        if uploaded_file.name.endswith(".mp3"):
            audio = AudioSegment.from_file(uploaded_file)
            tmp_path = tmp.name
            audio.export(tmp_path, format="wav")
        else:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

    st.write("✅ File uploaded and converted")

    with st.spinner("Analyzing emotion..."):
        predicted_label, probabilities = predict_emotion_web(tmp_path)

    st.success(f"🎧 Predicted Emotion: **{predicted_label}**")
    st.subheader("📊 Class Probabilities")
    for label, prob in zip(
        ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised'],
        probabilities
    ):
        st.write(f"{label.capitalize()}: {prob:.4f}")
