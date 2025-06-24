# Speech Emotion Recognition System

> Accurately classify emotions in speech and song using deep learning.

---

## Overview

This project uses a **1D Convolutional Neural Network (CNN)** trained on the **RAVDESS** dataset (audio_speech and audio_song) to classify human emotions from audio files. It supports both **batch predictions** and a **real-time Streamlit web interface**.

---

## Features

- ✅ **Batch Prediction**: Process multiple `.wav` files at once and export results to CSV.
- ✅ **Web Interface**: Upload an audio file and get real-time emotion predictions.
- ✅ **Comprehensive Output**: Predicted label and probability distribution across all emotion classes.
- ✅ **Robust Preprocessing**: Extracts MFCCs, chroma features, and mel spectrograms.

---

## Usage

### 1. Batch Prediction

1. Place your audio files in the `test_audio/` directory.
2. Run:
   ```bash
   python run_batch_prediction.py


3. The output will be saved as `batch_predictions.csv` containing:

   * Filename
   * Predicted emotion
   * Probability distribution for all 8 classes

### 2. Real-Time Web App

1. Launch the web app with:

   ```bash
   streamlit run app.py
   ```
2. Upload a `.wav` audio file.
3. The app will display:

   * Predicted emotion
   * Probability scores for each emotion class

---

## Model Performance

| Metric             | Value     |
| ------------------ | --------- |
| Accuracy (overall) | **81%**   |
| Macro F1 Score     | **0.794** |



### Per-Class Accuracy

| Emotion   | Accuracy |
| --------- | -------- |
| Angry     | 0.84     |
| Calm      | 0.92     |
| Disgust   | 0.61     |
| Fearful   | 0.82     |
| Happy     | 0.79     |
| Neutral   | 0.75     |
| Sad       | 0.79     |
| Surprised | 0.63     |

---

## Classification Report after dropping 'disgust'

```
              precision    recall  f1-score   support

       angry       0.96      0.89      0.92        74
        calm       0.80      0.91      0.85        78
     fearful       0.82      0.76      0.79        71
       happy       0.86      0.76      0.81        85
     neutral       0.69      0.71      0.70        28
         sad       0.81      0.79      0.80        78
   surprised       0.74      0.87      0.80        46

    accuracy                           0.82       460
   macro avg       0.81      0.82      0.81       460
weighted avg       0.83      0.82      0.82       460
```
## Model Performance after dropping 'disgust'

| Metric             | Value     | Threshold |
| ------------------ | --------- | --------- |
| Accuracy (overall) | **82.17%**   | > 80% ✅   |
| Macro F1 Score     | **0.81** | > 0.80 ✅  |
---

## Training Graphs

You can visualize the model's learning progress with these plots:



### Training Accuracy

![Training Accuracy](https://github.com/khush-2709/SER/blob/main/plots/model%20acc.png)

### Training Loss

![Training Loss](https://github.com/khush-2709/SER/blob/main/plots/model%20loss.png)




---

## Confusion Matrix


![Confusion Matrix](https://github.com/khush-2709/SER/blob/main/plots/cm_ser.png)

---

## Technical Details

* **Model Architecture**: 1D CNN with 4 convolutional layers
* **Regularization**: L1 and L2 regularizers to prevent overfitting
* **Feature Extraction**:

  * 180-frame MFCCs
  * Chroma features
  * Mel spectrograms
    
## Preprocessing Details

The preprocessing pipeline includes:

- **Sample Rate Normalization**: Resampled all audio to a consistent rate using `librosa`.
- **Amplitude Normalization**: Scaled waveforms to a standard floating-point range (`[-1, 1]`).
- **Data Augmentation (Training Only)**:
  - **White Noise Addition**: Simulated background noise for better generalization.
  - **Time Shifting**: Slight shifts introduced to simulate speech variation.
---

## Dataset

**RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)

* 24 actors (12 male, 12 female)
* 8 emotion classes
* 2 intensity levels (normal and strong)
* 2 different spoken phrases
* Studio-quality `.wav` files at 48kHz

---

## Limitations

* Optimized for studio-recorded, high-quality audio
* Limited to **English** speech
* Performance may degrade with:

  * Noisy or real-world audio
  * Accents or speech variations not present in RAVDESS

---



## Author

**Khushi Agarwal** 
*(Final-year B.Tech student, Engineering Physics @ IIT Roorkee)*

---

## Contributions

Pull requests, issues, and feature suggestions are welcome!
Feel free to fork the repo and make it better 

---


