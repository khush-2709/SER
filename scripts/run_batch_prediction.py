import os
import pandas as pd
from test import predict_emotion  # assumes this handles file_path + model_path

import sys
import io

# Force stdout to use UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# Your settings
folder_path = 'test_audio/'          # Folder with .wav or .mp3 files
model_path = 'model/ser.keras'

# Output list
results = []

for filename in os.listdir(folder_path):
    if filename.endswith('.wav') or filename.endswith('.mp3'):
        file_path = os.path.join(folder_path, filename)
        print(f"\nProcessing: {filename}")
        
        # Get prediction from your existing test.py
        predicted_label, probabilities = predict_emotion(file_path, model_path, return_all=True)
        
        # Save to results
        result_entry = {
            "filename": filename,
            "predicted_emotion": predicted_label,
            **{f"{emotion}_prob": round(prob, 4) for emotion, prob in zip(
                ['angry', 'calm', 'disgust', 'fearful','happy','neutral','sad','surprised'], probabilities)}
        }
        results.append(result_entry)

# Save to CSV
df = pd.DataFrame(results)
df.to_csv("batch_predictions.csv", index=False)
print("\nAll results saved to batch_predictions.csv")
