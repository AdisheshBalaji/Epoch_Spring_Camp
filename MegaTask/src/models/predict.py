import torch
import torch.nn as nn
import librosa
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import sounddevice as sd
import librosa.display

# Emotion labels
emotion_labels = {
    0: "neutral", 1: "calm", 2: "happy", 3: "sad",
    4: "angry", 5: "fearful", 6: "disgust", 7: "surprised"
}

class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.15),  

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.15),  

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.15)  
        )
        self.fc = nn.Linear(64 * 16 * 16, 8)  # Adjust size based on input shape

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x






# Preprocessing function
def preprocess_audio(file_path, sr=16000, n_mels=128):
    audio, _ = librosa.load(file_path, sr=sr)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / mel_db.std()
    mel_db = np.expand_dims(mel_db, axis=0)  # shape: (1, mel, time)
    return torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0), audio, sr, mel_db[0]

# Main function
def predict_random_wav(wav_dir="/mnt/e/Epoch_Spring_Camp/MegaTask/data/ravdess", model_path="best_cnn_model.pth"):
    # Load model (saved using torch.save(model))
    model = EmotionCNN()



    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")), weights_only=True)
    model.eval()

    
    # Randomly pick a .wav file
    wav_files = [os.path.join(root, f)
                 for root, _, files in os.walk(wav_dir)
                 for f in files if f.endswith(".wav")]

    if not wav_files:
        print("No .wav files found.")
        return

    file_path = random.choice(wav_files)
    print(f"\nSelected file: {file_path}")

    # Preprocess and predict
    input_tensor, audio, sr, mel_to_plot = preprocess_audio(file_path)

    # Play the audio
    print("🔊 Playing audio...")
    sd.play(audio, sr)
    sd.wait()

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        pred_label = torch.argmax(output, dim=1).item()
        emotion = emotion_labels[pred_label]
        print(f"Predicted Emotion: {emotion}")

    # Plot mel spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_to_plot.numpy(), sr=sr, x_axis='time', y_axis='mel', cmap='magma')
    plt.title(f"Mel Spectrogram — Predicted: {emotion}")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    predict_random_wav()
