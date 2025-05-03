import librosa
import numpy as np
import os


# "E:\Epoch_Spring_Camp\MegaTask\data\ravdess"


def preprocess_audio(input_dir="/mnt/e/Epoch_Spring_Camp/MegaTask/data/ravdess", output_dir="/mnt/e/Epoch_Spring_Camp/MegaTask/data/processed", sr=16000, n_mels=128):
    os.makedirs(output_dir, exist_ok=True)

    # Recursively collect all .wav files
    wav_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.endswith(".wav"):
                wav_files.append(os.path.join(root, f))

    print(f"Found {len(wav_files)} .wav files.")

    for i, path in enumerate(wav_files):
        audio, _ = librosa.load(path, sr=sr)

        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        #converting the decibels to 0 mean and unit variance
        mel_db = (mel_db - mel_db.mean()) / mel_db.std()

        # Save using filename only (you can also include actor folder if needed)
        file_name = os.path.basename(path).replace(".wav", ".npy")
        out_path = os.path.join(output_dir, file_name)
        np.save(out_path, mel_db)

        if i % 100 == 0:
            print(f"Processed {i+1}/{len(wav_files)}: {file_name}")


if __name__ == "__main__":
    try:
        preprocess_audio()
    except Exception as e:
        print("Error:", e)
