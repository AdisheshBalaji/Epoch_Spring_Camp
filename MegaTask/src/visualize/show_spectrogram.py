import os
import numpy as np
import matplotlib.pyplot as plt


plot_no = 0
def show_spectrogram(npy_path):
    mel_db = np.load(npy_path)
    plot_no += 1

    plt.figure(figsize=(10, 4))
    plt.imshow(mel_db, aspect='auto', origin='lower', cmap='magma')
    plt.title("Mel Spectrogram")
    plt.xlabel("Time")
    plt.ylabel("Mel Frequency")
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(f"../plots/MelSpectrograms_{plot_no}")
    plt.show()

import random

if __name__ == "__main__":
    directory = "../../data/processed"
    files = [f for f in os.listdir(directory) if f.endswith(".npy")]
    sample_files = random.sample(files, 4)

    for f in sample_files:
        print(f"Showing {f}")
        show_spectrogram(os.path.join(directory, f))
