import torch
from torch.utils.data import Dataset
import os
import numpy as np

# Map emotion codes to labels
emotion_labels = {
    "01": 0, "02": 1, "03": 2, "04": 3,
    "05": 4, "06": 5, "07": 6, "08": 7
}


def get_emotion_label(file_name):
    emotion_code = file_name.split("-")[2]
    return emotion_labels[emotion_code]

class AudioEmotionDataset(Dataset):
    def __init__(self, data_dir="data/processed"):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith(".npy")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        path = os.path.join(self.data_dir, file_name)
        spec = np.load(path)

        # (1, mel_bins, time_steps)
        spec = np.expand_dims(spec, axis=0)

        label = get_emotion_label(file_name)

        return torch.tensor(spec, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
