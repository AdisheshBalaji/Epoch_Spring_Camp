import os
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import numpy as np

# === CONFIG ===
DATA_PATH = "../../data/ravdess"
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}
NUM_CLASSES = len(EMOTION_MAP)
SAMPLE_RATE = 16000
N_MFCC = 40
MAX_LEN = 300  # Truncate/pad to 300 frames

# === DATASET ===
class RAVDESSDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file = self.file_paths[idx]
        label = self.labels[idx]
        mfcc = self.extract_features(file)
        return mfcc, label

    def extract_features(self, file_path):
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        mfcc = torch.tensor(mfcc.T[:MAX_LEN], dtype=torch.float32)  # shape: (time, features)
        if mfcc.shape[0] < MAX_LEN:
            pad_len = MAX_LEN - mfcc.shape[0]
            mfcc = torch.cat([mfcc, torch.zeros(pad_len, N_MFCC)], dim=0)
        return mfcc

# === LOAD DATA ===
def parse_emotion(filename):
    return int(filename.split("-")[2]) - 1  # 0-based indexing

all_files = [os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH) if f.endswith(".wav")]
labels = [parse_emotion(os.path.basename(f)) for f in all_files]

train_files, val_files, train_labels, val_labels = train_test_split(all_files, labels, test_size=0.2, random_state=42)

train_dataset = RAVDESSDataset(train_files, train_labels)
val_dataset = RAVDESSDataset(val_files, val_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# === MODEL ===
class EmotionRNN(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=128, output_dim=NUM_CLASSES):
        super(EmotionRNN, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, (hn, _) = self.rnn(x)
        return self.fc(hn[-1])  # last hidden state

# === TRAINING ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionRNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_epoch(model, loader):
    model.train()
    total_loss, total_correct = 0, 0
    for x, y in loader:
        x, y = x.to(device), torch.tensor(y).to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_correct += (outputs.argmax(1) == y).sum().item()
    return total_loss / len(loader), total_correct / len(loader.dataset)

def eval_model(model, loader):
    model.eval()
    total_correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), torch.tensor(y).to(device)
            outputs = model(x)
            total_correct += (outputs.argmax(1) == y).sum().item()
    return total_correct / len(loader.dataset)

# === RUN TRAINING ===
EPOCHS = 20
for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader)
    val_acc = eval_model(model, val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
