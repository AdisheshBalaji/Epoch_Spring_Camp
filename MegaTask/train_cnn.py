import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchaudio.transforms as T
import matplotlib.pyplot as plt

# Emotion label mapping
emotion_map = {1: "neutral", 2: "calm", 3: "happy", 4: "sad", 5: "angry", 6: "fearful", 7: "disgust", 8: "surprised"}

# Dataset class for mel spectrograms
class MelSpectrogramDataset(Dataset):
    def __init__(self, directory, augment=False):
        self.files = [f for f in os.listdir(directory) if f.endswith('.npy')]
        self.directory = directory
        self.augment = augment


        self.freq_mask = T.FrequencyMasking(freq_mask_param=15)
        self.time_mask = T.TimeMasking(time_mask_param=35)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        spectrogram = np.load(os.path.join(self.directory, file))

        eps = 1e-6
        spec_mean = np.mean(spectrogram)
        spec_std = np.std(spectrogram) + eps
        spectrogram = (spectrogram - spec_mean) / spec_std
        
        # Clip outliers for better stability
        spectrogram = np.clip(spectrogram, -3, 3)



        # Pad/truncate to 128x128
        padded = np.zeros((128, 128), dtype=np.float32)
        h, w = spectrogram.shape
        padded[:min(h, 128), :min(w, 128)] = spectrogram[:128, :128]
        x = torch.tensor(padded).unsqueeze(0)

        if self.augment:
            # Apply augmentation with 70% probability
            if np.random.random() < 0.7:
                x = self.freq_mask(x)
            if np.random.random() < 0.7:
                x = self.time_mask(x)
            
            # Random gain adjustment
            if np.random.random() < 0.5:
                gain = 0.9 + 0.2 * np.random.random()  # 0.9 to 1.1
                x = x * gain

        # Label: zero-indexed emotion code
        emotion_code = int(file.split("-")[2]) - 1  # e.g., "03" -> 2 (happy)
        y = torch.tensor(emotion_code)

        return x, y

# Simple CNN model
# Adding BatchNorm and dropout 0.2, 0.2, 0.3
# Dropout reduced, extra layers added, increased lr = 0.003
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(EmotionCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # [B, 16, 128, 128]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 16, 64, 64]
            nn.Dropout(0.1),


            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # [B, 32, 64, 64]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 32, 32, 32]
            nn.Dropout(0.1),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # [B, 64, 32, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 64, 16, 16]
            nn.Dropout(0.3),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),

        )
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # [B, 64, 1, 1]
            nn.Flatten(),                  # [B, 64]
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc(self.net(x))

patience = 5
best_val_acc = 0
patience_counter = 0


# Training function
def train_model(data_path, epochs=30, batch_size=32, lr=0.003):




    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = MelSpectrogramDataset(os.path.join(data_path, "train"), augment = True)
    val_set = MelSpectrogramDataset(os.path.join(data_path, "val"), augment = False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model = EmotionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    best_val_acc = 0
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        model.train()
        total, correct = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        train_acc = 100 * correct / total

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_correct += (out.argmax(1) == y).sum().item()
                val_total += y.size(0)
        val_acc = 100 * val_correct / val_total
        scheduler.step(1 - val_acc / 100)

        train_accs.append(train_acc)
        val_accs.append(val_acc)
        log_line = f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%\n"
        with open("accuracy_log.txt", "a") as f:
            f.write(log_line)


        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early Stopping\n")
                break

    # Plot accuracy
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../plots/accuracy_plot.png")
    plt.show()




# Assuming model is loaded and val_loader is defined

# Entry point
if __name__ == "__main__":
    train_model("../../data/split", epochs=30)
