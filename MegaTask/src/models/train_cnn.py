import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt
import torchaudio.transforms as T




# Emotion label mapping
emotion_labels = {
    "neutral": 0, "calm": 1, "happy": 2, "sad": 3,
    "angry": 4, "fearful": 5, "disgust": 6, "surprised": 7
}

# Dataset class
class SpectrogramDataset(Dataset):
    def __init__(self, data_dir, augment = False):
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith(".npy")]
        self.augment = augment


        self.freq_mask = T.FrequencyMasking(freq_mask_param=15)
        self.time_mask = T.TimeMasking(time_mask_param=20)


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        x = np.load(os.path.join(self.data_dir, f))

        # Pad or truncate to a fixed size (e.g., 128x128)
        target_shape = (128, 128)  # (mel_bins, time_steps)
        padded_x = np.zeros(target_shape, dtype=np.float32)
        x_shape = x.shape

        # Truncate if larger, pad if smaller
        padded_x[:x_shape[0], :min(x_shape[1], target_shape[1])] = x[:, :target_shape[1]]

        x = torch.tensor(padded_x).unsqueeze(0).float()  # (1, mel_bins, time_steps)

        # Extract emotion from filename
        emotion_id = int(f.split("-")[2])  # 03-01-**01**-01-01-01-01.wav
        emotion = emotion_id

        y = torch.tensor(emotion - 1)  # Make it 0-indexed (subtract 1)
        return x, y







# Simple CNN
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),  

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),  

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2)  
        )
        self.fc = nn.Linear(32 * 16 * 16, 8)  # Adjust size based on input shape

    def forward(self, x):
        x = self.conv_layers(x)
        # x = self.gap(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x
    

log_file = "training_log.txt"


# Training function
def train_model(data_dir, epochs=20, batch_size=32, lr=0.0005):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_dataset = SpectrogramDataset(train_dir, augment = True)
    val_dataset = SpectrogramDataset(val_dir, augment = False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = EmotionCNN().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val_acc = 0
    patience = 5
    patience_counter = 0

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        train_acc = 100 * correct / total

        


        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                _, predicted = torch.max(outputs.data, 1)
                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()

        val_acc = 100 * val_correct / val_total


        train_accs = []
        val_accs = []

        train_accs.append(train_acc)
        val_accs.append(val_acc)

        log_line = f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%\n"
        with open("accuracy_log.txt", "a") as f:
            f.write(log_line)


        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_cnn_model.pth")
        

    print("Training Finished.")


    train_accs = np.array(train_accs)
    val_accs = np.array(val_accs)

    xvals = np.arange(1, 51)
    plt.plot(xvals, train_accs, color = "blue")
    plt.plot(xvals, val_accs, color = "yellow")
    plt.savefig("../plots/Train_vs_Val_accs.png")
    plt.show()

if __name__ == "__main__":
    train_model("/mnt/e/Epoch_Spring_Camp/MegaTask/data/split", epochs=50)
