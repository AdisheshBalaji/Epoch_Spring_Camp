import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Emotion label mapping
emotion_map = {0: "neutral", 1: "calm", 2: "happy", 3: "sad", 4: "angry", 5: "fearful", 6: "disgust", 7: "surprised"}

# Dataset class for mel spectrograms
class MelSpectrogramDataset(Dataset):
    def __init__(self, directory, augment=False):
        self.files = [f for f in os.listdir(directory) if f.endswith('.npy')]
        self.directory = directory
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        spectrogram = np.load(os.path.join(self.directory, file))
        
        # Normalize the spectrograms
        mean = np.mean(spectrogram)
        std = np.std(spectrogram) + 1e-9  
        spectrogram = (spectrogram - mean) / std

       
        padded = np.zeros((128, 128), dtype=np.float32)
        h, w = spectrogram.shape
        padded[:min(h, 128), :min(w, 128)] = spectrogram[:min(h, 128), :min(w, 128)]
        x = torch.tensor(padded).unsqueeze(0)  # Add channel dimension

        
        emotion_code = int(file.split("-")[2]) - 1  # e.g., "03" -> 2 (happy)
        y = torch.tensor(emotion_code)

        return x, y


class EmotionCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(EmotionCNN, self).__init__()
        # Simplified architecture
        self.conv_layers = nn.Sequential(
            # First block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # 128x128 -> 64x64
            
            # Second block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # 64x64 -> 32x32
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2), # 32x32 -> 16x16
            
            # Fourth block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2), # 16x16 -> 8x8
        )
        
        # Global Average Pooling + Classification
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), # Global average pooling
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x
    
    def features(self, x):
        return self.conv_layers(x)




# Training function
def train_model(data_path, epochs=20, batch_size=32, lr=0.001, weight_decay=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # # Create directories for saving results
    # os.makedirs("../plots", exist_ok=True)
    # os.makedirs("../models", exist_ok=True)
    
    # Load datasets
    train_set = MelSpectrogramDataset(os.path.join(data_path, "train"))
    val_set = MelSpectrogramDataset(os.path.join(data_path, "val"))
    
    # DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    
    # Initialize model
    model = EmotionCNN().to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler - simple step decay
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Initialize tracking variables
    best_val_acc = 0
    patience = 5
    patience_counter = 0
    train_accs = []
    val_accs = []
    
    # Training log
    with open("../accuracy_logs/cnn_accuracy_log.txt", "w") as f:
        f.write("Training started\n")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track stats
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        val_accs.append(val_acc)
        
        # Update learning rate
        scheduler.step()
        
        # Print stats
        log_line = f"Epoch {epoch+1}/{epochs}: Train Acc: {train_acc:.2f}% | "
        log_line += f"Val Acc: {val_acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.6f}\n"
        print(log_line.strip())
        
        with open("../accuracy_logs/cnn_accuracy_log.txt", "a") as f:
            f.write(log_line)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, "../models/best_cnn_model.pth")
            
            # Create confusion matrix for best model
            cm = confusion_matrix(all_targets, all_preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=[emotion_map[i] for i in range(8)],
                        yticklabels=[emotion_map[i] for i in range(8)])
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix - Epoch {epoch+1} (Val Acc: {val_acc:.2f}%)')
            plt.tight_layout()
            plt.savefig(f"../../plots/confusion_matrices/cnn_confusion_matrix_epoch_{epoch+1}.png")
            plt.close()
        # else:
        #     patience_counter += 1
        #     if patience_counter >= patience:
        #         print(f"Early stopping at epoch {epoch+1}")
        #         break
    
    # Plot training/validation accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig("../../plots/accuracy_plots/cnn_accuracy_plot.png")
    plt.close()
    
    print(f"Training complete. Best validation accuracy: {best_val_acc:.2f}%")
    return model, best_val_acc

# Entry point
if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    print("Starting model training...")
    model, best_val_acc = train_model("../../../data/split", epochs=30)