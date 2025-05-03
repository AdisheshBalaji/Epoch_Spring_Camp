import os
import torch
from torch.utils.data import DataLoader
from train_cnn import SpectrogramDataset, EmotionCNN

def test_model(test_dir, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the test dataset
    test_dataset = SpectrogramDataset(test_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Load the trained model
    model = EmotionCNN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    test_acc = 100 * correct / total
    print(f"Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    test_dir = "/mnt/e/Epoch_Spring_Camp/MegaTask/data/split/test"
    model_path = "best_cnn_model.pth"
    test_model(test_dir, model_path)