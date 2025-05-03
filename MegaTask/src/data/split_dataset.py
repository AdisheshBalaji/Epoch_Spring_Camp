import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

def split_dataset(data_dir, output_dir, train_ratio=0.7, val_ratio=0.15):
    # Ensure output directories exist
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get all .npy files
    files = [f for f in os.listdir(data_dir) if f.endswith(".npy")]

    # Split into train, validation, and test sets
    train_files, temp_files = train_test_split(files, test_size=(1 - train_ratio), random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=(1 - val_ratio / (1 - train_ratio)), random_state=42)

    # Move files to respective directories
    for f in train_files:
        shutil.copy(os.path.join(data_dir, f), os.path.join(train_dir, f))
    for f in val_files:
        shutil.copy(os.path.join(data_dir, f), os.path.join(val_dir, f))
    for f in test_files:
        shutil.copy(os.path.join(data_dir, f), os.path.join(test_dir, f))

    print(f"Dataset split completed:")
    print(f"Train files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    print(f"Test files: {len(test_files)}")

if __name__ == "__main__":
    data_dir = "/mnt/e/Epoch_Spring_Camp/MegaTask/data/processed"
    output_dir = "/mnt/e/Epoch_Spring_Camp/MegaTask/data/split"
    split_dataset(data_dir, output_dir)