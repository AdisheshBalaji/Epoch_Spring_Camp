import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import re
import string
from collections import Counter

# Emotion label mapping
emotion_map = {0: "neutral", 1: "calm", 2: "happy", 3: "sad", 4: "angry", 5: "fearful", 6: "disgust", 7: "surprised"}

# Since we don't have actual transcripts, we'll simulate them based on emotions
# These are template sentences for each emotion that will be used for training
emotion_templates = {
    0: ["This is neutral", "I feel nothing special", "Just an ordinary day", 
        "Everything is normal", "Nothing unusual happening", "I'm feeling neutral today"],
    1: ["I am feeling calm", "Everything is peaceful", "I feel relaxed", 
        "I'm at ease", "There's tranquility in the air", "I feel serene and composed"],
    2: ["I am very happy", "This is such a joyful moment", "I'm so excited", 
        "What a wonderful day", "I'm thrilled about this", "I feel so cheerful"],
    3: ["I'm feeling sad today", "This is depressing", "I feel down", 
        "I'm in a gloomy mood", "I'm unhappy about this", "This is making me melancholic"],
    4: ["I am very angry", "This is infuriating", "I'm outraged by this", 
        "I'm furious right now", "This is making me mad", "I can't control my anger"],
    5: ["I'm feeling scared", "This is frightening", "I'm afraid of this", 
        "I'm terrified", "This is making me anxious", "I feel so fearful"],
    6: ["That's disgusting", "This makes me feel sick", "I'm revolted by this", 
        "How repulsive", "This is nauseating", "I feel so grossed out"],
    7: ["I'm so surprised", "This is unexpected", "I'm shocked", 
        "I didn't see that coming", "What a surprise", "I'm amazed by this"]
}

# For text preprocessing
def preprocess_text(text):
    """Basic preprocessing for text data"""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Simple vocabulary builder for text data
class Vocabulary:
    def __init__(self, min_freq=1):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.word_counts = Counter()
        self.min_freq = min_freq

    def build_vocabulary(self, texts):
        # Count words
        for text in texts:
            self.word_counts.update(text.split())
        
        # Add words that appear at least min_freq times
        idx = len(self.word2idx)
        for word, count in self.word_counts.items():
            if count >= self.min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
        
        print(f"Vocabulary size: {len(self.word2idx)}")
        return self

    def text_to_indices(self, text, max_length=20):
        # Convert text to indices
        words = text.split()
        indices = [self.word2idx.get(word, 1) for word in words]  # 1 is <UNK>
        
        # Truncate or pad
        if len(indices) > max_length:
            indices = indices[:max_length]
        else:
            indices += [0] * (max_length - len(indices))  # 0 is <PAD>
        
        return indices

# Dataset class for simulated text data
class TextEmotionDataset(Dataset):
    def __init__(self, file_list, vocabulary, max_length=20, use_templates=True):
        self.file_list = file_list
        self.vocabulary = vocabulary
        self.max_length = max_length
        self.use_templates = use_templates
        
        # Extract emotion labels from filenames
        self.emotion_labels = []
        for file in file_list:
            # Extract emotion code from filename (e.g., "03" -> 2)
            emotion_code = int(file.split("-")[2]) - 1
            self.emotion_labels.append(emotion_code)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Get emotion label
        emotion = self.emotion_labels[idx]
        
        # Generate or get text based on emotion
        if self.use_templates:
            # Randomly select a template for this emotion
            text = np.random.choice(emotion_templates[emotion])
        else:
            # In a real scenario, you would load the actual transcript here
            # For now, we'll just use a template
            text = np.random.choice(emotion_templates[emotion])
        
        # Preprocess text
        text = preprocess_text(text)
        
        # Convert to indices
        indices = self.vocabulary.text_to_indices(text, self.max_length)
        
        return torch.tensor(indices), torch.tensor(emotion)

# Simple GRU model for text classification
class TextGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=1, dropout=0.5):
        super(TextGRU, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layers, 
                          batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text shape: [batch_size, seq_len]
        embedded = self.dropout(self.embedding(text))
        # embedded shape: [batch_size, seq_len, embedding_dim]
        
        output, hidden = self.gru(embedded)
        # output shape: [batch_size, seq_len, hidden_dim]
        # hidden shape: [n_layers, batch_size, hidden_dim]
        
        # Use the final hidden state for classification
        hidden = hidden[-1]  # Shape: [batch_size, hidden_dim]
        return self.fc(self.dropout(hidden))

# Function to prepare data for Text-RNN model
def prepare_text_data(data_path, use_templates=True):
    # Get file lists for each split
    train_files = [f for f in os.listdir(os.path.join(data_path, "train")) if f.endswith('.npy')]
    val_files = [f for f in os.listdir(os.path.join(data_path, "val")) if f.endswith('.npy')]
    
    # Create vocabulary
    texts = []
    for emotion, templates in emotion_templates.items():
        texts.extend([preprocess_text(t) for t in templates])
    
    vocabulary = Vocabulary(min_freq=1).build_vocabulary(texts)
    
    # Create datasets
    train_dataset = TextEmotionDataset(train_files, vocabulary, use_templates=use_templates)
    val_dataset = TextEmotionDataset(val_files, vocabulary, use_templates=use_templates)
    
    return train_dataset, val_dataset, vocabulary

# Train the Text-RNN model
def train_text_model(data_path, embedding_dim=100, hidden_dim=128, n_layers=2, 
                    epochs=15, batch_size=32, lr=0.001, use_templates=True):
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # # Create directories for saving results
    # os.makedirs("../plots", exist_ok=True)
    # os.makedirs("../models", exist_ok=True)
    
    # Prepare data
    train_dataset, val_dataset, vocabulary = prepare_text_data(data_path, use_templates)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = TextGRU(
        vocab_size=len(vocabulary.word2idx),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=8,  # 8 emotion classes
        n_layers=n_layers,
        dropout=0.5
    ).to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Initialize tracking variables
    best_val_acc = 0
    patience = 5
    patience_counter = 0
    train_accs = []
    val_accs = []
    
    # Training log
    with open("../accuracy_logs/text_model_log.txt", "w") as f:
        f.write("Text-RNN Training started\n")
    
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
        
        # Print stats
        log_line = f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
        log_line += f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n"
        print(log_line.strip())
        
        with open("../accuracy_logs/text_model_log.txt", "a") as f:
            f.write(log_line)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'vocab': vocabulary.word2idx,
                'val_acc': val_acc,
            }, "../models/best_text_model.pth")
            
            # Create confusion matrix for best model
            cm = confusion_matrix(all_targets, all_preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=[emotion_map[i] for i in range(8)],
                        yticklabels=[emotion_map[i] for i in range(8)])
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Text Model - Confusion Matrix - Epoch {epoch+1} (Val Acc: {val_acc:.2f}%)')
            plt.tight_layout()
            plt.savefig(f"../../plots/confusion_matrices/text_confusion_matrix_epoch_{epoch+1}.png")
            plt.close()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Plot training/validation accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Text Model - Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig("../../plots/accuracy_plots/text_accuracy_plot.png")
    plt.close()
    
    print(f"Text model training complete. Best validation accuracy: {best_val_acc:.2f}%")
    return model, best_val_acc, vocabulary

# Function to predict emotion from text
def predict_emotion_from_text(model, vocabulary, text, device):
    # Preprocess the text
    text = preprocess_text(text)
    
    # Convert to indices
    indices = vocabulary.text_to_indices(text)
    indices = torch.tensor(indices).unsqueeze(0).to(device)  # Add batch dimension
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(indices)
        _, predicted = torch.max(output, 1)
        prob = torch.nn.functional.softmax(output, dim=1)[0]
    
    # Get emotion
    emotion_idx = predicted.item()
    emotion = emotion_map[emotion_idx]
    
    # Get confidence
    confidence = prob[emotion_idx].item()
    
    return emotion, confidence, prob.cpu().numpy()

# Entry point
if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    print("Starting Text-RNN model training...")
    model, best_val_acc, vocabulary = train_text_model("../../../data/split", use_templates=True)
    
    # Testing the model on a sample text
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_text = "I am feeling very happy today"
    emotion, confidence, _ = predict_emotion_from_text(model, vocabulary, test_text, device)
    print(f"Text: '{test_text}'")
    print(f"Predicted emotion: {emotion} (confidence: {confidence:.2f})")