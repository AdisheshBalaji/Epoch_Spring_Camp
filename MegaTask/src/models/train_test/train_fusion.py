import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


# from train_cnn import SimpleEmotionCNN, MelSpectrogramDataset
from train_cnn import EmotionCNN, MelSpectrogramDataset
from train_rnn import TextGRU, TextEmotionDataset, prepare_text_data, emotion_map


class MultimodalFusionModel(nn.Module):
    def __init__(self, audio_model, text_model, output_dim=8, fusion_dim=64):
        super(MultimodalFusionModel, self).__init__()
        # Store the pre-trained models
        self.audio_model = audio_model
        self.text_model = text_model
        
        # Freeze the base models (optional)
        for param in self.audio_model.parameters():
            param.requires_grad = False
        for param in self.text_model.parameters():
            param.requires_grad = False
        
        # Determine audio output dimensions based on model architecture
        if hasattr(self.audio_model, 'fc'):
            # If the model has an 'fc' layer (like the saved model)
            audio_features_shape = self._get_audio_feature_shape(audio_model)
            audio_output_dim = audio_features_shape[1] * audio_features_shape[2] * audio_features_shape[3]
            print(f"Detected old-style CNN with feature shape: {audio_features_shape}, output dim: {audio_output_dim}")
        else:
            # For the new model with classifier
            audio_output_dim = 256  # This should match what's in your SimpleEmotionCNN
            print(f"Detected new-style CNN with output dim: {audio_output_dim}")
        
        # For the text model, get the hidden dimension
        text_output_dim = self.text_model.gru.hidden_size
        if hasattr(self.text_model.gru, 'bidirectional') and self.text_model.gru.bidirectional:
            text_output_dim *= 2  # Double for bidirectional
        print(f"Text model output dim: {text_output_dim}")
        
        # Reduce dimensions before fusion
        self.audio_reducer = nn.Linear(audio_output_dim, fusion_dim)
        self.text_reducer = nn.Linear(text_output_dim, fusion_dim)
        
        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fusion_dim, output_dim)
        )
    
    def _get_audio_feature_shape(self, model):
        """Helper method to determine feature shape from the audio model"""
        # Create a dummy input
        dummy_input = torch.zeros(1, 1, 128, 128)  # Adjust size as needed
        # Get features shape
        with torch.no_grad():
            features = model.features(dummy_input)
        return features.shape
        
    def forward(self, audio_input, text_input):
        # Get embeddings from audio model
        with torch.no_grad():
            # Handle both model architectures
            audio_features = self.audio_model.features(audio_input)
            
            if hasattr(self.audio_model, 'fc'):
                # For old model with fc layer
                audio_features = torch.flatten(audio_features, 1)
            else:
                # For new model with classifier
                # Apply global average pooling if available
                if hasattr(self.audio_model.classifier, 'avg_pool') or isinstance(self.audio_model.classifier[0], nn.AdaptiveAvgPool2d):
                    audio_features = self.audio_model.classifier[0](audio_features)
                # Flatten
                audio_features = torch.flatten(audio_features, 1)
        
        # Get embeddings from text model
        with torch.no_grad():
            # Forward through embedding and GRU
            embedded = self.text_model.embedding(text_input)
            _, text_features = self.text_model.gru(embedded)
            
            # Get the last hidden state
            if hasattr(self.text_model.gru, 'bidirectional') and self.text_model.gru.bidirectional:
                # For bidirectional GRU
                text_features = torch.cat((text_features[-2,:,:], text_features[-1,:,:]), dim=1)
            else:
                # For unidirectional GRU
                text_features = text_features[-1]
        
        # Reduce dimensions
        audio_reduced = self.audio_reducer(audio_features)
        text_reduced = self.text_reducer(text_features)
        
        # Concatenate the features
        combined = torch.cat((audio_reduced, text_reduced), dim=1)
        
        # Classify
        output = self.classifier(combined)
        return output




# Dataset that combines audio and text
class MultimodalDataset(Dataset):
    def __init__(self, audio_dataset, text_dataset):
        self.audio_dataset = audio_dataset
        self.text_dataset = text_dataset
        
        # Ensure both datasets have the same length
        assert len(audio_dataset) == len(text_dataset), "Audio and text datasets must have the same length"
        
        # Make sure the emotion labels match between datasets
        for i in range(len(audio_dataset)):
            audio_label = audio_dataset[i][1]
            text_label = text_dataset[i][1]
            assert audio_label == text_label, f"Label mismatch at index {i}: audio={audio_label}, text={text_label}"
            
    def __len__(self):
        return len(self.audio_dataset)
    
    def __getitem__(self, idx):
        # Get audio and text samples
        audio_data, label = self.audio_dataset[idx]
        text_data, _ = self.text_dataset[idx]  # We already verified labels match
        
        # Return audio, text, and label
        return audio_data, text_data, label

# Training function for the multimodal model
def train_multimodal_model(model, train_loader, val_loader, criterion, optimizer, 
                          num_epochs=10, device='cuda', patience=5):
    # Move model to device
    model.to(device)
    
    # Initialize lists to track metrics
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # Initialize early stopping variables
    best_val_acc = 0
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for audio_inputs, text_inputs, labels in train_loader:
            # Move data to device
            audio_inputs = audio_inputs.to(device)
            text_inputs = text_inputs.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(audio_inputs, text_inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for audio_inputs, text_inputs, labels in val_loader:
                # Move data to device
                audio_inputs = audio_inputs.to(device)
                text_inputs = text_inputs.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(audio_inputs, text_inputs)
                loss = criterion(outputs, labels)
                
                # Statistics
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100 * val_correct / val_total
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)
        
        print(f'Validation Loss: {epoch_val_loss:.4f}, Validation Acc: {epoch_val_acc:.2f}%')
        
        # Early stopping check
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            epochs_no_improve = 0
            # Save the best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc
            }, '../state_dicts/final_multimodal_model.pth')
            print("Model saved!")
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    # Load the best model
    best_model = torch.load('../state_dicts/final_multimodal_model.pth')
    model.load_state_dict(best_model['model_state_dict'])
    
    return model, train_losses, val_losses, val_accuracies

# Evaluation function for the multimodal model
def evaluate_multimodal_model(model, test_loader, device='cuda'):
    # Move model to device
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for audio_inputs, text_inputs, labels in test_loader:
            # Move data to device
            audio_inputs = audio_inputs.to(device)
            text_inputs = text_inputs.to(device)
            
            # Forward pass
            outputs = model(audio_inputs, text_inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            # Add to lists
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=list(emotion_map.keys()), output_dict=True)
    
    return cm, report, all_preds, all_labels

# Function to visualize training history
def plot_training_history(train_losses, val_losses, val_accuracies):
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('../plots/accuracy_plots/multimodal_training_history.png')
    plt.show()

# Function to visualize confusion matrix
def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('../plots/confusion_matrices/multimodal_confusion_matrix.png')
    plt.show()

# Main function to run the multimodal fusion experiment
def run_multimodal_experiment(audio_train_dataset, audio_val_dataset, audio_test_dataset,
                              text_train_dataset, text_val_dataset, text_test_dataset,
                              audio_model, text_model):
    
    # Create multimodal datasets
    multimodal_train_dataset = MultimodalDataset(audio_train_dataset, text_train_dataset)
    multimodal_val_dataset = MultimodalDataset(audio_val_dataset, text_val_dataset)
    multimodal_test_dataset = MultimodalDataset(audio_test_dataset, text_test_dataset)
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(multimodal_train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(multimodal_val_dataset, batch_size=batch_size)
    test_loader = DataLoader(multimodal_test_dataset, batch_size=batch_size)
    
    # Initialize multimodal model
    output_dim = len(emotion_map)
    multimodal_model = MultimodalFusionModel(audio_model, text_model, output_dim=output_dim)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # Only optimize the fusion layers, since base models are frozen
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, multimodal_model.parameters()), lr=0.001)
    
    # Choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Train model
    model, train_losses, val_losses, val_accuracies = train_multimodal_model(
        multimodal_model, train_loader, val_loader, criterion, optimizer,
        num_epochs=20, device=device, patience=5
    )
    
    # Plot training history
    plot_training_history(train_losses, val_losses, val_accuracies)
    
    # Evaluate model
    cm, report, all_preds, all_labels = evaluate_multimodal_model(
        model, test_loader, device=device
    )
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, list(emotion_map.keys()))
    
    # Print classification report
    print("\nClassification Report:")
    for emotion, metrics in report.items():
        if emotion in ['accuracy', 'macro avg', 'weighted avg']:
            continue
        print(f"{emotion}: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
    
    print(f"\nOverall Accuracy: {report['accuracy']:.3f}")
    print(f"Macro Avg F1-Score: {report['macro avg']['f1-score']:.3f}")
    
    return model, report

# Example of how to use the above code
if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Path to your data
    data_dir = "../../data/split"
    
    # Load the saved audio model
    audio_model = EmotionCNN(num_classes=8)
    # Handle different checkpoint formats
    try:
        checkpoint = torch.load('../state_dicts/best_cnn_model.pth')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # If checkpoint is a dictionary with model_state_dict
            audio_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # If checkpoint is just the state dict
            audio_model.load_state_dict(checkpoint)
        audio_model.eval()
        print("Audio model loaded successfully!")
    except Exception as e:
        print(f"Failed to load audio model: {e}")
        exit(1)
    
    # Load the saved text model
    try:
        # Load the text model checkpoint
        checkpoint = torch.load('../state_dicts/best_text_model.pth')
        
        # Get vocabulary size from checkpoint if available, otherwise use default
        if isinstance(checkpoint, dict) and 'vocab' in checkpoint:
            vocab_size = len(checkpoint['vocab'])
        else:
            vocab_size = 10000  # Default vocab size
        
        # Initialize the text model with the same architecture used during training
        text_model = TextGRU(
            vocab_size=vocab_size, 
            embedding_dim=100, 
            hidden_dim=128, 
            output_dim=8,
            n_layers=2,
            dropout=0.5
        )
        
        # Load the weights
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            text_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            text_model.load_state_dict(checkpoint)
        
        text_model.eval()
        print("Text model loaded successfully!")
    except Exception as e:
        print(f"Failed to load text model: {e}")
        exit(1)
    
    # Load or prepare datasets
    try:
        # Load audio datasets
        audio_train_dataset = MelSpectrogramDataset(os.path.join(data_dir, 'train'))
        audio_val_dataset = MelSpectrogramDataset(os.path.join(data_dir, 'val'))
        audio_test_dataset = MelSpectrogramDataset(os.path.join(data_dir, 'test'))
        
        # Prepare text datasets
        # First get train and val sets to access the vocabulary
        train_text_dataset, val_text_dataset, vocabulary = prepare_text_data(data_dir)
        
        # Then create the test dataset using the same vocabulary
        test_files = [f for f in os.listdir(os.path.join(data_dir, 'test')) if f.endswith('.npy')]
        test_text_dataset = TextEmotionDataset(test_files, vocabulary)
        
        print("Datasets prepared successfully!")
    except Exception as e:
        print(f"Failed to prepare datasets: {e}")
        exit(1)
    
    # Run the multimodal experiment
    multimodal_model, results = run_multimodal_experiment(
        audio_train_dataset, audio_val_dataset, audio_test_dataset,
        train_text_dataset, val_text_dataset, test_text_dataset,
        audio_model, text_model
    )
    
    # Save the final model (as a checkpoint)
    torch.save({
        'model_state_dict': multimodal_model.state_dict(),
        'results': results
    }, '../state_dicts/final_multimodal_model.pth')
    print("Final model saved successfully!")