# Multimodal Emotion Classification

This project implements a deep learning system for classifying human emotions using both audio data (processed as spectrograms) and text transcripts generated from speech. The system leverages Convolutional Neural Networks (CNNs) for audio processing and Recurrent Neural Networks (RNNs) for text analysis, with a multimodal fusion approach to combine both modalities.

## Project Overview

The primary objective is to classify emotional speech into 8 emotions:
- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

We use the RAVDESS Emotional Speech Audio dataset, which contains 1440 speech clips from male and female actors expressing different emotions.

## Directory Structure

```
.
├── data                        # Raw and processed data storage
├── src                         # Source code
│   ├── data                    # Data processing scripts
│   │   ├── dataset.py          # Dataset handling and loading
│   │   ├── preprocess.py       # Audio and text preprocessing
│   │   └── split_dataset.py    # Train/validation/test splitting
│   ├── models                  # Model definitions and training
│   │   ├── accuracy_logs       # Training metrics and logs
│   │   ├── models              # Saved model architectures
│   │   │   ├── best_cnn_model.pth      # Best CNN model
│   │   │   ├── best_multimodal_model.pth  # Best fusion model
│   │   │   └── best_text_model.pth     # Best text model
│   │   └── state_dicts         # Saved model weights
│   │       ├── best_cnn_model.pth      # CNN weights
│   │       ├── best_text_model.pth     # Text model weights
│   │       └── final_multimodal_model.pth  # Fusion model weights
│   ├── train_test              # Training and evaluation scripts
│   │   ├── __pycache__         # Python cache files
│   │   ├── train_cnn.py        # CNN training script
│   │   ├── train_fusion.py     # Multimodal fusion training
│   │   └── train_rnn.py        # RNN training script
│   └── plots                   # Visualization outputs
│       ├── accuracy_plots      # Model accuracy visualizations
│       │   ├── cnn_accuracy_plot.png          # CNN accuracy
│       │   ├── multimodal_training_history.png # Fusion model performance
│       │   └── text_accuracy_plot.png         # Text model accuracy
│       ├── confusion_matrices  # Model evaluation matrices
│       └── sample_spectrograms # Example spectrogram visualizations
├── .gitignore                  # Git ignore file
├── README.md                   # Project documentation
└── requirements.txt            # Dependencies
```

## Implementation Details

### Phase 1: Unimodal Pipelines

#### Audio CNN
- Converts audio files to spectrograms or MFCCs
- Trains a CNN model to classify emotions from these 2D visual representations
- Implementation in `train_cnn.py`

#### Text RNN
- Generates transcripts from audio using speech-to-text technology
- Trains an RNN (LSTM or GRU) on these transcripts for emotion classification
- Implementation in `train_rnn.py`

### Phase 2: Multimodal Fusion
- Merges features from both CNN and RNN models
- Implements different fusion strategies:
  - Early fusion: concatenates embeddings before classification
  - Late fusion: uses voting or averaging of model outputs
- Implementation in `train_fusion.py`

## Results

The project includes various visualizations to evaluate model performance:
- Accuracy plots for individual modalities and the multimodal approach
- Confusion matrices for error analysis
- Training history plots

## Getting Started

### Prerequisites
```
pip install -r requirements.txt
```

### Data Preparation
1. Download the RAVDESS dataset
2. Run preprocessing:
```
python src/data/preprocess.py
```

### Training
```
# Train CNN model
python src/train_test/train_cnn.py

# Train RNN model
python src/train_test/train_rnn.py

# Train fusion model
python src/train_test/train_fusion.py
```

### Evaluation
Results are saved in the `src/plots` directory.

## License
[Insert license information here]

## Acknowledgements
- RAVDESS Emotional Speech Audio dataset
- [Add any other acknowledgements]
