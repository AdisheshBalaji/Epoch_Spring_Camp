# Multimodal Emotion Classification

This project implements a deep learning system for classifying human emotions using both audio data (processed as spectrograms) and text transcripts generated from speech. The system leverages Convolutional Neural Networks (CNNs) for audio processing and Recurrent Neural Networks (RNNs) for text analysis, with a multimodal fusion approach to combine both.

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

We use the RAVDESS Emotional Speech Audio dataset, which contains 1440 speech clips from male and female actors vocalizing different emotions using only two sentences
"Kids are talking by the door"
"Dogs are sitting by the door"


## Implementation Details

### Phase 1: Unimodal Pipelines

#### Audio CNN
- Converts audio files to Mel Spectrograms
- Trains a CNN model to classify emotions on these Mel Spectrograms
- Implementation in `train_cnn.py`

#### Text RNN
- Since there are only two sentences in the training data, for meaningful rnn training
- I tried to train it on the original dataset without templates, this gave an accuracy of `14.95%`, which is no better than random guessing 
- Simulates text data for 8 emotions using predefined template sentences.
- Trains an RNN (GRU) on these transcripts for emotion classification
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
- Confusion matrices after intervals of epochs to get a better understanding of the working of the models
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


