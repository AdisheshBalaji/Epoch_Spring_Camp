# Epoch IITH – Community Task

This repository contains code for the community task by **Epoch IITH**.

We use the [`RAVDESS`](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio) dataset to train an AudioCNN on `.wav` files by converting them into **Mel Spectrograms**.

---

## 🔧 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/AdisheshBalaji/Epoch_Spring_Camp.git
cd Epoch_Spring_Camp/MegaTask
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Dataset Setup

* Download the `ravdess` dataset from Kaggle.
* Place the dataset in the **root directory** of the project (`MegaTask/`).

---

### 4. Preprocess the Data

```bash
cd src/data
python3 dataset.py
python3 preprocess.py
python3 split_dataset.py
```

---

### 5. Visualize Spectrograms

```bash
cd ../visualize
python3 show_spectrograms.py
```

---

### 6. Train the CNN

```bash
cd ../models
python3 train_cnn.py
```

---

### 7. Check Training Accuracy

```bash
nvim accuracy_log.txt
```

(Or use any text editor of your choice)

---

### 8. Evaluate the Model on Test Data

```bash
python3 test.py
```

---

### 9. Run Inference on a Sample Training File

```bash
python3 predict.py
```

---

>  **Note**: The `/rnn` folder is still under development.

---


