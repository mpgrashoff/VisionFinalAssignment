# 📡 Wi-Fi Presence Detection using Deep Learning

Welcome to the Wi-Fi Presence Detection project! This system leverages Wi-Fi RSSI signals, processes them using STFT spectrograms, and trains deep learning models to detect human presence in a room. Perfect for smart home automation, security, or occupancy analytics.

---

## 🚀 Project Overview

This project collects Wi-Fi RSSI data from multiple sensors, generates time-frequency features using Short-Time Fourier Transform (STFT), and trains a neural network classifier using K-Fold cross-validation. The pipeline includes:

- Data querying & caching from InfluxDB
- STFT feature extraction per sensor
- Frame generation & labeling based on presence data
- Stratified K-Fold model training with early stopping
- Model evaluation with confusion matrix and ROC/PR curves
- Visualization of spectrograms, learning curves, and classification metrics

---

## 🧩 Components

### 1. Data Loader (`data_loader.py`)
- Loads and caches data from InfluxDB using Flux queries.
- Supports caching to CSV to speed up iterative experimentation.

### 2. Preprocessor (`preprocessor.py`)
- Generates STFT spectrogram features from RSSI signals.
- Splits spectrograms into fixed-length frames.
- Generates binary presence labels per frame.
- Removes noisy frames at label transitions (rising/falling edges).

### 3. Trainer (`trainer.py`)
- Trains models using Stratified K-Fold cross-validation.
- Implements early stopping to avoid overfitting.
- Saves the best model per fold with timestamped filenames.

### 4. Visualizer (`visualizer.py`)
- Plots STFT spectrograms overlaid with presence signals.
- Displays training/validation accuracy and loss per fold.
- Generates confusion matrices and classification reports.
- Plots multi-class ROC and Precision-Recall curves.

---

## ⚙️ Configuration

Configure parameters in `config.py` such as:

- InfluxDB credentials (`URL`, `TOKEN`, `ORG`, `INFLUX_BUCKET`)
- Data collection range (`DATA_START`, `DATA_STOP`)
- Signal processing settings (`FRAME_TIME`, `NPERSEG`, `OVERLAP`)
- Model training hyperparameters (`EPOCHS`, `BATCH_SIZE`, `PATiENCE`, `FOLDS`)
- Sensor count (`SENSOR_COUNT`)

---

## 💡 How to Run

1. Set your InfluxDB credentials and parameters in `config.py`.
2. Run the main pipeline:

```bash
python main.py
```

---
## 📚 Dependencies
- Python 3.8+
- pandas
- numpy
- scipy
- influxdb-client
- tensorflow / keras
- scikit-learn
- matplotlib
