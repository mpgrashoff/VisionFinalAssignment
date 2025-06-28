import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_recall_curve, auc, roc_curve
from sklearn.preprocessing import label_binarize
from config import *


def plot_history(history):
    """
    Plots training loss and accuracy over epochs.

    Args:
        history (keras.callbacks.History): History object returned by model.fit().
    """

    epochs = range(1, len(history.history['loss']) + 1)
    plt.figure()
    plt.plot(epochs, history.history['loss'], 'bo', label='Training loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epochs, history.history['accuracy'], 'b', label='Training accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def plot_spectrogram(spectrogram, control_df, max_freq_bin=60, time_start=0, time_end=None,
                     sensor_index=0, sensors=['RSS_A', 'RSS_B', 'RSS_C']):
    """
    Plots an STFT spectrogram for a selected sensor, overlaid with the presence signal.

    Args:
        spectrogram (np.ndarray): Array of shape (freq_bins, time_bins, channels).
        control_df (pd.DataFrame): DataFrame containing '_time' and 'Pressence' columns.
        max_freq_bin (int): Upper frequency bin limit to plot. Defaults to 60.
        time_start (int): Starting time bin index for the plot.
        time_end (int or None): Ending time bin index. If None, plots to end.
        sensor_index (int): Index of the sensor/channel to plot.
        sensors (List[str]): List of sensor names for labeling purposes.
    """

    if spectrogram.ndim != 3:
        raise ValueError("Unexpected spectrogram shape.")

    data = spectrogram[:max_freq_bin, :, sensor_index]
    if time_end is None:
        time_end = data.shape[1]
    data = data[:, time_start:time_end]

    # Determine hop length and time resolution of STFT
    hop_length = NPERSEG - OVERLAP
    stft_time_resolution = hop_length  # in original sampling units (e.g., seconds for 1 Hz RSSI)

    # Normalize timestamps to seconds since start
    start_time = control_df['_time'].min()
    control_df['seconds'] = (control_df['_time'] - start_time).dt.total_seconds()

    # Map seconds to STFT time bin indices
    control_df['stft_time_bin'] = control_df['seconds'] / stft_time_resolution

    # Filter control_df to match the plotted spectrogram time range
    mask = (control_df['stft_time_bin'] >= time_start) & (control_df['stft_time_bin'] <= time_end)
    control_df = control_df[mask]

    # Prepare line for overlay
    x_overlay = control_df['stft_time_bin'].values - time_start
    y_overlay = control_df['Pressence'].astype(int).values * (max_freq_bin - 1)

    # Plot
    plt.figure(figsize=(10, 4))
    plt.imshow(data, aspect='auto', origin='lower', cmap='magma')
    plt.plot(x_overlay, y_overlay, color='cyan', linewidth=2, label='Presence Signal')
    plt.colorbar(label='Magnitude')
    plt.title(f"STFT Spectrogram for {sensors[sensor_index]}")
    plt.xlabel('Time bins')
    plt.ylabel('Frequency bins')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_kfold_history(histories):
    """
    Plot training and validation loss/accuracy for each fold.

    Args:
        histories (list): List of Keras History objects from each fold.
    """
    for fold_idx, hist in enumerate(histories, start=1):
        # Accuracy plot
        plt.figure()
        plt.plot(hist.history['accuracy'], label='train_accuracy')
        plt.plot(hist.history['val_accuracy'], label='val_accuracy')
        plt.title(f'Fold {fold_idx} Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Loss plot
        plt.figure()
        plt.plot(hist.history['loss'], label='train_loss')
        plt.plot(hist.history['val_loss'], label='val_loss')
        plt.title(f'Fold {fold_idx} Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.show()


def plot_confusion_matrix(model, X, y_true):
    """
    Compute and plot the confusion matrix and classification report for multi-class classification.

    Args:
        model: A trained Keras model.
        X (np.array): Input data.
        y_true (np.array): True integer labels (0 through n_classes-1).
    """
    # Predict class probabilities and take the argmax
    y_pred = np.argmax(model.predict(X), axis=1)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    num_classes = cm.shape[0]

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', aspect='auto', origin='lower', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.colorbar()
    ticks = np.arange(num_classes)
    plt.xticks(ticks)
    plt.yticks(ticks)

    # Annotate counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='black')

    plt.tight_layout()
    plt.show()

    # Classification report
    print('Classification Report:')
    print(classification_report(y_true, y_pred, digits=3))


def plot_roc_pr_curves(model, X, y_true):
    """
    Plot ROC and Precision-Recall curves for multi-class classification using one-vs-rest.

    Args:
        model: A trained Keras model.
        X (np.array): Input data.
        y_true (np.array): True integer labels (0 through n_classes-1).
    """
    # Predict class probabilities
    y_prob = model.predict(X)  # shape (n_samples, n_classes)
    n_classes = y_prob.shape[1]

    # Binarize the true labels for one-vs-rest metrics
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    # ROC curve and AUC for each class
    plt.figure()
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curves')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

    # Precision-Recall curve and AUC for each class
    plt.figure()
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f'Class {i} (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Multi-class Precision-Recall Curves')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.show()

