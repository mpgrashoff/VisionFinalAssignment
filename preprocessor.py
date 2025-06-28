import pandas as pd
from config import *
from helpers import classify_boolean_frame
from scipy.signal import stft
import numpy as np
from config import FRAME_TIME, NPERSEG, OVERLAP


def generate_frames(spectrogram):
    """
    Splits a full STFT spectrogram into fixed-duration frames.

    Frames are generated along the time axis of the spectrogram using
    parameters defined in config (FRAME_TIME, NPERSEG, OVERLAP).

    Args:
        spectrogram (np.ndarray): Spectrogram of shape (freq_bins, time_bins, channels).

    Returns:
        np.ndarray: Array of shape (n_frames, freq_bins, window_bins, channels),
                    where each frame corresponds to a fixed time window.
    """

    hop = NPERSEG - OVERLAP                   # seconds per STFT time bin
    window_bins = FRAME_TIME // hop           # how many STFT bins in one frame
    freq_bins, time_bins, channels = spectrogram.shape

    frames = []

    for start_bin in range(0, time_bins - window_bins + 1, window_bins):
        end_bin = start_bin + window_bins
        frame = spectrogram[:, start_bin:end_bin, :]
        frames.append(frame)

    return np.stack(frames, axis=0)


def generate_frame_labels(validation_data):
    """
    Generates binary labels for each time frame based on presence data.

    The validation data is split into frames of duration FRAME_TIME and
    each frame is labeled using a boolean classification function.

    Args:
        validation_data (pd.DataFrame): DataFrame with '_time' and 'Pressence' columns.

    Returns:
        np.ndarray: Array of shape (n_frames,) containing binary labels (0 or 1).
    """

    label_list = []
    frame_start = pd.to_datetime(DATA_START)
    frame_duration = pd.Timedelta(seconds=FRAME_TIME)

    while True:
        frame_stop = frame_start + frame_duration

        # Extract current frame slice
        label_slice = validation_data[(validation_data['_time'] > frame_start) &
                                      (validation_data['_time'] <= frame_stop)]

        # If not enough data, assume we've reached the end
        if label_slice.shape[0] < FRAME_TIME * PULLING_RATE:
            break

        # Classify current frame
        label = classify_boolean_frame(label_slice['Pressence'])
        label_list.append(label)

        # Advance to the next frame
        frame_start = frame_stop

    return np.array(label_list)


def compute_stft(signal, fs=PULLING_RATE, nperseg=NPERSEG, noverlap=OVERLAP):
    """
    Computes the Short-Time Fourier Transform (STFT) magnitude of a 1D signal.

    Args:
        signal (np.ndarray): 1D time-series input signal.
        fs (int, optional): Sampling rate in Hz. Defaults to PULLING_RATE.
        nperseg (int, optional): Length of each segment for STFT. Defaults to NPERSEG.
        noverlap (int, optional): Number of overlapping samples. Defaults to OVERLAP.

    Returns:
        np.ndarray: 2D STFT magnitude array of shape (freq_bins, time_bins).
    """
    f, t, zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return np.abs(zxx)  # magnitude only


def generate_stft_features(raw_data):
    """
    Converts raw RSSI data into a 3-channel STFT spectrogram image.

    Applies STFT to each sensor's time-series signal and stacks the
    resulting spectrograms along the channel axis.

    Args:
        raw_data (pd.DataFrame): DataFrame with 'device', 'RSSI', and '_time' columns.

    Returns:
        np.ndarray: Spectrogram image of shape (freq_bins, time_bins, SENSOR_COUNT).
    """

    spectrograms = []
    sensors = raw_data['device'].unique()
    if len(sensors) != SENSOR_COUNT:
        raise ValueError(f"Expected {SENSOR_COUNT} sensors, found {len(sensors)}: {sensors}")

    rssi_data = np.vstack([
        raw_data[raw_data['device'] == sen]['RSSI'].to_numpy()
        for sen in sensors
    ])

    for i in range(SENSOR_COUNT):
        signal = rssi_data[i, :]
        # signal = 10 ** (signal/10)  # reconstruct the log scale signal for a more accurate real world representation
        # also maybe the worst line of code i have ever written ^^ this function made everything not work and i did not realise for way to long
        spec = compute_stft(signal)
        spectrograms.append(spec)
    # Stack as image channels: shape → (freq_bins, time_bins, 3)
    return np.stack(spectrograms, axis=-1)


def remove_rising_falling_edge_frames(frames, frame_labels):
    """
    Filters out frames that lie on rising or falling edges of presence transitions.

    Only retains frames whose labels are stable across the previous and next frame,
    to avoid transitional ambiguity.

    Args:
        frames (np.ndarray): Array of frames, shape (n_frames, freq_bins, window_bins, channels).
        frame_labels (np.ndarray): Array of frame labels, shape (n_frames,).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Filtered frames and corresponding labels.
    """

    filtered_frames = []
    filtered_labels = []

    for i in range(1, len(frame_labels) - 1):
        if frame_labels[i - 1] == frame_labels[i] == frame_labels[i + 1]:
            filtered_frames.append(frames[i])
            filtered_labels.append(frame_labels[i])

    return np.array(filtered_frames), np.array(filtered_labels)
