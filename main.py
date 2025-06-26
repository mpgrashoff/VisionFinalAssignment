from data_loader import load_data_from_influx
from preprocessor import *
from trainer import *
from visualizer import *
import pandas as pd
import os
import tensorflow as tf


def main():
    data_query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: {DATA_START}, stop: {DATA_STOP})
      |> filter(fn: (r) => r["_measurement"] == "wifi_status")
      |> filter(fn: (r) => r["_field"] == "RSSI")
      |> filter(fn: (r) => r["device"] == "RSS_A" or r["device"] == "RSS_B" or r["device"] == "RSS_C")
      |> aggregateWindow(every: 1s, fn: mean, createEmpty: true)
      |> fill(column: "_value", value: 0.0)
      |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> keep(columns: ["_time", "device", "RSSI"])
    '''

    val_query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: {DATA_START}, stop: {DATA_STOP})
      |> filter(fn: (r) => r["_measurement"] == "wifi_status")
      |> filter(fn: (r) => r["_field"] == "Pressence")
      |> aggregateWindow(every: 1s, fn: last, createEmpty: true)
      |> fill(usePrevious: true)
      |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> keep(columns: ["_time", "Pressence"])
    '''

    print("starting main()")
    print(os.environ['PATH'])  # confirms PATH inclusion for debugging
    print(f"gpu: {tf.config.list_physical_devices('GPU')}")

    raw_data = load_data_from_influx(DATA_CACHE_FILE, data_query)
    validation_data = load_data_from_influx(VALIDATION_CACHE_FILE, val_query)
    raw_data['_time'] = pd.to_datetime(raw_data['_time'])
    validation_data['_time'] = pd.to_datetime(validation_data['_time'])

    # --- feature generation & framing ---
    full_spec = generate_stft_features(raw_data)

    for i in range(SENSOR_COUNT):
        plot_spectrogram(full_spec, validation_data, max_freq_bin=46,
                         time_start=PLOT_START_TIME, time_end=PLOT_END_TIME, sensor_index=i)

    labels = generate_frame_labels(validation_data)
    frames = generate_frames(full_spec)

    frames, labels = remove_rising_falling_edge_frames(frames, labels)

    print(f"Full spectrogram shape: {full_spec.shape}")
    print(f"Frames shape: {frames.shape}, Labels shape: {labels.shape}")

    # --- k-fold training ---
    models, histories = train_kfold(frames, labels, n_splits=FOLDS)

    # --- plot learning curves for each fold ---
    # plot_kfold_history(histories)

    # --- evaluate each fold model on the full dataset (or test split) ---
    for idx, model in enumerate(models, start=1):
        print(f"\n=== Fold {idx} Evaluation on Full Set ===")
        plot_confusion_matrix(model, frames, labels)
        # plot_roc_pr_curves(model, frames, labels)



if __name__ == '__main__':
    main()
