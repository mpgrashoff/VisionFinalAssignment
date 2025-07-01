from data_loader import load_data_from_influx
from preprocessor import *
from trainer import *
from visualizer import *
import pandas as pd
import os
import tensorflow as tf


def main():
    print("loading model")
    try:
        model = tf.keras.models.load_model('./models/GOATV2.keras')
        print("model loaded")
    except Exception as e:
        print(f"model failed to load with error: {e} ")
        return -1

    # Show the model architecture
    model.summary()

    # grabs last 120 seconds of data
    data_query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -120s)
      |> filter(fn: (r) => r["_measurement"] == "wifi_status")
      |> filter(fn: (r) => r["_field"] == "RSSI")
      |> filter(fn: (r) => r["device"] == "RSS_A" or r["device"] == "RSS_B" or r["device"] == "RSS_C")
      |> aggregateWindow(every: 1s, fn: mean, createEmpty: true)
      |> fill(column: "_value", value: 0.0)
      |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> keep(columns: ["_time", "device", "RSSI"])
    '''

    while True:
        # fetch data data_frame from influxDB
        data_frame = load_data_from_influx(data_query)
        data_frame['_time'] = pd.to_datetime(data_frame['_time'])
        frame = generate_stft_features(data_frame)
        model.predict(frame)
        

if __name__ == "__main__":
    main()
