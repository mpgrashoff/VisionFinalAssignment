# ⚠️⚠️ this file is deprecated and is only here to show the original prototype and preserve the code ⚠️⚠️

# # Press Shift+F10 to execute it or replace it with your code.
#
# import datetime
# import os.path
# import tensorflow as tf
# from tensorflow import keras
# from influxdb_client import InfluxDBClient, Point, WritePrecision
# from keras.callbacks import EarlyStopping
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import stft
#
# # do not change sensor_count and pulling_rate
# SENSOR_COUNT = 3
# PULLING_RATE = 1    # 1HZ
# FRAME_TIME = 120  # 120 seconds
# data_cache_file = "query_results.csv"
# validation_cache_file = "validation_data.csv"
# bucket = "RSSI_DATA"
# org = "User"
# token = "s3i2RyDAbl_wEt1svvqQ6liyvh7OyHjo9l_LUEW58JScQLDaDFqlZHYMtT-ry2PnfZJ00omoVoIvQx52ryXYdA=="
# url = "http://192.168.2.30:8086"
# dataStart = "2025-06-03T11:59:59Z"
# dataStop = "2025-06-06T01:00:00Z"
#
#
# def plot_data():
#     # Function to plot the data
#     # Plot the data
#     plt.figure(figsize=(10, 6))
#     for device in ['RSS_A', 'RSS_B', 'RSS_C']:
#         device_data = raw_data[raw_data['device'] == device]
#         if not device_data.empty:
#             plt.plot(device_data['_time'], device_data['RSSI'], label=device)
#
#     # Add labels, title, and legend
#     plt.xlabel('Time')
#     plt.ylabel('RSSI (dBm)')
#     plt.title('RSSI Over Time')
#     plt.legend()
#     plt.grid(True)
#
#     # Show the plot
#     plt.show()
#
#
# def classify_boolean_frame(series: pd.Series) -> int:
#     """
#     Classify a boolean series into one of the following categories:
#     0 = always low (all False)
#     1 = always high (all True)
#     2 = rising edge only
#     3 = falling edge only
#     4 = both rising and falling edges
#     """
#     # Ensure input is a boolean Series
#     s = pd.Series(series).astype(bool)
#     s_int = s.astype(int)
#     diff = s_int.diff().fillna(0)
#
#     has_rising = (diff == 1).any()
#     has_falling = (diff == -1).any()
#     always_low = s_int.sum() == 0
#     always_high = s_int.sum() == len(s_int)
#
#     if always_low:
#         return 0
#     elif always_high:
#         return 1
#     elif has_rising and has_falling:
#         return 4
#     elif has_rising:
#         return 2
#     elif has_falling:
#         return 3
#     else:
#         return -1  # undefined or unexpected case
#
#
# def grab_data(file_name, query):
#     # check if tensorflow is installed and print the version and number of GPUs available
#     print("TensorFlow version:", tf.__version__)
#     print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
#
#     # Function to check if the cache file exists and load it if available
#     if os.path.exists(file_name):
#         print(f"Loading cached results from {file_name}.")
#         try:
#             result_ = pd.read_csv(file_name)
#         except Exception as e:
#             print(f"Error loading cache file: {e}")
#             result_ = None
#     else:
#         try:
#             result_ = query_api.query_data_frame(org=org, query=query)
#             if result_ is not None and not result_.empty:
#                 # Save the result to a CSV file for caching
#                 result_.to_csv(file_name, index=False)
#                 print(f"Query results saved to {file_name}.")
#         except Exception as e:
#             print(f"Error executing query: {e}")
#             result_ = None
#     # EXIT if no data is found
#     if result_ is None or result_.empty:
#         print("No data found for the specified query.")
#         exit(1)
#     return result_
#
#
# if __name__ == '__main__':
#     # Initialize the InfluxDB client
#     client = InfluxDBClient(url=url, token=token, org=org)
#     query_api = client.query_api()
#
#     # Create a frame to hold the data
#     data_frame_list = []
#     validation_frame_list = []
#     print(f"frame shape = [sensors,frame-time * pulling-rate]= {SENSOR_COUNT}, {FRAME_TIME * PULLING_RATE}]")
#
#     # Query pulls all data from 12:00:00 03-06-25 to 01:00:00 06-06-25
#     # query = f'''
#     # from(bucket: "RSSI_DATA")
#     #   |> range(start: {dataStart}, stop: {dataStop})
#     #   |> filter(fn: (r) => r["_measurement"] == "wifi_status")
#     #   |> filter(fn: (r) => r["_field"] == "RSSI")
#     #   |> filter(fn: (r) => r["device"] == "RSS_A" or r["device"] == "RSS_B" or r["device"] == "RSS_C")
#     #   |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
#     #   |> keep(columns: ["_time", "device", "RSSI"])
#     # '''
#     data_query = f'''
#     from(bucket: "RSSI_DATA")
#       |> range(start: {dataStart}, stop: {dataStop})
#       |> filter(fn: (r) => r["_measurement"] == "wifi_status")
#       |> filter(fn: (r) => r["_field"] == "RSSI")
#       |> filter(fn: (r) => r["device"] == "RSS_A" or r["device"] == "RSS_B" or r["device"] == "RSS_C")
#       |> aggregateWindow(every: 1s, fn: mean, createEmpty: true)
#       |> fill(column: "_value", value: 0.0)
#       |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
#       |> keep(columns: ["_time", "device", "RSSI"])
#     '''
#     validation_query = f'''
#     from(bucket: "RSSI_DATA")
#       |> range(start: {dataStart}, stop: {dataStop})
#       |> filter(fn: (r) => r["_measurement"] == "wifi_status")
#       |> filter(fn: (r) => r["_field"] == "Pressence")
#       |> aggregateWindow(every: 1s, fn: last, createEmpty: true)
#       |> fill(usePrevious: true)
#       |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
#       |> keep(columns: ["_time", "Pressence"])
#     '''
#
#     # Check if the cache file exists and load it if available
#     raw_data = grab_data(data_cache_file, data_query)
#     validation_data = grab_data(validation_cache_file, validation_query)
#     # Ensure '_time' is in datetime format
#     raw_data['_time'] = pd.to_datetime(raw_data['_time'])
#     validation_data['_time'] = pd.to_datetime(validation_data['_time'])
#
#     # Perform the filtering
#     raw_data = raw_data[['_time', 'device', 'RSSI']]
#     # while results contain valid 2 minute frames extract all the values within the frame and fit them in frame list
#     # any excess values will be dropped
#     # check the count of the results it should be 3* 120
#     # if correct then fill the frame with the values
#     # if to low fill with the average of the current frame
#     # if to high drop all the tailing values
#     # update the query to the next 2 minutes
#
#     frame_start = pd.to_datetime(dataStart)
#     frame_stop = frame_start + pd.Timedelta(seconds=FRAME_TIME)
#     validation_frame = validation_data[(validation_data['_time'] > frame_start) & (validation_data['_time'] <= frame_stop)]
#     tempFrame = raw_data[(raw_data['_time'] > frame_start) & (raw_data['_time'] <= frame_stop)]
#     frameCount = 0
#     while not tempFrame.empty:  # query grabbing 2 minutes of data is valid
#         # print(f"Frame {frameCount} - Start: {frame_start}, Stop: {frame_stop} - {tempFrame.shape[0]} values found.")
#         frameCount += 1
#         if tempFrame.shape[0] == SENSOR_COUNT * FRAME_TIME * PULLING_RATE:
#             # normalize the data to 1 to 0
#             tempFrame.loc[:, 'RSSI'] = (tempFrame['RSSI'] + 100) / 70.0
#             # reshape data
#             rssi_a = tempFrame[tempFrame['device'] == 'RSS_A']['RSSI'].to_numpy()
#             rssi_b = tempFrame[tempFrame['device'] == 'RSS_B']['RSSI'].to_numpy()
#             rssi_c = tempFrame[tempFrame['device'] == 'RSS_C']['RSSI'].to_numpy()
#             # Stack them into a (3, 120) array
#             frame = np.vstack([rssi_a, rssi_b, rssi_c])
#             data_frame_list.append(frame)
#         elif tempFrame.shape[0] < SENSOR_COUNT * FRAME_TIME * PULLING_RATE:
#             # no values should be missing throw error
#             if tempFrame.shape[0] < 6 * PULLING_RATE:
#                 break
#             else:
#                 raise ValueError("Not enough data points in the frame, expected 3 * 120 * 1 but got less.")
#         elif tempFrame.shape[0] > SENSOR_COUNT * FRAME_TIME * PULLING_RATE:
#             # error
#             raise ValueError("To manny data points in the frame, expected 3 * 120 * 1 but got more.")
#         if validation_frame.shape[0] == FRAME_TIME * PULLING_RATE:
#             # classify the boolean series into one of the categories
#             validation_series = validation_frame['Pressence'].astype(bool)
#             classification = classify_boolean_frame(validation_series)
#             validation_frame_list.append(classification)
#
#         # Update the frame start time to the next 2 minutes
#         frame_start += pd.Timedelta(seconds=FRAME_TIME)
#         frame_stop += pd.Timedelta(seconds=FRAME_TIME)
#         tempFrame = raw_data[(raw_data['_time'] > frame_start) & (raw_data['_time'] <= frame_stop)]
#         validation_frame = validation_data[(validation_data['_time'] > frame_start) & (validation_data['_time'] <= frame_stop)]
#
#     # plot_data()
#
# # whyyy
#     print(f"Total frames collected: {len(data_frame_list)}")
#     # Convert the list of frames to a numpy array
#     data_frame_list = np.array(data_frame_list)
#     validation_frame_list = np.array(validation_frame_list)
#
#     model = keras.models.Sequential([
#         keras.layers.Input(shape=(3, 120)),
#         keras.layers.Convolution2D(32, 1, padding='same', activation='relu'),
#         keras.layers.Flatten(),
#         keras.layers.Dense(512, activation='relu'),
#         # keras.layers.BatchNormalization(),
#         keras.layers.Dropout(0.3),
#         keras.layers.Dense(512, activation='relu'),
#         # keras.layers.BatchNormalization(),
#         keras.layers.Dropout(0.3),
#         keras.layers.Dense(512, activation='relu'),
#         keras.layers.BatchNormalization(),
#         keras.layers.Dropout(0.3),
#         keras.layers.Dense(1024, activation='relu'),
#         keras.layers.BatchNormalization(),
#         keras.layers.Dropout(0.3),
#         keras.layers.Dense(1024, activation='relu'),
#         keras.layers.BatchNormalization(),
#         keras.layers.Dropout(0.3),
#         keras.layers.Dense(5, activation='softmax')
#     ])
#
#     early_stop = EarlyStopping(
#         monitor='val_loss',  # What to monitor
#         patience=100,  # How many epochs to wait for improvement
#         restore_best_weights=True  # Optional: go back to the best weights
#     )
#
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),  # an optimizer - the mechanism through which the model will update itself
#                   loss="sparse_categorical_crossentropy",
#                   # loss function - how the model will be able to measure its performance on the training data
#                   metrics=["accuracy"])  # metric to monitor during training and testing
#
#     # Reshape the data to fit the model input
#     with tf.device('/GPU:0'):
#         history = model.fit(data_frame_list, validation_frame_list, validation_split=0.3, epochs=400,
#                         batch_size=32, callbacks=[early_stop])
#     history_dict = history.history
#     history_dict.keys()
#     print(model.summary())
#     model.save(datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '-myRSSIModel.keras')
#
#     history_dict = history.history
#     loss_values = history_dict["loss"]
#     # val_loss_values = history_dict["val_loss"]
#     epochs = range(1, len(loss_values) + 1)
#     plt.plot(epochs, loss_values, "bo", label="Training loss")
#     # plt.plot(epochs, val_loss_values, "b", label="Validation loss")
#     plt.title("Training and validation loss")
#     plt.xlabel("Epochs")
#     plt.ylabel("Loss")
#     plt.legend()
#     plt.show()
#
#     plt.clf()
#     acc = history_dict["accuracy"]
#     # val_acc = history_dict["val_accuracy"]
#     plt.plot(epochs, acc, "b", label="Training acc")
#     # plt.plot(epochs, val_acc, "b", label="Validation acc")
#     plt.title("Training and validation accuracy")
#     plt.xlabel("Epochs")
#     plt.ylabel("Accuracy")
#     plt.legend()
#     plt.show()
#
#     test_loss, test_accuracy = model.evaluate(data_frame_list, validation_frame_list)
#     print(f"test_accuracy: {test_accuracy}")
#
#     # from sklearn.metrics import classification_report, confusion_matrix
#     #
#     # pred = model.predict(X_test)
#     # print(classification_report(y_test, np.argmax(pred, axis=1)))
