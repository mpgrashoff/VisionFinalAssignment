import os
import pandas as pd
from influxdb_client import InfluxDBClient
from config import *


def load_data_from_influx(file_name, query):
    if os.path.exists(file_name):
        print(f"Loading cached data from {file_name}")
        return pd.read_csv(file_name)

    with InfluxDBClient(url=URL, token=TOKEN, org=ORG) as client:
        query_api = client.query_api()
        result = query_api.query_data_frame(org=ORG, query=query)
        if result is not None and not result.empty:
            result.to_csv(file_name, index=False)
        return result
