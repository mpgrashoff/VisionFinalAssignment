import os
import pandas as pd
from influxdb_client import InfluxDBClient
from config import *


def load_data_from_influx(file_name, query):
    """
    Loads data from InfluxDB or from a cached CSV file if it exists.

    If the specified cache file exists, the function reads from it using pandas.
    Otherwise, it queries the InfluxDB server using the provided Flux query,
    caches the result to a CSV file, and returns the DataFrame.

    Args:
        file_name (str): Path to the CSV cache file.
        query (str): Flux query string to fetch data from InfluxDB.

    Returns:
        pd.DataFrame: DataFrame containing the queried or cached data.
                      Returns an empty DataFrame if the query yields no result.
    """
    if os.path.exists(file_name):
        print(f"Loading cached data from {file_name}")
        return pd.read_csv(file_name)

    with InfluxDBClient(url=URL, token=TOKEN, org=ORG) as client:
        query_api = client.query_api()
        result = query_api.query_data_frame(org=ORG, query=query)
        if result is not None and not result.empty:
            result.to_csv(file_name, index=False)
        return result
