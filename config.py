SENSOR_COUNT = 3
PULLING_RATE = 1
FRAME_TIME = 120

DATA_CACHE_FILE = "cache/query_results.csv"
VALIDATION_CACHE_FILE = "cache/validation_data.csv"

INFLUX_BUCKET = "RSSI_DATA"
ORG = "User"
TOKEN = "s3i2RyDAbl_wEt1svvqQ6liyvh7OyHjo9l_LUEW58JScQLDaDFqlZHYMtT-ry2PnfZJ00omoVoIvQx52ryXYdA=="
URL = "http://192.168.2.30:8086"

DATA_START = "2025-06-03T11:59:59Z"
DATA_STOP = "2025-06-06T01:00:00Z"

# NPERSEG is used to determine the number of segments in the STFT
# don't change if you don't know what your doing
NPERSEG = 15
OVERLAP = 10

# Training parameters
EPOCHS = 300
BATCH_SIZE = 32
PATIENCE = 80
FOLDS = 15

# visualising parameters
PLOT_START_TIME = 0
PLOT_END_TIME = 500

# PRESENT GONE ONLY, ENERTING LEAVING ONLY, VISITING ONY,
