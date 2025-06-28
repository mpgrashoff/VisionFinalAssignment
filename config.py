# ─────────────────────────────────────────────────────────────────────────────
# Data Collection Parameters
# ⚠️ Do not change
# ─────────────────────────────────────────────────────────────────────────────
SENSOR_COUNT = 3              # Number of RSSI sensors collecting data
PULLING_RATE = 1              # Sampling rate in Hz (1 sample per second)
FRAME_TIME = 120              # Frame length in seconds for model input (120 samples)

# ─────────────────────────────────────────────────────────────────────────────
# Data Cache Files
# ─────────────────────────────────────────────────────────────────────────────
DATA_CACHE_FILE = "cache/query_results.csv"         # Path to training data cache
VALIDATION_CACHE_FILE = "cache/validation_data.csv" # Path to validation data cache

# ─────────────────────────────────────────────────────────────────────────────
# InfluxDB Configuration
# ─────────────────────────────────────────────────────────────────────────────
INFLUX_BUCKET = "RSSI_DATA"
ORG = "User"
TOKEN = "s3i2RyDAbl_wEt1svvqQ6liyvh7OyHjo9l_LUEW58JScQLDaDFqlZHYMtT-ry2PnfZJ00omoVoIvQx52ryXYdA=="
URL = "http://192.168.2.30:8086"

DATA_START = "2025-06-03T11:59:59Z"  # ISO 8601 start time of data window
DATA_STOP  = "2025-06-06T01:00:00Z"  # ISO 8601 stop time of data window

# ─────────────────────────────────────────────────────────────────────────────
# Spectrogram Parameters
# ─────────────────────────────────────────────────────────────────────────────
NPERSEG = 15   # Number of samples per segment in STFT
OVERLAP = 10   # Number of overlapping samples between segments

# ⚠️ Do not change NPERSEG or OVERLAP unless you understand STFT internals

# ─────────────────────────────────────────────────────────────────────────────
# Training Parameters
# ─────────────────────────────────────────────────────────────────────────────
EPOCHS = 300
BATCH_SIZE = 32
PATIENCE = 80       # Early stopping patience
FOLDS = 15          # Number of cross-validation folds

# ─────────────────────────────────────────────────────────────────────────────
# Visualization Parameters
# ─────────────────────────────────────────────────────────────────────────────
PLOT_START_TIME = 1000   # Start index for plotting
PLOT_END_TIME = 2300     # End index for plotting

# ─────────────────────────────────────────────────────────────────────────────
# Classification Modes IN DEVELOPMENT (for developer reference)
# ─────────────────────────────────────────────────────────────────────────────
# PRESENT_GONE_ONLY      → Binary presence detection
# ENTERING_LEAVING_ONLY → Entry/exit event classification
# VISITING_ONLY          → Detecting visit patterns or durations
