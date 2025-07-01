import pandas as pd


def classify_boolean_frame(series):
    """
    classifies the boolean series into one of the following classes:
    - 0: absence
    - 1: presence
    - 2: rising presence ie entering
    - 3: falling presence ie leaving
    - 4: both rising and falling presence ie entering and leaving
    - -1: unknown state (should not happen)
    Args:
        series : np.ndarray of bools with data rate of PULLING_RATE

    Returns:
        Int classification of the series:
    """
    s = pd.Series(series).astype(bool)
    diff = s.astype(int).diff().fillna(0)
    if s.sum() == 0:
        return 0
    elif s.sum() == len(s):
        return 1
    has_rising = (diff == 1).any()
    has_falling = (diff == -1).any()
    if has_rising and has_falling:
        return 4
    elif has_rising:
        return 2
    elif has_falling:
        return 3
    return -1
