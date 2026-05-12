# ma - moving averages
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from constants import TS_MIN_SEQUENCE_LEN


def get_simple_moving_average(ts_data: np.ndarray | pd.Series, window: int=TS_MIN_SEQUENCE_LEN) -> np.ndarray:
    if isinstance(ts_data, pd.Series):
        return ts_data.rolling(window=window).mean().to_numpy()
    elif isinstance(ts_data, np.ndarray):
        return pd.Series(ts_data).rolling(window=window).mean().to_numpy()
    else:
        raise ValueError(f"Expected (np.ndarray | pd.Series) type for ts_data")


def get_simple_moving_average_why(ts_data: np.ndarray | pd.Series, window: int=TS_MIN_SEQUENCE_LEN) -> np.ndarray:
    # just because I want to
    ma = [np.nan] * (window - 1)
    ts_data = np.array(ts_data)
    for i in range(len(ts_data) - window + 1):
        ma.append(ts_data[i:(i + window)].mean())
    return np.array(ma)


def get_weighted_moving_average(ts_data: np.ndarray | pd.Series, window: int=TS_MIN_SEQUENCE_LEN, weights=None) -> np.ndarray:
    if weights is None:
        weights = np.arange(1, window + 1)
    else:
        assert len(weights) == window

    if isinstance(ts_data, pd.Series):
        return ts_data.rolling(window=window).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True).to_numpy()
    elif isinstance(ts_data, np.ndarray):
        return pd.Series(ts_data).rolling(window=window).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True).to_numpy()
    else:
        raise ValueError(f"Expected (np.ndarray | pd.Series) type for ts_data")


def get_weighted_moving_average_why(ts_data: np.ndarray | pd.Series, window: int=TS_MIN_SEQUENCE_LEN, weights=None) -> np.ndarray:
    # because I'm cool
    if weights is None:
        weights = np.arange(1, window + 1)
    else:
        assert len(weights) == window

    wma = [np.nan] * (window - 1)
    norm_weights = weights/sum(weights)
    ts_data = np.array(ts_data)
    for i in range(len(ts_data) - window + 1):
        wma.append((ts_data[i:(i + window)] * norm_weights).mean())
    return np.array(wma)


# most common for financials, also can be calculated at the start of sequence
def get_exponential_moving_average(ts_data: np.ndarray | pd.Series, window: int=TS_MIN_SEQUENCE_LEN) -> np.ndarray:
    if isinstance(ts_data, pd.Series):
        return ts_data.ewm(span=window, adjust=False).mean().to_numpy()
    elif isinstance(ts_data, np.ndarray):
        return pd.Series(ts_data).ewm(span=window, adjust=False).mean().to_numpy()
    else:
        raise ValueError(f"Expected (np.ndarray | pd.Series) type for ts_data")


if __name__ == "__main__":
    array = pd.Series(np.array([10, 8, 1, 2, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7, 8, 3, 2, 10, 9, 8, 3]))
    ma = get_exponential_moving_average(array, 5)
    myma = get_weighted_moving_average(array, 5)
    print(array.shape, ma.shape)
    print(myma.shape)
    print(ma[:10])
    print(myma[:10])
    plt.plot(array)
    plt.plot(ma)
    plt.plot(myma)
    plt.show()
