# ma - moving averages
from enum import Enum
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from constants import TS_MIN_SEQUENCE_LEN


class MAType(Enum):
    SimpleMA = 0,
    WeightedMA = 1,
    ExponentMA = 2,


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


def get_weighted_moving_average_why(ts_data: np.ndarray | pd.Series, window: int=TS_MIN_SEQUENCE_LEN, weights: list[int]=None) -> np.ndarray:
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


def plot_ma_for_timeseries(ts_data: np.ndarray | pd.Series, window: int=TS_MIN_SEQUENCE_LEN, weights: list[int]=None,
                           ma_type: MAType=MAType.SimpleMA):
    match ma_type:
        case MAType.SimpleMA:
            ma_label = "SMA"
            vals = get_simple_moving_average(ts_data, window)
        case MAType.WeightedMA:
            ma_label = "WMA"
            vals = get_weighted_moving_average(ts_data, window, weights)
        case MAType.ExponentMA:
            ma_label = "EMA"
            vals = get_exponential_moving_average(ts_data, window)
        case _:
            raise ValueError("Unknown type of Moving Average is given")

    plt.plot(np.arange(len(ts_data)), ts_data, label="Timeseries")
    plt.plot(np.arange(len(vals)), vals, label=f"{ma_label} {window}")
    plt.xlabel("time")
    plt.ylabel("values")
    plt.legend(loc="best")
    plt.show()


if __name__ == "__main__":
    from datetime import datetime
    from data.candles_tink import convert_datetime_api_format
    from preproc.xy import get_candles_xy
    from constants import YDEX_TICKER, CANDLES_UNI_TARGET_FEATURE

    from_iso = convert_datetime_api_format(datetime.fromisoformat("2024-01-01"))
    to_iso = convert_datetime_api_format(datetime.fromisoformat("2026-01-01"))
    X, y = get_candles_xy(from_iso, to_iso, YDEX_TICKER, target_features=CANDLES_UNI_TARGET_FEATURE, to_cache=True)
    y = y.reshape(-1)

    plot_ma_for_timeseries(y, window=10, ma_type=MAType.ExponentMA)
    plot_ma_for_timeseries(y, window=20, ma_type=MAType.ExponentMA)
    plot_ma_for_timeseries(y, window=50, ma_type=MAType.ExponentMA)
