from typing import Tuple
import os
import sys

from matplotlib.patheffects import Normal
from sympy import sequence

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from enum import Enum
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from constants import TS_MAX_SEQUENCE_LEN, CACHE_DIR_FPATH, CANDLES_UNI_FEATURE, TEST_SIZE, RANDOM_STATE
from data.candles import get_candles_data, get_candles_df


class NormalizationType(Enum):
    NoNormalization = 0,
    Standardize = 1,
    MinMax = 2,


def split_sequence(sequence: list | np.ndarray, n_steps: int=TS_MAX_SEQUENCE_LEN) -> Tuple[np.ndarray, np.ndarray]:
    X = list()
    y = list()
    for i in range(len(sequence) - n_steps):
        X.append(sequence[i:(i + n_steps)])
        y.append(sequence[i + n_steps])

    return np.array(X), np.array(y)


def split_seq_xy_pipe(flat_sequence: np.ndarray, sequence_len: int=TS_MAX_SEQUENCE_LEN, test_size: float=TEST_SIZE,
                  random_state: int=RANDOM_STATE) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    X, y = split_sequence(flat_sequence, sequence_len)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def get_candles_seq_uni(from_utc: str, to_utc: str, instrument_id: str, interval: str="CANDLE_INTERVAL_DAY",
                        cache_fpath: str=CACHE_DIR_FPATH, to_cache: bool=False) -> np.ndarray:
    candles_data = get_candles_data(from_utc, to_utc, instrument_id, interval, cache_fpath, to_cache)
    candles_df = get_candles_df(candles_data, CANDLES_UNI_FEATURE)
    sequence = candles_df.to_numpy().flatten().reshape(-1, 1)
    return sequence


def normalize_seq_uni(ts_data: np.ndarray, norm_type: NormalizationType=NormalizationType.Standardize) -> np.ndarray:
    match norm_type:
        case NormalizationType.NoNormalization:
            sequence = ts_data.reshape(-1)
        case NormalizationType.Standardize:
            sc = StandardScaler()
            sequence = sc.fit_transform(ts_data).reshape(-1)
        case NormalizationType.MinMax:
            mm = MinMaxScaler()
            sequence = mm.fit_transform(ts_data).reshape(-1)
        case _:
            raise ValueError("Unknown type of Normalization is given")

    return sequence
