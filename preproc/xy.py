from typing import Tuple
import os
import sys

from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import numpy as np
from sklearn.preprocessing import StandardScaler

from constants import TS_SEQUENCE_LEN, CACHE_FPATH, CANDLES_UNI_FEATURE, TEST_SIZE, RANDOM_STATE
from data.candles import get_candles_data, get_candles_df


def split_sequence(sequence: list | np.ndarray, n_steps: int=TS_SEQUENCE_LEN) -> Tuple[np.ndarray, np.ndarray]:
    X = list()
    y = list()
    for i in range(len(sequence) - n_steps):
        X.append(sequence[i:(i + n_steps)])
        y.append(sequence[i + n_steps])

    return np.array(X), np.array(y)


def get_candles_uni_xy_pipe(from_utc: str, to_utc: str, instrument_id: str, normalize: bool=False, test_size: float=TEST_SIZE,
                            interval: str="CANDLE_INTERVAL_DAY", sequence_len: int=TS_SEQUENCE_LEN, random_state: int = RANDOM_STATE,
                            cache_fpath: str=CACHE_FPATH, to_cache: bool=False
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    candles_data = get_candles_data(from_utc, to_utc, instrument_id, interval, cache_fpath, to_cache)
    candles_df = get_candles_df(candles_data, CANDLES_UNI_FEATURE)
    sequence = candles_df.to_numpy().flatten().reshape(-1, 1)

    if normalize:
        sc = StandardScaler()
        sequence = sc.fit_transform(sequence).reshape(-1)
    else:
        sequence = sequence.reshape(-1)
    X, y = split_sequence(sequence, sequence_len)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
