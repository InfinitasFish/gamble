from typing import Tuple
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from constants import TS_MAX_SEQUENCE_LEN, CACHE_DIR_FPATH, CANDLES_UNI_FEATURE, TEST_SIZE, RANDOM_STATE
from data.candles import get_candles_data, get_candles_df


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


def get_candles_seq_uni_pipe(from_utc: str, to_utc: str, instrument_id: str, normalize: bool=False,
                             interval: str="CANDLE_INTERVAL_DAY", cache_fpath: str=CACHE_DIR_FPATH, to_cache: bool=False
                             ) -> np.ndarray:

    candles_data = get_candles_data(from_utc, to_utc, instrument_id, interval, cache_fpath, to_cache)
    candles_df = get_candles_df(candles_data, CANDLES_UNI_FEATURE)
    sequence = candles_df.to_numpy().flatten().reshape(-1, 1)

    if normalize:
        sc = StandardScaler()
        sequence = sc.fit_transform(sequence).reshape(-1)
    else:
        sequence = sequence.reshape(-1)

    return sequence
