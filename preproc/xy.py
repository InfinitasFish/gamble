from typing import Tuple
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from enum import Enum
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from constants import TS_MAX_SEQUENCE_LEN, CACHE_DIR_FPATH, CANDLES_UNI_FEATURE, TEST_SIZE, RANDOM_STATE
from data.candles import get_candles_data, get_candles_df


class NormType(Enum):
    NoNorm = 0,
    Standardize = 1,
    MinMax = 2,


def split_sequence(sequence: list | np.ndarray, n_steps: int=TS_MAX_SEQUENCE_LEN) -> Tuple[np.ndarray, np.ndarray]:
    X = list()
    y = list()
    for i in range(len(sequence) - n_steps):
        X.append(sequence[i:(i + n_steps)])
        y.append(sequence[i + n_steps])

    X = np.squeeze(np.array(X), 2)
    y = np.squeeze(np.array(y), 1)
    return X, y


# TODO: random train_test_split is kinda trash for time-series, use appropriate strategy
def split_seq_xy_pipe(flat_sequence: np.ndarray, sequence_len: int=TS_MAX_SEQUENCE_LEN, test_size: float=TEST_SIZE,
                      norm_type: NormType=NormType.NoNorm, random_state: int=RANDOM_STATE
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, TransformerMixin]:

    X, y = split_sequence(flat_sequence, sequence_len)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_test, y_train, y_test, scaler = normalize_splits_uni(X_train, X_test, y_train, y_test, norm_type)
    return X_train, X_test, y_train, y_test, scaler


def get_candles_seq_uni(from_utc: str, to_utc: str, instrument_id: str, interval: str="CANDLE_INTERVAL_DAY",
                        cache_fpath: str=CACHE_DIR_FPATH, to_cache: bool=False) -> np.ndarray:
    candles_data = get_candles_data(from_utc, to_utc, instrument_id, interval, cache_fpath, to_cache)
    candles_df = get_candles_df(candles_data, CANDLES_UNI_FEATURE)
    sequence = candles_df.to_numpy().flatten().reshape(-1, 1)
    return sequence


def normalize_sequence_uni(X: np.ndarray, y: np.ndarray, norm_type: NormType=NormType.NoNorm
                           ) -> Tuple[np.ndarray, np.ndarray, TransformerMixin]:

    scaler = None
    if norm_type != NormType.NoNorm:
        match norm_type:
            case NormType.Standardize:
                scaler = StandardScaler()
            case NormType.MinMax:
                scaler = MinMaxScaler()
            case _:
                raise ValueError("Unknown type of Normalization is given")
        X = scaler.fit_transform(X)

    return X, y, scaler


def normalize_splits_uni(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
                         norm_type: NormType=NormType.NoNorm
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, TransformerMixin]:

    # TODO: maybe it's a good idea to scale (not norm) targets
    # https://datascience.stackexchange.com/questions/35603/it-is-helpful-to-normalize-target-variables-for-a-regression-neural-network
    
    scaler = None
    if norm_type != NormType.NoNorm:
        match norm_type:
            case NormType.Standardize:
                scaler = StandardScaler()
            case NormType.MinMax:
                scaler = MinMaxScaler()
            case _:
                raise ValueError("Unknown type of Normalization is given")
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler

