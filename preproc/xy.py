from typing import Tuple
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from enum import Enum
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from constants import TS_MAX_SEQUENCE_LEN, CACHE_DIR_FPATH, CANDLES_MULTI_TRAINING_FEATURES, CANDLES_UNI_TARGET_FEATURE, TEST_SIZE, RANDOM_STATE
from data.candles import get_candles_data, get_candles_df


class NormType(Enum):
    NoNorm = 0,
    Standardize = 1,
    MinMax = 2,


def split_xy_to_sequences(X: np.ndarray, y: np.ndarray, n_steps: int=TS_MAX_SEQUENCE_LEN) -> Tuple[np.ndarray, np.ndarray]:
    X_seqs = list()
    y_for_seqs = list()
    assert X.shape[0] == y.shape[0]

    for i in range(X.shape[0] - n_steps):
        X_seqs.append(X[i:(i + n_steps)])
        y_for_seqs.append(y[i + n_steps - 1])

    # have to reshape
    X_seqs = np.squeeze(np.array(X_seqs)).reshape(len(X_seqs), -1)
    y_for_seqs = np.squeeze(np.array(y_for_seqs))
    return X_seqs, y_for_seqs


def split_seq_xy_pipe(X: np.ndarray, y: np.ndarray, seq_len: int, temporal: bool=True, test_size: float=TEST_SIZE,
                      norm_type: NormType=NormType.NoNorm, scale_y: bool=True, random_state: int=RANDOM_STATE
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, TransformerMixin, TransformerMixin]:

    if temporal:
        X_train, X_test, y_train, y_test = train_test_split_temporal(X, y, test_size=test_size)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    X_train, X_test, y_train, y_test, X_scaler, y_scaler = normalize_splits_uni(X_train, X_test, y_train, y_test, norm_type, scale_y)

    # making n-len sequences after normalization
    X_train, y_train = split_xy_to_sequences(X_train, y_train, seq_len)
    X_test, y_test = split_xy_to_sequences(X_test, y_test, seq_len)
    return X_train, X_test, y_train, y_test, X_scaler, y_scaler


def train_test_split_temporal(X: np.ndarray, y: np.ndarray, test_size: float
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    assert X.shape[0] == y.shape[0]
    train_idx_border = int(X.shape[0] * (1 - test_size))
    X_train, y_train = X[:train_idx_border], y[:train_idx_border]
    X_test, y_test = X[train_idx_border:], y[train_idx_border:]


    return X_train, X_test, y_train, y_test


def get_candles_xy(from_utc: str, to_utc: str, instrument_id: str, interval: str= "CANDLE_INTERVAL_DAY",
                   train_features: list=CANDLES_MULTI_TRAINING_FEATURES, target_features: list=CANDLES_UNI_TARGET_FEATURE,
                   cache_fpath: str=CACHE_DIR_FPATH, to_cache: bool=False) -> Tuple[np.ndarray, np.ndarray]:

    candles_data = get_candles_data(from_utc, to_utc, instrument_id, interval, cache_fpath, to_cache)
    candles_df = get_candles_df(candles_data, train_features)

    target_features_df = []
    for feature in target_features:
        target_features_df.append(f"target_{feature}")
        candles_df[f"target_{feature}"] = candles_df[feature].shift(-1)
    candles_df = candles_df.dropna()

    X = candles_df[train_features].to_numpy()
    y = candles_df[target_features_df].to_numpy()
    return X, y


def normalize_sequence_uni(X: np.ndarray, y: np.ndarray, norm_type: NormType=NormType.NoNorm, scale_y: bool=True,
                           ) -> Tuple[np.ndarray, np.ndarray, TransformerMixin, TransformerMixin]:

    X_scaler = None
    y_scaler = None
    if norm_type != NormType.NoNorm:
        match norm_type:
            case NormType.Standardize:
                X_scaler = StandardScaler()
                y_scaler = StandardScaler()
            case NormType.MinMax:
                X_scaler = MinMaxScaler()
                y_scaler = MinMaxScaler()
            case _:
                raise ValueError("Unknown type of Normalization is given")
        X = X_scaler.fit_transform(X)

        if scale_y:
            y = y_scaler.fit_transform(y)

    return X, y, X_scaler, y_scaler


def normalize_splits_uni(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
                         norm_type: NormType=NormType.NoNorm, scale_y: bool=True,
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, TransformerMixin, TransformerMixin]:
    
    X_scaler = None
    y_scaler = None
    if norm_type != NormType.NoNorm:
        match norm_type:
            case NormType.Standardize:
                X_scaler = StandardScaler()
                y_scaler = StandardScaler()
            case NormType.MinMax:
                X_scaler = MinMaxScaler()
                y_scaler = MinMaxScaler()
            case _:
                raise ValueError("Unknown type of Normalization is given")
        X_train = X_scaler.fit_transform(X_train)
        X_test = X_scaler.transform(X_test)

        if scale_y:
            y_train = y_scaler.fit_transform(y_train)
            y_test = y_scaler.transform(y_test)
        else:
            y_scaler = None

    return X_train, X_test, y_train, y_test, X_scaler, y_scaler

