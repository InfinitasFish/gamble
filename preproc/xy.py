from typing import Tuple, Optional
from enum import Enum
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit

from strats.ma import get_exponential_moving_average
from constants import TS_MAX_SEQUENCE_LEN, CACHE_DIR_FPATH, CANDLES_MULTI_TRAINING_FEATURES, CANDLES_UNI_TARGET_FEATURE, \
    TEST_SIZE, RANDOM_STATE, TS_MIN_SEQUENCE_LEN
from data.candles_tink import get_candles_data_consecutive, get_tcandles_df


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

    # have to reshape, because mlp and scaler accept [ndim <= 2]
    X_seqs = np.squeeze(np.array(X_seqs)).reshape(len(X_seqs), -1)
    y_for_seqs = np.squeeze(np.array(y_for_seqs))
    return X_seqs, y_for_seqs


def denoise_xy_features_wma(X: np.ndarray, y: np.ndarray=None, window: int=TS_MIN_SEQUENCE_LEN) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    X_ = np.copy(X)

    for i in range(X.shape[1]):
        xi_ma = get_exponential_moving_average(X[:, i], window)
        X[:, i] = np.array(xi_ma)

    if y is not None:
        y_ = np.copy(y)
        y_[:, 0] = get_exponential_moving_average(y[:, 0], window)
        return X_, y_

    return X_, None


def split_seq_xy_pipe(X: np.ndarray, y: np.ndarray, seq_len: int, test_size: float=TEST_SIZE,
                      norm_type: NormType=NormType.NoNorm, scale_y: bool=True, random_state: int=RANDOM_STATE
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, TransformerMixin, TransformerMixin]:

    X_train, X_test, y_train, y_test = train_test_split_thresh(X, y, test_size=test_size)
    X_train, X_test, y_train, y_test, X_scaler, y_scaler = normalize_splits_uni(X_train, X_test, y_train, y_test, norm_type, scale_y)

    # making n-len sequences after normalization
    X_train, y_train = split_xy_to_sequences(X_train, y_train, seq_len)
    X_test, y_test = split_xy_to_sequences(X_test, y_test, seq_len)
    return X_train, X_test, y_train, y_test, X_scaler, y_scaler


def train_test_split_thresh(X: np.ndarray, y: np.ndarray, test_size: float
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    assert X.shape[0] == y.shape[0]
    train_idx_border = int(X.shape[0] * (1 - test_size))
    X_train, y_train = X[:train_idx_border], y[:train_idx_border]
    X_test, y_test = X[train_idx_border:], y[train_idx_border:]

    return X_train, X_test, y_train, y_test


def target_log_transform(feature_col: pd.Series) -> pd.Series:
    transformed = np.log(feature_col.shift(-1) / feature_col)
    return transformed


def get_candles_xy(from_iso: str, to_iso: str, instrument_id: str, interval: str= "CANDLE_INTERVAL_DAY",
                   train_features: list=CANDLES_MULTI_TRAINING_FEATURES, target_features: list=CANDLES_UNI_TARGET_FEATURE,
                   cache_fpath: str=CACHE_DIR_FPATH, to_cache: bool=False) -> Tuple[np.ndarray, np.ndarray]:

    candles_data = get_candles_data_consecutive(from_iso, to_iso, instrument_id, interval, cache_fpath, to_cache)
    candles_df = get_tcandles_df(candles_data, train_features)

    target_features_df = []
    for feature in target_features:
        target_features_df.append(f"target_{feature}")
        candles_df[f"target_{feature}"] = target_log_transform(candles_df[feature])
    candles_df = candles_df.dropna()

    X = np.array(candles_df[train_features])
    y = np.array(candles_df[target_features_df])
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
        else:
            y_scaler = None

    return X, y, X_scaler, y_scaler


def normalize_splits_uni(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
                         norm_type: NormType=NormType.NoNorm, scale_y: bool=False,
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

