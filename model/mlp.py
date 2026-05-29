from typing import Dict, Tuple
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import warnings
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.experimental import enable_halving_search_cv
# after scaling targets model converges without warning
#from sklearn.exceptions import ConvergenceWarning
#warnings.filterwarnings("ignore", category=ConvergenceWarning)
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, median_absolute_error, mean_absolute_percentage_error

from preproc.xy import split_sequence, split_seq_xy_pipe, normalize_sequence_uni, NormType
from constants import RANDOM_STATE, MAX_ITER, CV_FOLDS, TS_MIN_SEQUENCE_LEN, TS_MAX_SEQUENCE_LEN, TEST_SIZE


def init_mlp_uni_reg(max_iter: int=MAX_ITER, verbose: bool=False, random_state: int=RANDOM_STATE) -> MLPRegressor:
    mlp_reg = MLPRegressor(loss="squared_error", hidden_layer_sizes=(150,), activation='relu', learning_rate_init=0.001,
                           shuffle=False, verbose=verbose, max_iter=max_iter, random_state=random_state)
    return mlp_reg


def train_mlp_uni_reg(mlp_reg: MLPRegressor, X_train: np.ndarray, y_train: np.ndarray, param_distr: dict=None,
                      cv: int=CV_FOLDS, verbose: int=0, random_state: int=RANDOM_STATE) -> MLPRegressor:
    if param_distr is None:
        mlp_reg.fit(X_train, y_train)
        return mlp_reg
    else:
        # HalvingRandomSearchCV iteratively increases the resource (data n_samples by default) to fit with CV
        #   while also decreases the amount of candidates at each step
        # with 'refit' returns instance of model fitted with best params
        search = HalvingRandomSearchCV(mlp_reg, param_distributions=param_distr, n_candidates="exhaust", factor=1.5, refit=True,
                                       scoring="neg_root_mean_squared_error", cv=cv, random_state=random_state, n_jobs=2, verbose=verbose,
                                       ).fit(X_train, y_train)
        return search.best_estimator_


def outer_seq_len_search(init_mlp_reg: MLPRegressor, val_metric: str, ts_seq: np.ndarray, norm_type: NormType=NormType.NoNorm,
                         test_size: float=TEST_SIZE, min_len: int=TS_MIN_SEQUENCE_LEN, max_len: int=TS_MAX_SEQUENCE_LEN,
                         scale_y: bool=True, param_distr: dict=None, cv: int=CV_FOLDS, verbose: int=0, random_state: int=RANDOM_STATE
                         ) -> Tuple[MLPRegressor, int]:
    """Training multiple MLP to find best sequence length for prediction"""

    best_val_error = float("inf")
    best_mlp_reg = None
    best_seq_len = -1
    for seq_len in range(min_len, max_len + 1):
        X_train, X_test, y_train, y_test, _, _ = split_seq_xy_pipe(ts_seq, seq_len, test_size, norm_type, scale_y, random_state)
        trained_mlp_reg = train_mlp_uni_reg(init_mlp_reg, X_train, y_train, param_distr, cv, verbose, random_state)
        target_error_metric = calc_metrics_mlp_uni_reg(trained_mlp_reg, X_test, y_test)[val_metric]

        if target_error_metric < best_val_error:
            best_val_error = target_error_metric
            best_mlp_reg = trained_mlp_reg
            best_seq_len = seq_len
            print(f"New best '{val_metric}': {best_val_error:.6f} with sequence_len {best_seq_len}")

    return best_mlp_reg, best_seq_len


def fit_mlp_uni_reg(mlp_reg: MLPRegressor, ts_seq: np.ndarray, seq_len: int, norm_type: NormType=NormType.NoNorm,
                    scale_y: bool=True) -> Tuple[MLPRegressor, TransformerMixin, TransformerMixin]:
    """Fit the best trained model on all data for future predictions"""
    X, y = split_sequence(ts_seq, seq_len)
    X, y, X_scaler, y_scaler = normalize_sequence_uni(X, y, norm_type, scale_y)
    mlp_reg.fit(X, y)
    return mlp_reg, X_scaler, y_scaler


def predict_uni_next_price(mlp_reg: MLPRegressor, ts_seq: np.ndarray, seq_len: int,
                           X_scaler: TransformerMixin, y_scaler: TransformerMixin=None) -> float:
    """Predict next price with fit model and its sequence len"""
    if len(ts_seq) < seq_len:
        raise ValueError(f"Need a sequence with len >={seq_len} for prediction")
    if ((X_scaler is not None) and (not hasattr(X_scaler, "transform")) or
        (y_scaler is not None) and (not hasattr(y_scaler, "transform"))):
        raise ValueError(f"Invalid scaler object")

    input_seq = ts_seq[-seq_len:].reshape(1, -1)
    if X_scaler is not None:
        input_seq = X_scaler.transform(input_seq)

    pred = mlp_reg.predict(input_seq)[0]
    if y_scaler is not None:
        pred = y_scaler.inverse_transform(pred.reshape(-1, 1)).reshape(-1)[0]

    return pred


def log_metrics(metrics: Dict[str, float], split: str="train"):
    print()
    for metric, value in metrics.items():
        print(f"{metric}: {split} {value:.6f}")


def calc_metrics_mlp_uni_reg(mlp_reg: MLPRegressor, X: np.ndarray, y: np.ndarray, y_scaler: TransformerMixin=None) -> Dict[str, float]:

    metrics = {}
    preds = mlp_reg.predict(X)
    if y_scaler is not None:
        y = y_scaler.inverse_transform(y.reshape(-1, 1)).reshape(-1)
        preds = y_scaler.inverse_transform(preds.reshape(-1, 1)).reshape(-1)

    metrics["R2 Score"] = mlp_reg.score(X, y)
    metrics["Root Mean Squared Error"] = root_mean_squared_error(y, preds)
    metrics["Mean Absolute Error"] = mean_absolute_error(y, preds)
    metrics["Mean Absolute Percentage Error"] = mean_absolute_percentage_error(y, preds)
    metrics["Median Absolute Error"] = median_absolute_error(y, preds)

    return metrics
