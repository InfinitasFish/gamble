from typing import Dict, Tuple
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import warnings
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.experimental import enable_halving_search_cv
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
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
                         param_distr: dict=None, cv: int=CV_FOLDS, verbose: int=0, random_state: int=RANDOM_STATE
                         ) -> Tuple[MLPRegressor, int]:
    """Training multiple MLP to find best sequence length for prediction"""

    best_val_error = float("inf")
    best_mlp_reg = None
    best_seq_len = -1
    for seq_len in range(min_len, max_len + 1):
        X_train, X_test, y_train, y_test, _ = split_seq_xy_pipe(ts_seq, seq_len, test_size, norm_type, random_state)
        trained_mlp_reg = train_mlp_uni_reg(init_mlp_reg, X_train, y_train, param_distr, cv, verbose, random_state)
        target_error_metric = calc_metrics_mlp_uni_reg(trained_mlp_reg, X_test, y_test)[val_metric]["test"]

        if target_error_metric < best_val_error:
            best_val_error = target_error_metric
            best_mlp_reg = trained_mlp_reg
            best_seq_len = seq_len
            print(f"New best '{val_metric}': {best_val_error:.6f} with sequence_len {best_seq_len}")

    return best_mlp_reg, best_seq_len


def fit_mlp_uni_reg(mlp_reg: MLPRegressor, ts_seq: np.ndarray, seq_len: int, norm_type: NormType=NormType.NoNorm
                    ) -> Tuple[MLPRegressor, object]:
    """Fit the best trained model on all data for future predictions"""
    X, y = split_sequence(ts_seq, seq_len)
    X, y, scaler = normalize_sequence_uni(X, y, norm_type)
    mlp_reg.fit(X, y)
    return mlp_reg, scaler


def predict_uni_next_price(mlp_reg: MLPRegressor, ts_seq: np.ndarray, seq_len: int, scaler_object: object) -> float:
    """Predict next price with fit model and its sequence len"""
    if len(ts_seq) < seq_len:
        raise ValueError(f"Need a sequence with len >={seq_len} for prediction")
    if (scaler_object is not None) and (not hasattr(scaler_object, "transform")):
        raise ValueError(f"Invalid scaler object")

    input_seq = ts_seq[-seq_len:].reshape(1, -1)
    if scaler_object is not None:
        input_seq = scaler_object.transform(input_seq)

    pred = mlp_reg.predict(input_seq)[0]
    return pred


def log_metrics(metrics: Dict[str, Dict[str, float]]):
    """ metrics: {'m1': {'train': float, 'test': float}, 'm2': {...} } """
    for metric, splits in metrics.items():
        print(f"{metric}:")
        for split, val in splits.items():
            print(f"  - {split}: {val:.6f}")


# TODO: remove the difference between train and test please
def calc_metrics_mlp_uni_reg(mlp_reg: MLPRegressor, X_test: np.ndarray, y_test: np.ndarray,
                             X_train: np.ndarray=None, y_train: np.ndarray=None) -> Dict[str, Dict[str, float]]:

    metrics = {}
    preds_train = None
    preds_test = mlp_reg.predict(X_test)
    if X_train is not None and y_train is not None:
        preds_train = mlp_reg.predict(X_train)

    metrics["R2 Score"] = {"test": mlp_reg.score(X_test, y_test)}
    metrics["Root Mean Squared Error"] = {"test": root_mean_squared_error(y_test, preds_test)}
    metrics["Mean Absolute Error"] = {"test": mean_absolute_error(y_test, preds_test)}
    metrics["Mean Absolute Percentage Error"] = {"test": mean_absolute_percentage_error(y_test, preds_test)}
    metrics["Median Absolute Error"] = {"test": median_absolute_error(y_test, preds_test)}

    if preds_train is not None:
        metrics["R2 Score"]["train"] = mlp_reg.score(X_train, y_train)
        metrics["Root Mean Squared Error"]["train"] = root_mean_squared_error(y_train, preds_train)
        metrics["Mean Absolute Error"]["train"] = mean_absolute_error(y_train, preds_train)
        metrics["Mean Absolute Percentage Error"]["train"] = mean_absolute_percentage_error(y_train, preds_train)
        metrics["Median Absolute Error"]["train"] = median_absolute_error(y_train, preds_train)

    return metrics
