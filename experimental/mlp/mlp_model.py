from typing import Dict, Tuple
from collections import defaultdict
import warnings
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.experimental import enable_halving_search_cv
# after scaling targets experimental converges without warnings
# from sklearn.exceptions import ConvergenceWarning
# warnings.filterwarnings("ignore", category=ConvergenceWarning)
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import HalvingRandomSearchCV, TimeSeriesSplit
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, median_absolute_error, mean_absolute_percentage_error, r2_score

from preproc.xy import split_xy_to_sequences, split_seq_xy_pipe, normalize_sequence_uni, NormType, normalize_splits_uni
from constants import RANDOM_STATE, MAX_ITER, INNER_CV_FOLDS, OUTER_CV_FOLDS

# TODO: there are specific mlp for timeseries, consider implementing, e.g. https://arxiv.org/pdf/2311.06184


def init_mlp_uni_reg(max_iter: int=MAX_ITER, verbose: bool=False, random_state: int=RANDOM_STATE) -> MLPRegressor:
    mlp_reg = MLPRegressor(loss="squared_error", hidden_layer_sizes=(150,), activation='relu', learning_rate_init=0.001,
                           shuffle=False, verbose=verbose, max_iter=max_iter, random_state=random_state)
    return mlp_reg


# nested cv would be good here - outer cv for train/test splitting, and inner cv for param tuning on train
# average test metrics on outer splits - and get something very close to real test metrics
# nested cv lets us incorporate all the data in the single pipeline while avoiding any biases on test metrics
def outer_train_mlp_uni_reg(X_cand: np.ndarray, y_cand: np.ndarray, seq_len: int, param_distr: dict=None,
                            outer_cv: int=OUTER_CV_FOLDS, inner_cv: int=INNER_CV_FOLDS, norm_type: NormType=NormType.NoNorm,
                            scale_y: bool=False, scoring: str="neg_root_mean_squared_error", verbose: int=0,
                            random_state: int=RANDOM_STATE) -> dict[str, float]:

    kfold = TimeSeriesSplit(n_splits=outer_cv)
    test_metrics = defaultdict(list)
    for i, (train_index, test_index) in enumerate(kfold.split(X_cand)):

        X_train, X_test, y_train, y_test = X_cand[train_index], X_cand[test_index], y_cand[train_index], y_cand[test_index]
        X_train, X_test, y_train, y_test, X_scaler, y_scaler = normalize_splits_uni(X_train, X_test, y_train, y_test,
                                                                                    norm_type, scale_y)

        # making n-len sequences after normalization
        X_train, y_train = split_xy_to_sequences(X_train, y_train, seq_len)
        X_test, y_test = split_xy_to_sequences(X_test, y_test, seq_len)

        mlp_i = inner_train_mlp_uni_reg(init_mlp_uni_reg(), X_train, y_train, param_distr,
                                        inner_cv, scoring, verbose, random_state)
        test_metrics_i = calc_metrics_mlp_uni_reg(mlp_i, X_test, y_test, y_scaler)
        for k, v in test_metrics_i.items():
            test_metrics[k].append(v)

    # calculating average metrics on outer test splits
    agg_metrics = {}
    for k, v in test_metrics.items():
        agg_metrics[k] = sum(v) / len(v)
    return agg_metrics


def inner_train_mlp_uni_reg(mlp_reg: MLPRegressor, X_train: np.ndarray, y_train: np.ndarray, param_distr: dict=None,
                            cv: int=INNER_CV_FOLDS, scoring: str= "neg_root_mean_squared_error", verbose: int=0,
                            random_state: int=RANDOM_STATE) -> MLPRegressor:

    if y_train.ndim == 2 and y_train.shape[1] == 1:
        y_train = y_train.reshape(-1)

    if param_distr is None:
        mlp_reg.fit(X_train, y_train)
        return mlp_reg
    else:
        # HalvingRandomSearchCV iteratively increases the resource (data n_samples by default) to fit with CV
        #   while also decreases the amount of candidates at each step
        # with 'refit' returns instance of experimental fitted with best params
        ts_cv = TimeSeriesSplit(n_splits=cv)
        search = HalvingRandomSearchCV(mlp_reg, param_distributions=param_distr, n_candidates="exhaust", factor=1.5, refit=True,
                                       scoring=scoring, cv=ts_cv, random_state=random_state, n_jobs=2, verbose=verbose,
                                       ).fit(X_train, y_train)
        return search.best_estimator_


def fit_mlp_uni_reg(mlp_reg: MLPRegressor, X: np.ndarray, y: np.ndarray, seq_len: int, norm_type: NormType=NormType.NoNorm,
                    scale_y: bool=True) -> Tuple[MLPRegressor, TransformerMixin, TransformerMixin]:
    """Fit the best trained experimental on all data for future predictions"""

    X, y, X_scaler, y_scaler = normalize_sequence_uni(X, y, norm_type, scale_y)
    X, y = split_xy_to_sequences(X, y, seq_len)

    mlp_reg.fit(X, y)

    return mlp_reg, X_scaler, y_scaler


def predict_next_prices(mlp_reg: MLPRegressor, X: np.ndarray, seq_len: int, y_scaler: TransformerMixin=None) -> np.ndarray:
    """Predict next price with fit experimental and its sequence len"""
    if X.shape[0] < seq_len:
        raise ValueError(f"Need a sequence with len >={seq_len} for prediction")
    if (y_scaler is not None) and (not hasattr(y_scaler, "transform")):
        raise ValueError(f"Invalid scaler object")

    input_seq = X[-seq_len:, :]
    preds = mlp_reg.predict(input_seq)
    if y_scaler is not None:
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        preds = y_scaler.inverse_transform(preds)

    if preds.ndim == 2 and preds.shape[1] == 1:
        preds = preds.reshape(-1)
    return preds


def log_metrics(metrics: Dict[str, float], split: str="train"):
    print()
    for metric, value in metrics.items():
        print(f"{metric}: {split} {value:.6f}")


# TODO: calc errors variance, average daily return of a mlp, remove MAPE (works poorly for log-profits), add Sharpe Ratio
def calc_metrics_mlp_uni_reg(mlp_reg: MLPRegressor, X: np.ndarray, y: np.ndarray, y_scaler: TransformerMixin=None) -> Dict[str, float]:

    metrics = {}
    preds = mlp_reg.predict(X)
    if y_scaler is not None:
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)

        y = y_scaler.inverse_transform(y)
        preds = y_scaler.inverse_transform(preds)

    if y.ndim == 2 and y.shape[1] == 1:
        y = y.reshape(-1)
    if preds.ndim == 2 and preds.shape[1] == 1:
        preds = preds.reshape(-1)

    # calculates how often pred price moves the same direction as y
    # will not work for multivariate predictions
    correct_direction_percent = np.mean(((preds >= 0) & (y >= 0)) | ((preds < 0) & (y < 0)))

    metrics["R2 Score"] = r2_score(y, preds)
    metrics["Root Mean Squared Error"] = root_mean_squared_error(y, preds)
    metrics["Mean Absolute Error"] = mean_absolute_error(y, preds)
    metrics["Mean Absolute Percentage Error"] = mean_absolute_percentage_error(y, preds)
    metrics["Median Absolute Error"] = median_absolute_error(y, preds)
    metrics["Correct Price Direction Percentage"] = correct_direction_percent

    return metrics


def calc_metrics_for_predictions(preds: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    metrics = {}

    if y.ndim == 2 and y.shape[1] == 1:
        y = y.reshape(-1)
    if preds.ndim == 2 and preds.shape[1] == 1:
        preds = preds.reshape(-1)

    correct_direction_percent = np.mean(((preds >= 0) & (y >= 0)) | ((preds < 0) & (y < 0)))

    metrics["R2 Score"] = r2_score(y, preds)
    metrics["Root Mean Squared Error"] = root_mean_squared_error(y, preds)
    metrics["Mean Absolute Error"] = mean_absolute_error(y, preds)
    metrics["Mean Absolute Percentage Error"] = mean_absolute_percentage_error(y, preds)
    metrics["Median Absolute Error"] = median_absolute_error(y, preds)
    metrics["Correct Price Direction Percentage"] = correct_direction_percent

    return metrics
