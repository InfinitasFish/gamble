from typing import Dict, Tuple
import warnings
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.experimental import enable_halving_search_cv
# after scaling targets model converges without warnings
# from sklearn.exceptions import ConvergenceWarning
# warnings.filterwarnings("ignore", category=ConvergenceWarning)
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import HalvingRandomSearchCV, TimeSeriesSplit
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, median_absolute_error, mean_absolute_percentage_error, r2_score

from preproc.xy import split_xy_to_sequences, normalize_sequence_uni, NormType
from constants import RANDOM_STATE, MAX_ITER, CV_FOLDS

# TODO: there are specific mlp for timeseries, consider implementing, e.g. https://arxiv.org/pdf/2311.06184


def init_mlp_uni_reg(max_iter: int=MAX_ITER, verbose: bool=False, random_state: int=RANDOM_STATE) -> MLPRegressor:
    mlp_reg = MLPRegressor(loss="squared_error", hidden_layer_sizes=(150,), activation='relu', learning_rate_init=0.001,
                           shuffle=False, verbose=verbose, max_iter=max_iter, random_state=random_state)
    return mlp_reg


def train_mlp_uni_reg(mlp_reg: MLPRegressor, X_train: np.ndarray, y_train: np.ndarray, param_distr: dict=None,
                      cv: int=CV_FOLDS, verbose: int=0, random_state: int=RANDOM_STATE) -> MLPRegressor:

    if y_train.ndim == 2 and y_train.shape[1] == 1:
        y_train = y_train.reshape(-1)

    if param_distr is None:
        mlp_reg.fit(X_train, y_train)
        return mlp_reg
    else:
        # HalvingRandomSearchCV iteratively increases the resource (data n_samples by default) to fit with CV
        #   while also decreases the amount of candidates at each step
        # with 'refit' returns instance of model fitted with best params
        ts_cv = TimeSeriesSplit(n_splits=cv)
        search = HalvingRandomSearchCV(mlp_reg, param_distributions=param_distr, n_candidates="exhaust", factor=1.5, refit=True,
                                       scoring="neg_root_mean_squared_error", cv=ts_cv, random_state=random_state, n_jobs=2, verbose=verbose,
                                       ).fit(X_train, y_train)
        return search.best_estimator_


def fit_mlp_uni_reg(mlp_reg: MLPRegressor, X: np.ndarray, y: np.ndarray, seq_len: int, norm_type: NormType=NormType.NoNorm,
                    scale_y: bool=True) -> Tuple[MLPRegressor, TransformerMixin, TransformerMixin]:
    """Fit the best trained model on all data for future predictions"""

    X, y, X_scaler, y_scaler = normalize_sequence_uni(X, y, norm_type, scale_y)
    X, y = split_xy_to_sequences(X, y, seq_len)

    mlp_reg.fit(X, y)

    return mlp_reg, X_scaler, y_scaler


def predict_next_prices(mlp_reg: MLPRegressor, X: np.ndarray, seq_len: int, y_scaler: TransformerMixin=None) -> np.ndarray:
    """Predict next price with fit model and its sequence len"""
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


# TODO: calc errors variance, average daily return of a mlp
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

