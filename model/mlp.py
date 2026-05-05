import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, median_absolute_error, mean_absolute_percentage_error

from constants import RANDOM_STATE, MAX_ITER, CV_FOLDS


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
        return search


def calc_metrics_mlp_uni_reg(mlp_reg: MLPRegressor, X_test: np.ndarray, y_test: np.ndarray,
                             X_train: np.ndarray=None, y_train: np.ndarray=None):

    preds_test = mlp_reg.predict(X_test)
    if X_train is not None and y_train is not None:
        preds_train = mlp_reg.predict(X_train)
        print(f"R2 Score:\n  - test {mlp_reg.score(X_test, y_test):.6f}\n  - train {mlp_reg.score(X_train, y_train):.6f}")
        print(f"Root Mean Squared Error:\n  - test {root_mean_squared_error(y_test, preds_test):.6f}\n  - train {root_mean_squared_error(y_train, preds_train):.6f}")
        print(f"Mean Absolute Error:\n  - test {mean_absolute_error(y_test, preds_test):.6f}\n  - train {mean_absolute_error(y_train, preds_train):.6f}")
        print(f"Mean Absolute Percentage Error:\n  - test {mean_absolute_percentage_error(y_test, preds_test):.6f}\n  - train {mean_absolute_percentage_error(y_train, preds_train):.6f}")
        print(f"Median Absolute Error:\n  - test {median_absolute_error(y_test, preds_test):.6f}\n  - train {median_absolute_error(y_train, preds_train):.6f}")

    else:
        print(f"Test R2 Score: {mlp_reg.score(X_test, y_test):.6f}")
        print(f"Test Root Mean Squared Error: {root_mean_squared_error(y_test, preds_test):.6f}")
        print(f"Test Mean Absolute Error: {mean_absolute_error(y_test, preds_test):.6f}")
        print(f"Test Mean Absolute Percentage Error: {mean_absolute_percentage_error(y_test, preds_test):.6f}")
        print(f"Test Median Absolute Error: {median_absolute_error(y_test, preds_test):.6f}")
