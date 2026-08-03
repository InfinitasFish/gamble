from collections import defaultdict
from sklearn.base import RegressorMixin
from sklearn.model_selection import TimeSeriesSplit, HalvingRandomSearchCV
import numpy as np

from experimental.mlp.mlp_model import calc_metrics_for_predictions
from constants import INNER_CV_FOLDS, RANDOM_STATE, OUTER_CV_FOLDS


def outer_train_gb(gb_reg: RegressorMixin, X: np.ndarray, y: np.ndarray, outer_cv: int=OUTER_CV_FOLDS,
                   inner_cv: int=INNER_CV_FOLDS, param_distr: dict=None, random_state: int=RANDOM_STATE, verbose: int=0,
                   ) -> dict[str, float]:

    kfold = TimeSeriesSplit(n_splits=outer_cv)
    test_metrics = defaultdict(list)
    for i, (train_index, test_index) in enumerate(kfold.split(X)):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

        gb_reg_i = inner_train_gb(gb_reg, X_train, y_train, param_distr, inner_cv, random_state, verbose)
        preds_i = gb_reg_i.predict(X_test)

        test_metrics_i = calc_metrics_for_predictions(preds_i, y_test)
        for k, v in test_metrics_i.items():
            test_metrics[k].append(v)

    # calculating average metrics on outer test splits
    agg_metrics = {}
    for k, v in test_metrics.items():
        agg_metrics[k] = sum(v) / len(v)
    return agg_metrics


def inner_train_gb(gb_reg: RegressorMixin, X_train: np.ndarray, y_train: np.ndarray, param_distr: dict=None, cv: int=INNER_CV_FOLDS,
                   random_state: int=RANDOM_STATE, verbose: int=0) -> RegressorMixin:

    if y_train.ndim == 2 and y_train.shape[1] == 1:
        y_train = y_train.reshape(-1)

    if param_distr is None:
        gb_reg.fit(X_train, y_train)
        return gb_reg
    else:
        ts_cv = TimeSeriesSplit(n_splits=cv)
        search = HalvingRandomSearchCV(gb_reg, param_distributions=param_distr,
                                       n_candidates="exhaust", factor=1.5, refit=True,
                                       scoring="neg_root_mean_squared_error", cv=ts_cv,
                                       random_state=random_state, n_jobs=2, verbose=verbose)
        search.fit(X_train, y_train)
        return search.best_estimator_

