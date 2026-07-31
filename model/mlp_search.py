from typing import Dict, Tuple
from datetime import datetime
import numpy as np
from scipy import stats
from sklearn.neural_network import MLPRegressor

from data.candles_tink import convert_datetime2api_format
from preproc.xy import NormType, split_seq_xy_pipe, get_candles_xy, denoise_xy_features_wma, split_xy_to_sequences, normalize_sequence_uni
from model.mlp import init_mlp_uni_reg, train_mlp_uni_reg, calc_metrics_mlp_uni_reg, log_metrics, predict_next_prices
from constants import FROM_ISO, TO_ISO, YDEX_TICKER, RANDOM_STATE, CV_FOLDS, TEST_SIZE, TS_MIN_SEQUENCE_LEN, TS_MAX_SEQUENCE_LEN


def norm_type_search(init_mlp_reg: MLPRegressor, val_metric: str, more_better: bool, X: np.ndarray, y: np.ndarray, temporal: bool=True,
                     test_size: float=TEST_SIZE, seq_len: int=TS_MIN_SEQUENCE_LEN, param_distr: dict=None,
                     cv: int=CV_FOLDS, verbose: int=0, random_state: int=RANDOM_STATE,
                     ) -> Tuple[MLPRegressor, NormType]:
    """Training multiple MLP to find the best normalization type for prediction"""

    best_val_error = -float("inf") if more_better else float("inf")
    best_mlp_reg = None
    best_norm_type = None
    for norm_type in [NormType.NoNorm, NormType.Standardize, NormType.MinMax]:
        X_train, X_test, y_train, y_test, X_scaler, y_scaler = split_seq_xy_pipe(X, y, seq_len, temporal, test_size,
                                                                                 norm_type, scale_y=False)
        trained_mlp_reg = train_mlp_uni_reg(init_mlp_reg, X_train, y_train, param_distr, cv, verbose, random_state)
        target_error_metric = calc_metrics_mlp_uni_reg(trained_mlp_reg, X_test, y_test, y_scaler)[val_metric]

        if more_better:
            if target_error_metric > best_val_error:
                best_val_error = target_error_metric
                best_mlp_reg = trained_mlp_reg
                best_norm_type = norm_type
                print(f"New best '{val_metric}': {best_val_error:.6f} with norm type {best_norm_type}")
        else:
            if target_error_metric < best_val_error:
                best_val_error = target_error_metric
                best_mlp_reg = trained_mlp_reg
                best_norm_type = norm_type
                print(f"New best '{val_metric}': {best_val_error:.6f} with norm type {best_norm_type}")

    return best_mlp_reg, best_norm_type


def seq_len_search(init_mlp_reg: MLPRegressor, val_metric: str, more_better: bool, X: np.ndarray, y: np.ndarray, norm_type: NormType=NormType.NoNorm,
                   temporal: bool=True, test_size: float=TEST_SIZE, min_len: int=TS_MIN_SEQUENCE_LEN, max_len: int=TS_MAX_SEQUENCE_LEN,
                   param_distr: dict=None, cv: int=CV_FOLDS, verbose: int=0, random_state: int=RANDOM_STATE
                   ) -> Tuple[MLPRegressor, int]:
    """Training multiple MLP to find the best sequence length for prediction"""

    best_val_error = -float("inf") if more_better else float("inf")
    best_mlp_reg = None
    best_seq_len = -1
    for seq_len in range(min_len, max_len + 1):
        X_train, X_test, y_train, y_test, X_scaler, y_scaler = split_seq_xy_pipe(X, y, seq_len, temporal, test_size,
                                                                                 norm_type, scale_y=False)
        trained_mlp_reg = train_mlp_uni_reg(init_mlp_reg, X_train, y_train, param_distr, cv, verbose, random_state)
        target_error_metric = calc_metrics_mlp_uni_reg(trained_mlp_reg, X_test, y_test, y_scaler)[val_metric]

        if more_better:
            if target_error_metric > best_val_error:
                best_val_error = target_error_metric
                best_mlp_reg = trained_mlp_reg
                best_seq_len = seq_len
                print(f"New best '{val_metric}': {best_val_error:.6f} with sequence_len {best_seq_len}")
        else:
            if target_error_metric < best_val_error:
                best_val_error = target_error_metric
                best_mlp_reg = trained_mlp_reg
                best_seq_len = seq_len
                print(f"New best '{val_metric}': {best_val_error:.6f} with sequence_len {best_seq_len}")

    return best_mlp_reg, best_seq_len


def ma_window_search(init_mlp_reg: MLPRegressor, val_metric: str, more_better: bool, X: np.ndarray, y: np.ndarray,
                     norm_type: NormType, temporal: bool=True, test_size: float=TEST_SIZE, seq_len: int=TS_MIN_SEQUENCE_LEN,
                     param_distr: dict=None, max_ma_window: int=TS_MAX_SEQUENCE_LEN // 2, cv: int=CV_FOLDS, verbose: int=0,
                     random_state: int=RANDOM_STATE,
                     ) -> Tuple[MLPRegressor, int]:
    """Training multiple MLP to find best moving average window for prediction"""

    best_val_error = -float("inf") if more_better else float("inf")
    best_mlp_reg = None
    best_ma_window = None
    for ma_window in range(0, max_ma_window + 1):

        X_ma = np.copy(X)
        if ma_window > 0:
            X_ma, _ = denoise_xy_features_wma(X_ma, window=ma_window)
        X_train, X_test, y_train, y_test, X_scaler, y_scaler = split_seq_xy_pipe(X_ma, y, seq_len, temporal, test_size,
                                                                                 norm_type, scale_y=False)

        trained_mlp_reg = train_mlp_uni_reg(init_mlp_reg, X_train, y_train, param_distr, cv, verbose, random_state)
        target_error_metric = calc_metrics_mlp_uni_reg(trained_mlp_reg, X_test, y_test, y_scaler)[val_metric]

        if more_better:
            if target_error_metric > best_val_error:
                best_val_error = target_error_metric
                best_mlp_reg = trained_mlp_reg
                best_ma_window = ma_window
                print(f"New best '{val_metric}': {best_val_error:.6f} with ma window {best_ma_window}")
        else:
            if target_error_metric < best_val_error:
                best_val_error = target_error_metric
                best_mlp_reg = trained_mlp_reg
                best_ma_window = ma_window
                print(f"New best '{val_metric}': {best_val_error:.6f} with ma window {best_ma_window}")

    return best_mlp_reg, best_ma_window


# useless test
def is_temporal_better(init_mlp_reg: MLPRegressor, val_metric: str, more_better: bool, X: np.ndarray, y: np.ndarray,
                       norm_type: NormType, test_size: float=TEST_SIZE, seq_len: int=TS_MIN_SEQUENCE_LEN,
                       param_distr: dict=None, cv: int=CV_FOLDS, verbose: int=0, random_state: int=RANDOM_STATE,
                       ) -> Tuple[MLPRegressor, bool]:

    best_val_error = -float("inf") if more_better else float("inf")
    best_mlp_reg = None
    is_temporal = None
    for temporal_flag in [True, False]:

        X_train, X_test, y_train, y_test, X_scaler, y_scaler = split_seq_xy_pipe(X, y, seq_len, temporal_flag, test_size,
                                                                                 norm_type, scale_y=False)
        trained_mlp_reg = train_mlp_uni_reg(init_mlp_reg, X_train, y_train, param_distr, cv, verbose, random_state)
        target_error_metric = calc_metrics_mlp_uni_reg(trained_mlp_reg, X_test, y_test, y_scaler)[val_metric]

        if more_better:
            if target_error_metric > best_val_error:
                best_val_error = target_error_metric
                best_mlp_reg = trained_mlp_reg
                is_temporal = temporal_flag
                print(f"New best '{val_metric}': {best_val_error:.6f} with is_temporal={is_temporal}")
        else:
            if target_error_metric < best_val_error:
                best_val_error = target_error_metric
                best_mlp_reg = trained_mlp_reg
                is_temporal = temporal_flag
                print(f"New best '{val_metric}': {best_val_error:.6f} with is_temporal={is_temporal}")

    return best_mlp_reg, is_temporal


def main():
    # search all non-model hyperparameters, while using the best parameter in consecutive search.
    # search is naive because we assume that best parameter on first step will contribute to the best quality
    # on the next step

    # smaller search than main fit
    search_params_distr = {"loss": ["squared_error"],
                           "learning_rate": ["adaptive"],
                           "hidden_layer_sizes": [(50,), (100,), (150,), (200,), (50, 50), (100, 100), (150, 150),
                                                  (200, 200)],
                           # [loc, loc + scale]
                           "learning_rate_init": stats.uniform(0.0001, 0.1),
                           # [loc, scale]
                           "max_iter": stats.randint(1000, 4000)}

    local_test_size = 0.25
    ticker = YDEX_TICKER
    from_iso = convert_datetime2api_format(datetime.fromisoformat(FROM_ISO))
    to_iso = convert_datetime2api_format(datetime.fromisoformat(TO_ISO))
    X, y = get_candles_xy(from_iso, to_iso, ticker, to_cache=True)
    val_metric = "Correct Price Direction Percentage"
    more_better = True
    print(X.shape, y.shape)

    _, bseq_len = seq_len_search(init_mlp_uni_reg(), val_metric=val_metric, more_better=more_better, X=X, y=y,
                                 norm_type=NormType.Standardize, test_size=local_test_size,
                                 min_len=TS_MIN_SEQUENCE_LEN, max_len=TS_MAX_SEQUENCE_LEN,
                                 param_distr=search_params_distr)


    _, bnorm_type = norm_type_search(init_mlp_uni_reg(), val_metric=val_metric, more_better=more_better, X=X, y=y,
                                     test_size=local_test_size, seq_len=bseq_len, param_distr=search_params_distr)


    _, bma_window = ma_window_search(init_mlp_uni_reg(), val_metric=val_metric, more_better=more_better, X=X, y=y,
                                     norm_type=bnorm_type, test_size=local_test_size, seq_len=bseq_len,
                                     param_distr=search_params_distr)


    _, is_temporal = is_temporal_better(init_mlp_uni_reg(), val_metric=val_metric, more_better=more_better, X=X, y=y,
                                        norm_type=bnorm_type, test_size=local_test_size, seq_len=bseq_len,
                                        param_distr=search_params_distr)


    print(f"Best seq_len for {ticker} is {bseq_len}")
    print(f"Best norm_type for {ticker} is {bnorm_type}")
    print(f"Best ma window for {ticker} is {bma_window}")
    print(f"Best split with is_temporal={is_temporal}")


if __name__ == "__main__":
    main()
