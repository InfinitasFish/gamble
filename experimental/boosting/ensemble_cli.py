# without this `uv run` can't find constants module
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
from datetime import datetime
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor

from experimental.mlp.mlp_model import log_metrics, calc_metrics_for_predictions
from preproc.xy import split_seq_xy_pipe, get_candles_xy, NormType, denoise_xy_features_wma, normalize_sequence_uni, split_xy_to_sequences
from constants import YDEX_TICKER, FROM_ISO, TO_ISO, TINK_INTERVALS, REG_METRICS_USER, REG_METRICS_TO_SKLEARN


parser = argparse.ArgumentParser()
# no '--' means positional argument
parser.add_argument("--from_iso", type=str, nargs='?', default=FROM_ISO, help="Date to take candles data from (iso format)")
parser.add_argument("--to_iso", type=str, nargs='?', default=TO_ISO, help="Date to take candles data up to (iso format)")
parser.add_argument("--ticker", type=str, nargs='?', default=YDEX_TICKER, help="Instrument's ticker on which experimental will be trained")
parser.add_argument("--interval", choices=TINK_INTERVALS, default="CANDLE_INTERVAL_DAY", help="Time interval for a candle to train on")
parser.add_argument("--seq_len", type=int, nargs='?', default=2, help="Number of candles to train and predict the next value on")
parser.add_argument("--norm_type", choices=["none", "minmax", "standardize"], nargs='?', default="standardize", help="Type of data normalization for training a experimental")
parser.add_argument("--scale_y", action=argparse.BooleanOptionalAction, default=False, help="Enable target normalization for training a experimental")
parser.add_argument("--ma_window", type=int, nargs='?', default=0, help="Denoise data features with exp moving averages with window N")
parser.add_argument("--val_metric", type=str, nargs='?', default="Root Mean Squared Error", help="Metric for selecting the best experimental")
parser.add_argument("--verbose", type=int, nargs='?', default=1, help="Set verbosity for training a experimental")


def main():
    args = parser.parse_args()
    ticker = args.ticker.upper()
    from_iso = datetime.fromisoformat(args.from_iso).isoformat()
    to_iso = datetime.fromisoformat(args.to_iso).isoformat()
    interval = args.interval.upper()
    seq_len = args.seq_len
    match args.norm_type:
        case "none": norm_type = NormType.NoNorm
        case "minmax": norm_type = NormType.MinMax
        case "standardize": norm_type = NormType.Standardize
        case _: raise ValueError("how")
    scale_y = args.scale_y
    ma_window = args.ma_window
    val_metric = REG_METRICS_TO_SKLEARN[args.val_metric]
    # verbose = args.verbose

    # data prep
    local_test_size = 0.25
    X, y = get_candles_xy(from_iso, to_iso, ticker, interval)
    if ma_window > 0:
        X, _ = denoise_xy_features_wma(X, window=ma_window)

    X_train, X_test, y_train, y_test, X_scaler, y_scaler = split_seq_xy_pipe(X, y, seq_len=seq_len,
                                                                             test_size=local_test_size,
                                                                             norm_type=norm_type,
                                                                             scale_y=scale_y)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # lightgbm grows leaf-by-leaf (unbalanced tree), xgb grows level-by-level (balanced tree),
    # so ensemble is solid no-brainer
    # todo: add metric selection
    models = {
        "xgb": XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8),
        "lgbm": LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.05, min_child_samples=20),
        "et": ExtraTreesRegressor(n_estimators=300, max_features="sqrt"),
    }

    models_preds = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        models_preds[name] = model.predict(X_test)

    # as final forecast average models predictions
    preds = np.mean([models_preds["xgb"], models_preds["lgbm"], models_preds["et"]], axis=0)
    metrics = calc_metrics_for_predictions(preds, y_test)
    log_metrics(metrics, "test ensemble")

    # comparing with naive model
    naive_preds = np.array([np.mean(y_train) for _ in range(y_test.shape[0])])
    metrics = calc_metrics_for_predictions(naive_preds, y_test)
    log_metrics(metrics, "test naive")

    # full data fit, metrics
    X, y, X_scaler, y_scaler = normalize_sequence_uni(X, y, norm_type, scale_y)
    X, y = split_xy_to_sequences(X, y, seq_len)

    models_preds = {}
    for name, model in models.items():
        model.fit(X, y)
        models_preds[name] = model.predict(X)
    preds = np.mean([models_preds["xgb"], models_preds["lgbm"], models_preds["et"]], axis=0)
    metrics = calc_metrics_for_predictions(preds, y)
    log_metrics(metrics, "full ensemble")


if __name__ == "__main__":
    main()

