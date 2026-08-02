# without this `uv run` can't find constants module
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
from datetime import datetime
from scipy import stats
import numpy as np

from constants import YDEX_TICKER, FROM_ISO, TO_ISO, TINK_INTERVALS, REG_METRICS_USER, REG_METRICS_TO_SKLEARN
from experimental.mlp.mlp_model import (init_mlp_uni_reg, predict_next_prices, inner_train_mlp_uni_reg, calc_metrics_mlp_uni_reg,
                                        calc_metrics_for_predictions, log_metrics)
from preproc.xy import get_candles_xy, split_xy_to_sequences, split_seq_xy_pipe, normalize_sequence_uni, NormType, denoise_xy_features_wma


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
parser.add_argument("--val_metric", choices=REG_METRICS_USER, default="Root Mean Squared Error", help="Metric for selecting the best experimental")
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
    verbose = args.verbose

    search_params_distr = {"loss": ["squared_error"],
                           "learning_rate": ["constant", "adaptive"],
                           "hidden_layer_sizes": [(50,), (100,), (150,), (200,), (50, 50), (100, 100), (150, 150),
                                                  (200, 200), (300,), (300, 300), (50, 50, 50), (100, 100, 100)],
                           # [loc, loc + scale]
                           "learning_rate_init": stats.uniform(0.0001, 0.1),
                           # [loc, scale]
                           "max_iter": stats.randint(2000, 4000)}

    local_test_size = 0.25
    X, y = get_candles_xy(from_iso, to_iso, ticker, interval=interval, to_cache=True)
    if ma_window > 0:
        X, _ = denoise_xy_features_wma(X, window=ma_window)

    X_train, X_test, y_train, y_test, X_scaler, y_scaler = split_seq_xy_pipe(X, y, seq_len, test_size=local_test_size,
                                                                             norm_type=norm_type, scale_y=scale_y)

    # halving random cv search
    mlp_reg = inner_train_mlp_uni_reg(init_mlp_uni_reg(), X_train, y_train, search_params_distr, scoring=val_metric, verbose=verbose)

    # final test metrics on splits
    metrics = calc_metrics_mlp_uni_reg(mlp_reg, X_test, y_test, y_scaler)
    log_metrics(metrics, "test")

    # comparing with naive model
    naive_preds = np.array([np.mean(y_train) for _ in range(y_test.shape[0])])
    metrics = calc_metrics_for_predictions(naive_preds, y_test)
    log_metrics(metrics, "test naive")

    # full data fit, metrics, predict
    X, y, X_scaler, y_scaler = normalize_sequence_uni(X, y, norm_type, scale_y)
    X, y = split_xy_to_sequences(X, y, seq_len)
    mlp_reg.fit(X, y)
    metrics = calc_metrics_mlp_uni_reg(mlp_reg, X, y, y_scaler)
    log_metrics(metrics, "full")

    next_interval_pred = predict_next_prices(mlp_reg, X, seq_len, y_scaler)[-1]
    print(f"\nPredictions for the {interval} interval after {args.to_iso} is {next_interval_pred}")


if __name__ == "__main__":
    main()
