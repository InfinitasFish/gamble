import argparse
from datetime import datetime
from scipy import stats
import numpy as np
from constants import YDEX_TICKER, TS_MAX_SEQUENCE_LEN, RANDOM_STATE
from data.candles import convert_datetime_api_format
from model.mlp import (init_mlp_uni_reg, predict_next_prices, train_mlp_uni_reg, calc_metrics_mlp_uni_reg, log_metrics)
from preproc.xy import get_candles_xy, split_xy_to_sequences, split_seq_xy_pipe, normalize_sequence_uni, NormType, \
    denoise_xy_features_wma

parser = argparse.ArgumentParser()
# no '--' means positional argument
parser.add_argument("--ticker", type=str, nargs='?', default=YDEX_TICKER, help="Instrument's ticker on which model will be trained")
parser.add_argument("--from_iso", type=str, nargs='?', default="2024-01-01", help="Date to take candles data from (iso format)")
parser.add_argument("--to_iso", type=str, nargs='?', default="2026-01-01", help="Date to take candles data up to (iso format)")
parser.add_argument("--seq_len", type=int, nargs='?', default=1, help="Number of candles to train and predict the next value on")
parser.add_argument("--norm_type", choices=["none", "minmax", "standardize"], nargs='?', default="none", help="Type of data normalization for training a model")
parser.add_argument("--scale_y", action=argparse.BooleanOptionalAction, help="Enable target normalization for training a model")
parser.add_argument("--ma_window", type=int, nargs='?', default=0, help="Denoise data features with exp moving averages with window N")
parser.add_argument("--val_metric", type=str, nargs='?', default="Root Mean Squared Error", help="Metric for selecting the best model")
parser.add_argument("--verbose", type=int, nargs='?', default=1, help="Set verbosity for training a model")


def main():
    args = parser.parse_args()
    ticker = args.ticker.upper()
    from_iso = convert_datetime_api_format(datetime.fromisoformat(args.from_iso))
    to_iso = convert_datetime_api_format(datetime.fromisoformat(args.to_iso))
    seq_len = args.seq_len
    match args.norm_type:
        case "none": norm_type = NormType.NoNorm
        case "minmax": norm_type = NormType.MinMax
        case "standardize": norm_type = NormType.Standardize
        case _: raise ValueError("Bro")
    scale_y = args.scale_y
    ma_window = args.ma_window
    verbose = args.verbose
    temporal = True

    search_params_distr = {"loss": ["squared_error", "poisson"],
                           "learning_rate": ["constant", "adaptive"],
                           "hidden_layer_sizes": [(50,), (100,), (150,), (200,), (50, 50), (100, 100), (150, 150), (200, 200)],
                           "solver": ["lbfgs", "adam"],
                           "activation": ["relu", "logistic"],
                           # [loc, loc + scale]
                           "learning_rate_init": stats.uniform(0.0001, 0.1),
                           # [loc, scale]
                           "max_iter": stats.randint(1000, 4000)}

    local_test_size = 0.1
    X, y = get_candles_xy(from_iso, to_iso, ticker, to_cache=True)
    if ma_window > 0:
        X, _ = denoise_xy_features_wma(X, window=ma_window)

    X_train, X_test, y_train, y_test, X_scaler, y_scaler = split_seq_xy_pipe(X, y, seq_len, temporal=temporal,
                                                                             test_size=local_test_size,
                                                                             norm_type=norm_type, scale_y=scale_y)

    # halving random cv search
    mlp_reg = train_mlp_uni_reg(init_mlp_uni_reg(), X_train, y_train, search_params_distr, verbose=verbose)

    # final test metrics on splits
    metrics = calc_metrics_mlp_uni_reg(mlp_reg, X_test, y_test, y_scaler)
    log_metrics(metrics, "test")

    # TODO: metrics are better now, but R2 score is negative, so data is too noisy, also model tends to use smaller sequence length
    #    Possible fixes - feature selections, features denoising
    # full data fit, metrics, predict
    X, y, X_scaler, y_scaler = normalize_sequence_uni(X, y, norm_type, scale_y)
    X, y = split_xy_to_sequences(X, y, seq_len)
    mlp_reg.fit(X, y)
    metrics = calc_metrics_mlp_uni_reg(mlp_reg, X, y, y_scaler)
    log_metrics(metrics, "full")

    next_day_pred = predict_next_prices(mlp_reg, X, seq_len, y_scaler)[-1]
    print(f"\nPredictions for the day after {args.to_iso} is {next_day_pred}")
    print(f"Best sequence len is {seq_len}")

if __name__ == "__main__":
    main()
