import argparse
from datetime import datetime
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from scipy import stats

from constants import YDEX_TICKER, TS_MAX_SEQUENCE_LEN, RANDOM_STATE
from data.candles import convert_datetime_api_format
from model.mlp import (init_mlp_uni_reg, outer_seq_len_search, fit_mlp_uni_reg, predict_uni_next_price,
                       calc_metrics_mlp_uni_reg, log_metrics)
from preproc.xy import get_candles_seq_uni, split_sequence, split_seq_xy_pipe, normalize_sequence_uni, NormType


parser = argparse.ArgumentParser()
# no '--' means positional argument
parser.add_argument("--ticker", type=str, nargs='?', default=YDEX_TICKER, help="Instrument's ticker on which model will be trained")
parser.add_argument("--from_iso", type=str, nargs='?', default="2024-01-01", help="Date to take candles data from (iso format)")
parser.add_argument("--to_iso", type=str, nargs='?', default="2026-01-01", help="Date to take candles data up to (iso format)")
parser.add_argument("--max_seq_len", type=int, nargs='?', default=TS_MAX_SEQUENCE_LEN, help="Upper bound for number of candles to train and predict the next value on")
parser.add_argument("--norm_type", choices=["none", "minmax", "standardize"], nargs='?', default="none", help="Type of data normalization for training a model")
parser.add_argument("--scale_y", action=argparse.BooleanOptionalAction, help="Enable target normalization for training a model")


def main():
    args = parser.parse_args()
    ticker = args.ticker.upper()
    from_iso = convert_datetime_api_format(datetime.fromisoformat(args.from_iso))
    to_iso = convert_datetime_api_format(datetime.fromisoformat(args.to_iso))
    max_seq_len = args.max_seq_len
    match args.norm_type:
        case "none": norm_type = NormType.NoNorm
        case "minmax": norm_type = NormType.MinMax
        case "standardize": norm_type = NormType.Standardize
        case _: raise ValueError("Bro")
    scale_y = args.scale_y

    search_params_distr = {"loss": ["squared_error"],
                           "learning_rate": ["constant", "adaptive"],
                           "hidden_layer_sizes": [(50,), (100,), (150,), (200,), (50, 50), (100, 100), (150, 150), (200, 200)],
                           # [loc, loc + scale]
                           "learning_rate_init": stats.uniform(0.0001, 0.1),
                           # [loc, scale]
                           "max_iter": stats.randint(1000, 4000)}

    # standard normalization doesn't work btw
    local_test_size = 0.05
    ts_sequence = get_candles_seq_uni(from_iso, to_iso, ticker, to_cache=True)
    print(ts_sequence.shape)
    mlp_reg, seq_len = outer_seq_len_search(init_mlp_uni_reg(), "Root Mean Squared Error", ts_sequence, norm_type=norm_type,
                                            test_size=local_test_size, min_len=7, max_len=max_seq_len, param_distr=search_params_distr,
                                            scale_y=scale_y, verbose=1, random_state=RANDOM_STATE)

    # final test metrics on splits
    X_train, X_test, y_train, y_test, X_scaler, y_scaler = split_seq_xy_pipe(ts_sequence, seq_len, norm_type=norm_type,
                                                                             scale_y=scale_y, test_size=local_test_size,
                                                                             random_state=RANDOM_STATE)
    metrics = calc_metrics_mlp_uni_reg(mlp_reg, X_test, y_test, y_scaler)
    log_metrics(metrics, "test")

    # full data fit, metrics, predict
    mlp_reg, X_scaler, y_scaler = fit_mlp_uni_reg(mlp_reg, ts_sequence, seq_len, norm_type, scale_y)
    X, y = split_sequence(ts_sequence, seq_len)
    X, y, _, _ = normalize_sequence_uni(X, y, norm_type, scale_y)
    metrics = calc_metrics_mlp_uni_reg(mlp_reg, X, y, y_scaler)
    log_metrics(metrics, "full")

    next_day_pred = predict_uni_next_price(mlp_reg, ts_sequence, seq_len, X_scaler, y_scaler)
    print(f"\nPrediction for the day after {args.to_iso} is {next_day_pred:.4f}")
    print(f"Best sequence len is {seq_len}")


if __name__ == "__main__":
    main()
