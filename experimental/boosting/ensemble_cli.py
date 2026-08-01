import argparse
from datetime import datetime
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor

from experimental.mlp.mlp_model import log_metrics, calc_metrics_for_predictions
from preproc.xy import split_seq_xy_pipe, get_candles_xy, NormType
from constants import YDEX_TICKER, FROM_ISO, TO_ISO


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
    seq_len = args.seq_len

if __name__ == "__main__":
    from_iso = FROM_ISO
    to_iso = TO_ISO
    ticker = YDEX_TICKER
    interval = "CANDLE_INTERVAL_DAY"

    # data prep
    local_test_size = 0.25
    X, y = get_candles_xy(from_iso, to_iso, ticker, interval)
    X_train, X_test, y_train, y_test, X_scaler, y_scaler = split_seq_xy_pipe(X, y, seq_len=4, temporal=True, test_size=local_test_size,
                                                                             norm_type=NormType.Standardize, scale_y=False)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # lightgbm grows leaf-by-leaf (unbalanced tree), xgb grows level-by-level (balanced tree),
    # so ensemble is solid no-brainer
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
