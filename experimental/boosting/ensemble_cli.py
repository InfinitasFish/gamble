# without this `uv run` can't find constants module
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from collections import defaultdict
import argparse
from datetime import datetime
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor

from experimental.mlp.mlp_model import log_metrics, calc_metrics_for_predictions
from experimental.boosting.params_distributions import xgb_param_distr, lgbm_param_distr, et_param_distr
from experimental.boosting.ensemble_model import outer_train_gb, inner_train_gb
from preproc.xy import get_candles_xy, NormType, normalize_sequence_uni, split_xy_to_sequences
from constants import YDEX_TICKER, FROM_ISO, TO_ISO, TINK_INTERVALS, REG_METRICS_USER, REG_METRICS_TO_SKLEARN


# todo: refactor this file pls
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

    X, y = get_candles_xy(from_iso, to_iso, ticker, interval)

    # lightgbm grows leaf-by-leaf (unbalanced tree), xgb grows level-by-level (balanced tree),
    # so ensemble is solid no-brainer
    # todo: add metric selection
    models = {
        "xgb": (XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8), xgb_param_distr),
        "lgbm": (LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.05, min_child_samples=20), lgbm_param_distr),
        "et": (ExtraTreesRegressor(n_estimators=300, max_features="sqrt"), et_param_distr),
    }

    # nested cv ensemble validation
    models_metrics = {}
    for model, params in models.items():
        # freeze on et
        if model == "et": continue
        model_metrics = outer_train_gb(params[0], X, y, param_distr=params[1])
        models_metrics[model] = model_metrics

    agg_metrics = defaultdict(float)
    for k, v in models_metrics.items():
        for metric, val in v.items():
            agg_metrics[metric] += val
    for k, v in agg_metrics.items():
        agg_metrics[k] /= len(models_metrics.keys())
    log_metrics(agg_metrics, "cv ensemble")

    # refit on full data
    models_preds = {}
    for model, params in models.items():
        if model == "et":
            fit_model = params[0].fit(X, y)
        else:
            fit_model = inner_train_gb(params[0], X, y, params[1])
        models_preds[model] = fit_model.predict(X)
    preds = np.mean([models_preds["xgb"], models_preds["lgbm"], models_preds["et"]], axis=0)
    metrics = calc_metrics_for_predictions(preds, y)
    log_metrics(metrics, "refit ensemble")

    # comparing with naive model
    naive_preds = np.array([np.mean(y) for _ in range(y.shape[0])])
    metrics = calc_metrics_for_predictions(naive_preds, y)
    log_metrics(metrics, "refit naive")


if __name__ == "__main__":
    main()

