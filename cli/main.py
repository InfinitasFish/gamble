import argparse
from datetime import datetime
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, median_absolute_error, mean_absolute_percentage_error

from constants import YDEX_TICKER, TS_SEQUENCE_LEN
from data.candles import convert_datetime_api_format
from model.mlp import init_mlp_uni_reg, train_mlp_uni_reg
from preproc.xy import get_candles_uni_xy_pipe

parser = argparse.ArgumentParser()
# no '--' means positional argument
parser.add_argument("from_iso", type=str, nargs='?', default="2025-01-21", help="Date to take candles data from (iso format)")
parser.add_argument("to_iso", type=str, nargs='?', default="2026-01-01", help="Date to take candles data up to (iso format)")
parser.add_argument("--seq_len", type=int, nargs='?', default=TS_SEQUENCE_LEN, help="Number of candles for training and predicting next value")


def main():
    args = parser.parse_args()
    from_iso = convert_datetime_api_format(datetime.fromisoformat(args.from_iso))
    to_iso = convert_datetime_api_format(datetime.fromisoformat(args.to_iso))
    seq_len = args.seq_len
    X, y = get_candles_uni_xy_pipe(from_iso, to_iso, YDEX_TICKER, to_cache=True, sequence_len=seq_len)
    print(X.shape, y.shape)
    mlp_reg = train_mlp_uni_reg(init_mlp_uni_reg(), X, y)
    preds = mlp_reg.predict(X)
    # useless R2-score and useful metrics
    print(f"R2 score: {mlp_reg.score(X, y):.6f}")
    print(f"Root mean squared error: {root_mean_squared_error(y, preds):.6f}")
    print(f"Mean Absolute Error: {mean_absolute_error(y, preds):.6f}")
    print(f"Mean Absolute Percentage Error: {mean_absolute_percentage_error(y, preds):.6f}")
    print(f"Median Absolute Error: {median_absolute_error(y, preds):.6f}")


if __name__ == "__main__":
    main()
