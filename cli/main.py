import argparse
from datetime import datetime
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from scipy import stats

from constants import YDEX_TICKER, TS_SEQUENCE_LEN
from data.candles import convert_datetime_api_format
from model.mlp import init_mlp_uni_reg, train_mlp_uni_reg, calc_metrics_mlp_uni_reg
from preproc.xy import get_candles_uni_xy_pipe

parser = argparse.ArgumentParser()
# no '--' means positional argument
parser.add_argument("from_iso", type=str, nargs='?', default="2024-01-01", help="Date to take candles data from (iso format)")
parser.add_argument("to_iso", type=str, nargs='?', default="2026-01-01", help="Date to take candles data up to (iso format)")
parser.add_argument("--seq_len", type=int, nargs='?', default=TS_SEQUENCE_LEN, help="Number of candles for training and predicting next value")


def main():
    args = parser.parse_args()
    from_iso = convert_datetime_api_format(datetime.fromisoformat(args.from_iso))
    to_iso = convert_datetime_api_format(datetime.fromisoformat(args.to_iso))
    seq_len = args.seq_len

    # todo: add like another outer loop to search best TS_SEQUENCE_LEN
    search_params_distr = {"loss": ["squared_error"],
                           "learning_rate": ["constant", "adaptive"],
                           "hidden_layer_sizes": [(50,), (100,), (150,), (200,), (50, 50), (100, 100), (150, 150), (200, 200)],
                           # [loc, loc + scale]
                           "learning_rate_init": stats.uniform(0.0001, 0.1),
                           # [loc, scale]
                           "max_iter": stats.randint(1000, 3000)}

    # normalization doesn't work btw
    X_train, X_test, y_train, y_test = get_candles_uni_xy_pipe(from_iso, to_iso, YDEX_TICKER, to_cache=True,
                                                               sequence_len=seq_len, test_size=0.05, normalize=False)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    mlp_reg = train_mlp_uni_reg(init_mlp_uni_reg(), X_train, y_train, param_distr=search_params_distr, verbose=1)
    calc_metrics_mlp_uni_reg(mlp_reg, X_test, y_test, X_train, y_train)


if __name__ == "__main__":
    main()
