import argparse
from datetime import datetime, timezone, timedelta
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from constants import YDEX_TICKER
from data.candles import get_candles_uni_xy_pipe, convert_datetime_api_format

parser = argparse.ArgumentParser()
# no '--' means positional argument
parser.add_argument("from", type=str, nargs='?', default="2025-01-21", help="Date to take candles data from (iso format)")
parser.add_argument("to", type=str, nargs='?', default="2026-04-26", help="Date to take candles data up to (iso format)")


def data_test():
    now = convert_datetime_api_format(datetime.now(timezone.utc))
    ten_days_ago = convert_datetime_api_format(datetime.now(timezone.utc) - timedelta(days=10))

    X, y = get_candles_uni_xy_pipe(ten_days_ago, now, YDEX_TICKER)
    print(X.shape, y.shape)
    print(X[:5])
    print(y[:5])


def main():
    args = parser.parse_args()
    print(args)



if __name__ == "__main__":
    main()
