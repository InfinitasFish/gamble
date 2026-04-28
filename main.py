from datetime import datetime, timezone, timedelta

from constants import YDEX_TICKER
from data.candles import get_candles_uni_xy_pipe, convert_datetime_api_format


def main():
    now = convert_datetime_api_format(datetime.now(timezone.utc))
    ten_days_ago = convert_datetime_api_format(datetime.now(timezone.utc) - timedelta(days=10))

    X, y = get_candles_uni_xy_pipe(ten_days_ago, now, YDEX_TICKER)
    print(X.shape, y.shape)
    print(X[:5])
    print(y[:5])


if __name__ == "__main__":
    main()
