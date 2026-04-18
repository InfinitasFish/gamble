from datetime import datetime, timezone, timedelta

from constants import YDEX_TICKER
from data.bonds import get_all_bonds_data
from data.candles import get_candles_data, get_daily_candles_df, convert_datetime_api_format, inspect_candles_dict, load_candles_data


def main():
    # print(get_all_bonds_data()["instruments"][0]["ticker"])

    # now = convert_datetime_api_format(datetime.now(timezone.utc))
    # ten_days_ago = convert_datetime_api_format(datetime.now(timezone.utc) - timedelta(days=10))

    # ydex_daily_candle_data = get_candles_data(ten_days_ago, now, YDEX_TICKER, interval="CANDLE_INTERVAL_DAY")
    # ydex_half_hour_candle_data = get_candles_data(ten_days_ago, now, YDEX_TICKER, interval="CANDLE_INTERVAL_30_MIN")
    ydex_daily_candle_data = load_candles_data(candles_data_fpath="./cache/YDEX_TQBR_day_2026_04_04T154934_2026_04_14T154934.json")

    inspect_candles_dict(ydex_daily_candle_data)


if __name__ == "__main__":
    main()
