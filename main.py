from data.candles import get_candles_data, get_daily_candles_df, convert_datetime_api_format, inspect_candles_dict, load_candles_data


def main():
    ydex_daily_candle_data = load_candles_data(candles_data_fpath="./cache/YDEX_TQBR_day_2026_04_04T154934_2026_04_14T154934.json")
    ydex_30min_candle_data = load_candles_data(candles_data_fpath="./cache/YDEX_TQBR_30min_2026_04_08T180338_2026_04_18T180338.json")

    # inspect_candles_dict(ydex_daily_candle_data)
    candles_df = get_daily_candles_df(ydex_daily_candle_data)
    print(candles_df.head())

    candles_df = get_daily_candles_df(ydex_30min_candle_data)
    print(candles_df.head())


if __name__ == "__main__":
    main()
