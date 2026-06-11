from datetime import datetime
import pandas as pd
import numpy as np
import vectorbt as vbt

from data.candles import get_candles_data, get_candles_df, convert_datetime_api_format
from data.eodhd import download_symbol_candles
from constants import CACHE_DIR_FPATH, FROM_ISO, TO_ISO, EODHD_API_TOKEN, YDEX_TICKER


def sma_20_50_eod(symbol: str, from_iso: str=FROM_ISO, to_iso: str=TO_ISO, period: str='d',
                  api_token: str=EODHD_API_TOKEN) -> pd.DataFrame:

    prices_df = download_symbol_candles(symbol, from_iso, to_iso, period, api_token)
    prices = prices_df["close"]

    fast_ma = vbt.MA.run(prices, 20)
    slow_ma = vbt.MA.run(prices, 50)
    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)

    pf = vbt.Portfolio.from_signals(prices, entries, exits, fees=0.0005, slippage=0.0005, freq='D')
    stats = pf.stats()
    return stats


def sma_n1_n2_tink(ticker: str, from_iso: str=FROM_ISO, to_iso: str=TO_ISO, interval: str="CANDLE_INTERVAL_DAY",
                   fast_window: int=20, slow_window: int=50, to_cache: bool=False, cache_fpath: str=CACHE_DIR_FPATH
                   ) -> pd.DataFrame:

    from_utc = convert_datetime_api_format(datetime.fromisoformat(from_iso))
    to_utc = convert_datetime_api_format(datetime.fromisoformat(to_iso))
    candles_data = get_candles_data(from_utc, to_utc, ticker, interval, cache_fpath, to_cache)
    candles_df = get_candles_df(candles_data)
    prices = candles_df["close"]

    fast_ma = vbt.MA.run(prices, fast_window)
    slow_ma = vbt.MA.run(prices, slow_window)
    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)
    pf = vbt.Portfolio.from_signals(prices, entries, exits, fees=0.0005, slippage=0.0005, freq='D')

    stats = pf.stats()
    return stats


if __name__ == "__main__":
    print(sma_20_50_eod("BTC.US"))
    print()
    print(sma_n1_n2_tink(YDEX_TICKER, to_cache=True))
