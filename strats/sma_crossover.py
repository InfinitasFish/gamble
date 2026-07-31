from datetime import datetime
from typing import List

import pandas as pd
import numpy as np
import vectorbt as vbt

from data.candles_tink import get_candles_data, get_tcandles_df, convert_datetime2api_format
from data.candles_eodhd import download_symbol_candles
from strats import SignalStrategyInterface
from constants import CACHE_DIR_FPATH, FROM_ISO, TO_ISO, EODHD_API_TOKEN, YDEX_TICKER, TINK_INTERVALS


# proper object for experiments on tink
class SMACrossover(SignalStrategyInterface):
    def __init__(self, ticker: str, fast_ma: int, slow_ma: int, interval: str):
        self.ticker = ticker
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.interval = interval

    def train(self, from_iso: str, to_iso:str, interval: str, to_cache: bool=False, cache_fpath: str=CACHE_DIR_FPATH) -> pd.DataFrame:
        from_utc = convert_datetime2api_format(datetime.fromisoformat(from_iso))
        to_utc = convert_datetime2api_format(datetime.fromisoformat(to_iso))
        candles_data = get_candles_data(from_utc, to_utc, self.ticker, interval, cache_fpath, to_cache)
        candles_df = get_tcandles_df(candles_data)
        prices = candles_df["close"]

        fast_ma = vbt.MA.run(prices, self.fast_ma)
        slow_ma = vbt.MA.run(prices, self.slow_ma)
        entries = fast_ma.ma_crossed_above(slow_ma)
        exits = fast_ma.ma_crossed_below(slow_ma)
        pf = vbt.Portfolio.from_signals(prices, entries, exits, fees=0.0005, slippage=0.0005, freq='D')

        stats = pf.stats()
        return stats

    # check this out https://tinkoff.github.io/investAPI/load_history/
    # so for smaller interval we have to set smaller time period
    def tink_test_intervals(self, ticker: str, from_iso: str, to_iso: str, intervals: List[str], metric: str) -> str:
        best_interval = intervals[0]
        best_metric = -float("inf")
        for interval_ in intervals:
            current_metric = self.train(from_iso, to_iso, interval_)[metric]
            if current_metric > best_metric:
                print(f"found new best interval for sma{self.fast_ma}-{self.slow_ma}: {interval_}\n\t{metric}: {current_metric:.6f}")
                best_metric = current_metric
                best_interval = interval_

        return best_interval


# eod example
def sma_20_50_eod(symbol: str, from_iso: str=FROM_ISO, to_iso: str=TO_ISO, period: str='d',
                  api_token: str=EODHD_API_TOKEN) -> pd.DataFrame:

    prices_df = download_symbol_candles(symbol, from_iso, to_iso, period, api_token=api_token)
    prices = prices_df["close"]

    fast_ma = vbt.MA.run(prices, 20)
    slow_ma = vbt.MA.run(prices, 50)
    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)

    pf = vbt.Portfolio.from_signals(prices, entries, exits, fees=0.0005, slippage=0.0005, freq='D')
    stats = pf.stats()
    return stats


if __name__ == "__main__":
    s = SMACrossover(YDEX_TICKER, 20, 50, "CANDLE_INTERVAL_30_MIN")
    intervals = TINK_INTERVALS
    print(s.tink_test_intervals(YDEX_TICKER, "2026-06-01", "2026-07-01", intervals, "Expectancy"))
