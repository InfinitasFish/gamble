# my level
# mean-reversion works only on assets without strong trend (low ADX)
# can work on: sideways market, low ADX (under 20), pairs, volatility ETF's
# won't work on: trending market, high ADX (over 30), news-driven assets

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data.candles_tink import get_candles_data_consecutive, get_tcandles_df
from constants import YDEX_TICKER


def wilders_smooth_values(timeseries: pd.Series, period: int) -> pd.Series:
    sval = [np.nan] * (period - 1)
    sval.append(timeseries.iloc[:period].sum())
    for i in range(period, len(timeseries)):
        sval.append(sval[i - 1] - sval[i - 1] / period + timeseries.iloc[i])
    return pd.Series(sval)


def calculate_adx(timeseries_df: pd.DataFrame, period: int=14) -> pd.Series:
    # calc directional movements (DM)
    plus_dm = (timeseries_df["high"] - timeseries_df["high"].shift(1))
    plus_dm[plus_dm < 0] = 0

    minus_dm = (timeseries_df["low"].shift(1) - timeseries_df["low"])
    minus_dm[minus_dm < 0] = 0

    plus_greater = plus_dm >= minus_dm
    minus_greater = minus_dm >= plus_dm
    plus_dm[minus_greater] = 0
    minus_dm[plus_greater] = 0

    # calculate true range (TR)
    spread_range = (timeseries_df["high"] - timeseries_df["low"]).abs()
    volatile_range = (timeseries_df["high"] - timeseries_df["close"].shift(1)).abs()
    trend_range = (timeseries_df["low"] - timeseries_df["close"].shift(1)).abs()
    true_range = pd.concat([spread_range, volatile_range, trend_range], axis=1).max(axis=1)

    # smooth DMs and TR over period
    plus_dm_sm = wilders_smooth_values(plus_dm, period)
    minus_dm_sm = wilders_smooth_values(minus_dm, period)
    true_range_sm = wilders_smooth_values(true_range, period)

    # calculate Directional Indicators (DI)
    plus_di = plus_dm_sm / true_range_sm * 100
    minus_di = minus_dm_sm / true_range_sm * 100

    # calculate Directional Movement index (DX) and Average DX (ADX)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
    adx = wilders_smooth_values(dx, period) / period
    return adx


if __name__ == "__main__":

    json_data = get_candles_data_consecutive("2026-01-01", "2026-09-01", YDEX_TICKER, to_cache=True)
    df = get_tcandles_df(json_data)
    # print(df.head())
    # print(df.info())
    # print(df.describe())

    adx_indicator = calculate_adx(df)
    plt.plot(np.arange(len(df["time"])), (df["close"] - df["close"].min()) / 50, color="red")
    plt.plot(np.arange(len(df["time"])), adx_indicator)
    plt.show()

