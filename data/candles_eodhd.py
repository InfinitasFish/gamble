# popular financial data api
import os
import requests
import json
from datetime import datetime
import pandas as pd

from data.candles_tink import try_load_cache_dict
from constants import EODHD_API_TOKEN, FROM_ISO, TO_ISO, CACHE_DIR_FPATH


def download_symbol_candles(symbol: str, from_iso: str=FROM_ISO, to_iso: str=TO_ISO, period: str='d',
                            to_cache: bool=False, cache_dir: str=CACHE_DIR_FPATH,
                            api_token: str=EODHD_API_TOKEN) -> pd.DataFrame:
    # period can be 'd' for daily, 'w' for weekly, 'm' for monthly

    cache_fpath = f"{symbol.replace('.', '_')}_{from_iso}_{to_iso}_{period}.json".replace('-', '_')
    cache_fpath = os.path.join(cache_dir, cache_fpath)

    cache_dict = try_load_cache_dict(cache_fpath, test_key="close")
    if cache_dict is not None:
        df = get_eod_candles_df(cache_dict)
        return df

    url = f"https://eodhd.com/api/eod/{symbol}?api_token={api_token}&from={from_iso}&to={to_iso}&period={period}&fmt=json"
    resp = requests.get(url, timeout=15)
    if resp.status_code != 200:
        raise RuntimeError(f"Bad request {resp.status_code} : {url}")
    resp_json = resp.json()

    if to_cache and not os.path.exists(cache_fpath):
        with open(cache_fpath, 'w', encoding="utf-8") as f:
            json.dump(resp_json, f)

    df = get_eod_candles_df(resp_json)
    return df


def download_symbol_intraday_candles(symbol: str, from_iso: str=FROM_ISO, to_iso: str=TO_ISO, interval: str="5m",
                                     api_token: str=EODHD_API_TOKEN) -> pd.DataFrame:
    # todo: unfortunately doesn't work without subscription. search for alternatives?

    # a bit different url for intraday parsing
    # interval can be "1m", "5m", "1h"
    # max days for "1m" is 120 days btw

    # intraday endpoint uses unix timestamps
    from_iso = datetime.timestamp(datetime.fromisoformat(from_iso))
    to_iso = datetime.timestamp(datetime.fromisoformat(to_iso))

    url = f"https://eodhd.com/api/intraday/{symbol}?api_token={api_token}&interval={interval}&from={from_iso}&to={to_iso}&fmt=json"
    resp = requests.get(url, timeout=15)
    if resp.status_code != 200:
        raise RuntimeError(f"Bad request {resp.status_code} : {resp.content}")
    resp_json = resp.json()

    df = get_eod_candles_df(resp_json)
    return df


def get_eod_candles_df(candles_data: dict) -> pd.DataFrame:
    df = pd.DataFrame(candles_data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    return df


# btw, using websockets we can stream data live


if __name__ == "__main__":
    print(download_symbol_candles("AAPL.US", to_cache=True).head())
