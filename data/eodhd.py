# popular financial data api
import requests
import websocket
import json
from datetime import datetime
import pandas as pd

from constants import EODHD_API_TOKEN, FROM_ISO, TO_ISO


def download_symbol_candles(symbol: str, from_iso: str=FROM_ISO, to_iso: str=TO_ISO, period: str='d',
                            api_token: str=EODHD_API_TOKEN) -> pd.DataFrame:
    # period can be 'd' for daily, 'w' for weekly, 'm' for monthly

    url = f"https://eodhd.com/api/eod/{symbol}?api_token={api_token}&from={from_iso}&to={to_iso}&period={period}&fmt=json"
    resp = requests.get(url, timeout=15)
    if resp.status_code != 200:
        raise RuntimeError(f"Bad request {resp.status_code} : {url}")

    df = pd.DataFrame(resp.json())
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')
    return df


def download_symbol_intraday_candles(symbol: str, from_iso: str=FROM_ISO, to_iso: str=TO_ISO, interval: str="5m",
                                     api_token: str=EODHD_API_TOKEN) -> pd.DataFrame:
    # unfortunately doesn't work without subscription

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

    df = pd.DataFrame(resp.json())
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').set_index('datetime')
    return df


if __name__ == "__main__":
    print(download_symbol_candles("AAPL.US").head())
