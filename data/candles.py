import http.client
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import pandas as pd
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants import REST_API_DOMAIN, READ_ONLY_TOKEN, GET_CANDLES_REST, YDEX_TICKER, CACHE_FPATH


def get_daily_candles_df(candles_data: dict) -> pd.DataFrame:
    """parsing json data to pandas dataframe for training"""
    candles_data_for_df = defaultdict(list)
    for candle in candles_data["candles"]:
        candles_data_for_df["open"].append(float(f"{candle['open']['units']}.{candle['open']['nano']}"))
        candles_data_for_df["high"].append(float(f"{candle['high']['units']}.{candle['high']['nano']}"))
        candles_data_for_df["low"].append(float(f"{candle['low']['units']}.{candle['low']['nano']}"))
        candles_data_for_df["close"].append(float(f"{candle['close']['units']}.{candle['close']['nano']}"))
        candles_data_for_df["volume"].append(candle["volume"])
        candles_data_for_df["volumeBuy"].append(candle["volumeBuy"])
        candles_data_for_df["volumeSell"].append(candle["volumeSell"])
        candles_data_for_df["time"].append(candle["time"])

    candles_df = pd.DataFrame.from_dict(candles_data_for_df)
    return candles_df


def inspect_candles_dict(candles_data: dict, space: int=0):
    for i, (key, values) in enumerate(candles_data.items()):
        print(f"{' ' * space}{key}:")
        if isinstance(values, list):
            for item in values:
                inspect_candles_dict(item, space + 2)
        elif isinstance(values, dict):
            inspect_candles_dict(values, space + 2)
        else:
            print(f"{' ' * (space + 2)}{values}")


def map_api_interval_short(interval: str) -> str:
    full_to_short = {"CANDLE_INTERVAL_5_SEC": "5sec", "CANDLE_INTERVAL_10_SEC": "10sec", "CANDLE_INTERVAL_30_SEC": "30sec",
                     "CANDLE_INTERVAL_1_MIN": "1min", "CANDLE_INTERVAL_2_MIN": "2min", "CANDLE_INTERVAL_3_MIN": "3min",
                     "CANDLE_INTERVAL_5_MIN": "5min", "CANDLE_INTERVAL_10_MIN": "10min", "CANDLE_INTERVAL_15_MIN": "15min",
                     "CANDLE_INTERVAL_30_MIN": "30min", "CANDLE_INTERVAL_2_HOUR": "2hour", "CANDLE_INTERVAL_HOUR": "hour",
                     "CANDLE_INTERVAL_4_HOUR": "4hour", "CANDLE_INTERVAL_DAY": "day", "CANDLE_INTERVAL_WEEK": "week",
                     "CANDLE_INTERVAL_MONTH": "month"}
    return full_to_short.get(interval, '')


# https://developer.tbank.ru/invest/api/market-data-service-get-candles
# TODO: at some point better to make these 'await'
def get_candles_data(from_utc: str, to_utc: str, instrument_id: str, interval: str="CANDLE_INTERVAL_DAY", cache_fpath: str=CACHE_FPATH, to_cache: bool=False) -> dict:
    interval_short = map_api_interval_short(interval)
    save_data_fpath = f"{instrument_id}_{interval_short}_{from_utc[:-5]}_{to_utc[:-5]}.json".replace(':', '').replace('-', '_')
    save_data_fpath = os.path.join(cache_fpath, save_data_fpath)
    if os.path.exists(save_data_fpath):
        with open(save_data_fpath, 'r', encoding="utf-8") as f:
            candles_data_dict = json.load(f)
        return candles_data_dict

    connection = http.client.HTTPSConnection(REST_API_DOMAIN)
    payload = json.dumps({
        "from": from_utc,
        "to": to_utc,
        "interval": interval,
        "instrumentId": f"{instrument_id}",
    })

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {READ_ONLY_TOKEN}"
    }

    connection.request("POST", GET_CANDLES_REST, payload, headers)
    response = connection.getresponse()
    candles_data_dict = json.loads(response.read().decode("utf-8"))

    if to_cache:
        with open(save_data_fpath, 'w', encoding="utf-8") as f:
            json.dump(candles_data_dict, f)

    return candles_data_dict


def load_candles_data(candles_data_fpath: str) -> dict:
    with open(candles_data_fpath, 'r', encoding="utf-8") as f:
        candles_data_dict = json.load(f)
    return candles_data_dict


def convert_datetime_api_format(datetime: datetime) -> str:
    # target is like  "2026-03-02T09:15:19.971Z"
    return datetime.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + 'Z'


if __name__ == "__main__":
    now = convert_datetime_api_format(datetime.now(timezone.utc))
    ten_days_ago = convert_datetime_api_format(datetime.now(timezone.utc) - timedelta(days=10))

    ydex_candle_data = get_candles_data(ten_days_ago, now, YDEX_TICKER)
    print(ydex_candle_data)
