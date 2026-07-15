import http.client
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import json
import os
from typing import Dict, Optional, List
import pandas as pd

from constants import REST_API_DOMAIN, READ_ONLY_TOKEN, GET_CANDLES_REST, YDEX_TICKER, CACHE_DIR_FPATH, INTERVAL_TO_MAX_PERIOD


def split_time_period(from_iso: str, to_iso: str, max_delta: int) -> List[str]:
    from_date = datetime.fromisoformat(from_iso)
    to_date = datetime.fromisoformat(to_iso)
    diff = to_date - from_date
    split_periods = [from_iso]
    num_periods = diff.days // max_delta + 1

    for i in range(num_periods):
        if i + 1 != num_periods:
            p = from_date + timedelta(days=max_delta * (i + 1))
        else:
            tail = diff.days - max_delta * i
            p = from_date + timedelta(days=max_delta * i + tail)
        split_periods.append(p.isoformat())

    return split_periods


# https://developer.tbank.ru/invest/api/market-data-service-get-candles
# check this out https://tinkoff.github.io/investAPI/load_history/
# make consecutive request for small intervals with big time periods, collect all the data
# TODO: implement parallel requesting with mutex
def get_candles_data_consecutive(from_iso: str, to_iso: str, instrument_id: str, interval: str="CANDLE_INTERVAL_DAY",
                                 cache_fpath: str=CACHE_DIR_FPATH, to_cache: bool=False) -> dict:

    interval_short = map_api_interval_short(interval)
    save_data_fpath = f"{instrument_id}_{from_iso}_{to_iso}_{interval_short}.json".replace(':', '').replace(
        '-', '_')
    save_data_fpath = os.path.join(cache_fpath, save_data_fpath)

    # search in cache
    cache_dict = try_load_cache_dict(save_data_fpath, test_key="candles")
    if cache_dict is not None:
        return cache_dict

    max_time_delta = INTERVAL_TO_MAX_PERIOD[interval]
    joined_json_dict = {"candles": []}
    # have to make consecutive reqs
    if (datetime.fromisoformat(to_iso) - datetime.fromisoformat(from_iso)) > max_time_delta:
        split_periods = split_time_period(from_iso, to_iso, max_time_delta.days)

        for i in range(1, len(split_periods)):
            from_period = split_periods[i - 1]
            to_period = split_periods[i]

            period_candle_dict = get_candles_single_request(from_period, to_period, instrument_id, interval)
            if "candles" not in period_candle_dict:
                raise ValueError(f"Unknown response format: {period_candle_dict}")

            joined_json_dict["candles"].extend(period_candle_dict["candles"])
    # make single request
    else:
        period_candle_dict = get_candles_single_request(from_iso, to_iso, instrument_id, interval)
        if "candles" not in period_candle_dict:
            raise ValueError(f"Unknown response format: {period_candle_dict}")

        joined_json_dict["candles"].extend(period_candle_dict["candles"])

    if to_cache and not os.path.exists(save_data_fpath):
        with open(save_data_fpath, 'w', encoding="utf-8") as f:
            json.dump(joined_json_dict, f)

    return joined_json_dict


def get_candles_single_request(from_iso: str, to_iso: str, instrument_id: str,
                               interval: str="CANDLE_INTERVAL_DAY") -> dict:
    from_utc = convert_datetime_api_format(datetime.fromisoformat(from_iso))
    to_utc = convert_datetime_api_format(datetime.fromisoformat(to_iso))

    # make request
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

    return candles_data_dict


def get_tcandles_df(candles_data: dict, select_features: list=None) -> pd.DataFrame:
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
    if select_features is not None:
        candles_df = candles_df[select_features]
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


def try_load_cache_dict(file_path: str, test_key: str) -> Optional[Dict]:
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding="utf-8") as f:
            candles_data_dict = json.load(f)
        if test_key in candles_data_dict:
            return candles_data_dict
        # cache is invalid, delete and download again
        else:
            os.remove(file_path)
            return None


def convert_datetime_api_format(date_time: datetime) -> str:
    # target is like "2026-03-02T09:15:19.971Z"
    return date_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + 'Z'


if __name__ == "__main__":
    now = datetime.now().isoformat()
    ten_days_ago = (datetime.now() - timedelta(days=10)).isoformat()

    ydex_candle_data = get_candles_data(ten_days_ago, now, YDEX_TICKER, to_cache=True)
    ydex_candle_df = get_tcandles_df(ydex_candle_data)
    print(ydex_candle_df.head())
