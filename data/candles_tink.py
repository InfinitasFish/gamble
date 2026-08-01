import http.client
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from time import perf_counter
from collections import defaultdict
import json
import os
from typing import Dict, Optional, List
import pandas as pd

from constants import (MAX_WORKERS, REST_API_DOMAIN, READ_ONLY_TOKEN, GET_CANDLES_REST, YDEX_TICKER, CACHE_DIR_FPATH,
                       INTERVAL_TO_MAX_PERIOD, TINK_TIME_PERIODS_IN_DAYS)


def split_time_period(from_iso: str, to_iso: str, max_delta: int) -> List[str]:
    from_date = datetime.fromisoformat(from_iso)
    to_date = datetime.fromisoformat(to_iso)
    diff = to_date - from_date

    if diff <= timedelta(days=0) or max_delta <= 0:
        raise ValueError("to_date is less or equal than to_date")

    split_periods = [datetime.fromisoformat(from_iso).isoformat()]
    num_periods = diff.days // max_delta

    for i in range(num_periods):
        p = from_date + timedelta(days=max_delta * (i + 1))
        split_periods.append(p.isoformat())

    if diff.days % max_delta != 0:
        split_periods.append(datetime.fromisoformat(to_iso).isoformat())

    return split_periods


# https://developer.tbank.ru/invest/api/market-data-service-get-candles
# check this out https://tinkoff.github.io/investAPI/load_history/
# make consecutive request for small intervals with big time periods, collect all the data
def get_candles_data_consecutive(from_iso: str, to_iso: str, instrument_id: str, interval: str="CANDLE_INTERVAL_DAY",
                                 cache_fpath: str=CACHE_DIR_FPATH, to_cache: bool=False) -> dict:

    interval_short = map_api_interval_short(interval)
    save_data_fpath = (f"{instrument_id}_{from_iso.split('T')[0]}_{to_iso.split('T')[0]}_{interval_short}.json"
        .replace(':', '').replace('-', '_'))
    save_data_fpath = os.path.join(cache_fpath, save_data_fpath)

    # search in cache
    cache_dict = try_load_cache_dict(save_data_fpath, test_key="candles")
    if cache_dict is not None:
        return cache_dict

    max_time_delta = INTERVAL_TO_MAX_PERIOD[interval]
    joined_json_dict = {"candles": []}
    # make consecutive reqs concurrently
    if (datetime.fromisoformat(to_iso) - datetime.fromisoformat(from_iso)) > max_time_delta:
        split_periods = split_time_period(from_iso, to_iso, max_time_delta.days)
        tasks = [(split_periods[i - 1], split_periods[i], instrument_id, interval) for i in range(1, len(split_periods))]

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(get_candles_single_request, *t) for t in tasks]
            results = [f.result() for f in futures]

        for r in results:
            if "candles" not in r:
                raise ValueError(f"Unknown response format: {r}")
            joined_json_dict["candles"].extend(r["candles"])

    # make single request
    else:
        period_candle_dict = get_candles_single_request(from_iso, to_iso, instrument_id, interval)
        if "candles" not in period_candle_dict:
            raise ValueError(f"Unknown response format: {period_candle_dict}")

        joined_json_dict["candles"].extend(period_candle_dict["candles"])

    if to_cache:
        with open(save_data_fpath, 'w', encoding="utf-8") as f:
            json.dump(joined_json_dict, f)

    return joined_json_dict


def get_candles_single_request(from_iso: str, to_iso: str, instrument_id: str,
                               interval: str="CANDLE_INTERVAL_DAY") -> dict:
    from_utc = convert_datetime2api_format(datetime.fromisoformat(from_iso))
    to_utc = convert_datetime2api_format(datetime.fromisoformat(to_iso))

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


def convert_datetime2api_format(date_time: datetime) -> str:
    # target is like "2026-03-02T09:15:19.971Z"
    return date_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + 'Z'


if __name__ == "__main__":
    n = 100
    interval = "CANDLE_INTERVAL_10_MIN"
    max_delta = INTERVAL_TO_MAX_PERIOD[interval]
    now = datetime.now().isoformat()
    n_days_ago = (datetime.now() - timedelta(days=n)).isoformat()
    num_reqs = (datetime.fromisoformat(now) - datetime.fromisoformat(n_days_ago)) // max_delta

    s = perf_counter()
    ydex_candle_data = get_candles_data_consecutive(n_days_ago, now, YDEX_TICKER, interval, to_cache=True)
    ydex_candle_df = get_tcandles_df(ydex_candle_data)
    e = perf_counter()
    print(f"{num_reqs} requests done in {e - s:.4f} s.")
    print(ydex_candle_df.head())
