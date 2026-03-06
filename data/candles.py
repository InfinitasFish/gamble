import http.client
from datetime import datetime, timezone, timedelta
import pandas as pd
import json
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from constants import REST_API_DOMAIN, READ_ONLY_TOKEN, GET_CANDLES_REST, YDEX_TICKER


def process_daily_candles(candles_json):
    """parsing json data to pandas dataframe"""
    print(candles_json.keys())


# https://developer.tbank.ru/invest/api/market-data-service-get-candles
def get_candles_data(from_utc, to_utc, instrument_id, interval="CANDLE_INTERVAL_DAY") -> dict:
    conn = http.client.HTTPSConnection(REST_API_DOMAIN)

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

    conn.request("POST", GET_CANDLES_REST, payload, headers)
    res = conn.getresponse()
    data = res.read()
    data = data.decode("utf-8")

    # json string to dictionary
    return json.loads(data)


def get_datef(datetime):
    # target is like  "2026-03-02T09:15:19.971Z"
    return datetime.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'


if __name__ == "__main__":
    now = datetime.now(timezone.utc)
    ten_days_ago = now - timedelta(days=10)

    data = get_candles_data(get_datef(ten_days_ago), get_datef(now), YDEX_TICKER)
    process_daily_candles(data)
