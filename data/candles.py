import http.client
from datetime import datetime, timezone, timedelta
import json
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from constants import REST_API_DOMAIN, READ_ONLY_TOKEN, GET_CANDLES_REST, YDEX_TICKER


# https://developer.tbank.ru/invest/api/market-data-service-get-candles
def get_ydex_candles_data(from_utc, to_utc, interval="CANDLE_INTERVAL_DAY"):
    conn = http.client.HTTPSConnection(REST_API_DOMAIN)

    payload = json.dumps({
        "to": to_utc,
        "from": "2026-02-02T09:15:19.971Z",
        "interval": interval,
        "instrumentId": f"{YDEX_TICKER}",
    })

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {READ_ONLY_TOKEN}"
    }

    conn.request("POST", GET_CANDLES_REST, payload, headers)
    res = conn.getresponse()
    data = res.read()
    print(data.decode("utf-8"))


if __name__ == "__main__":
    now = datetime.now(timezone.utc)
    ten_days_ago = now - timedelta(days=10)
    # target is like "2026-03-02T09:15:19.971Z", but 0 idea what does .971 means, milliseconds?
    now_for_api = now.strftime('%Y-%m-%dT%H:%M:%SZ')
    print(now)
    print(now_for_api)

    get_ydex_candles_data(str(ten_days_ago), now_for_api)
