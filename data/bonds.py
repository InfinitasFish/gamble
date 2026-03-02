import http.client
import json
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from constants import REST_API_DOMAIN, READ_ONLY_TOKEN, BOND_REST


# https://developer.tbank.ru/invest/api/instruments-service-bonds
def get_all_bonds_data(save_fpath):
    conn = http.client.HTTPSConnection(REST_API_DOMAIN)
    payload = json.dumps({
        "instrumentStatus": "INSTRUMENT_STATUS_BASE",
        "instrumentExchange": "INSTRUMENT_EXCHANGE_UNSPECIFIED"
    })
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {READ_ONLY_TOKEN}"
    }
    conn.request("POST", BOND_REST, payload, headers)
    res = conn.getresponse()
    data = res.read()
    with open(save_fpath, "w", encoding="utf-8") as f:
        f.write(data.decode("utf-8"))


if __name__ == "__main__":
    get_all_bonds_data(save_fpath="./bonds_data.txt")
