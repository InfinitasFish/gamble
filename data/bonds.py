# tried to use it, but I don't understand how to pass SLL cert (Russian Trusted Root CA)
# import requests
# while http.client just works natively
import http.client
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants import REST_API_DOMAIN, READ_ONLY_TOKEN, BOND_REST, BONDS_DATA_FPATH


# https://developer.tbank.ru/invest/api/instruments-service-bonds
def get_all_bonds_data(cache_fpath: str=BONDS_DATA_FPATH, to_cache: bool=True) -> dict:
    if os.path.exists(cache_fpath):
        with open(cache_fpath, 'r', encoding="utf-8") as f:
            bonds_data_dict = json.load(f)
        return bonds_data_dict

    connection = http.client.HTTPSConnection(REST_API_DOMAIN)
    payload = json.dumps({
        "instrumentStatus": "INSTRUMENT_STATUS_BASE",
        "instrumentExchange": "INSTRUMENT_EXCHANGE_UNSPECIFIED"
    })
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {READ_ONLY_TOKEN}"
    }
    connection.request("POST", BOND_REST, payload, headers)
    response = connection.getresponse()
    bonds_data_dict = json.loads(response.read().decode("utf-8"))

    if to_cache:
        with open(cache_fpath, 'w', encoding="utf-8") as f:
            json.dump(bonds_data_dict, f)

    return bonds_data_dict


if __name__ == "__main__":
    print(get_all_bonds_data(cache_fpath=BONDS_DATA_FPATH))

