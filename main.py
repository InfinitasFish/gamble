# tried to use it, but I don't understand how to pass SLL cert (Russian Trusted Root CA)
# import requests
# while http.client just works natively
import http.client
import json

from constants import (READ_ONLY_TOKEN, REST_API_DOMAIN, BOND_REST, BOND_BY_REST, YDEX_ISIN)


# todo: not working idk why..
# probably api call is just wrong, shouldn't be bonds_by call
# this looks more like it
# https://developer.tbank.ru/invest/api/market-data-service-get-candles
def get_ydex_data():
    conn = http.client.HTTPSConnection(REST_API_DOMAIN)

    payload = json.dumps({
        "idType": "INSTRUMENT_ID_TYPE_TICKER",
        "classCode": "TQBR",
        "id": f"YDEX"
    })

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {READ_ONLY_TOKEN}'
    }

    conn.request('POST', BOND_BY_REST, payload, headers)
    res = conn.getresponse()
    data = res.read()
    print(data.decode('utf-8'))


def get_bonds_data():
    conn = http.client.HTTPSConnection(REST_API_DOMAIN)
    payload = json.dumps({
        "instrumentStatus": "INSTRUMENT_STATUS_BASE",
        "instrumentExchange": "INSTRUMENT_EXCHANGE_UNSPECIFIED"
    })
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {READ_ONLY_TOKEN}'
    }
    conn.request("POST", BOND_REST, payload, headers)
    res = conn.getresponse()
    data = res.read()
    with open('bonds_data.txt', 'w', encoding='utf-8') as f:
        f.write(data.decode('utf-8'))



def main():
    # get_bonds_data()
    get_ydex_data()


if __name__ == '__main__':
    main()


