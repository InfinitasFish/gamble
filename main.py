# tried to use it, but I don't understand how to pass SLL cert (Russian Trusted Root CA)
# import requests
# while http.client just works natively
import http.client
import json

from constants import (READ_ONLY_TOKEN, REST_API_DOMAIN, BOND_REST, BOND_BY_REST, YDEX_ISIN)



def main():
    get_ydex_candles_data()


if __name__ == '__main__':
    main()


