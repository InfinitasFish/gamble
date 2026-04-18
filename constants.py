import os
from dotenv import load_dotenv
load_dotenv()

# she bounces on my root until I peak
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

# tokens
READ_ONLY_TOKEN = os.environ["READ_ONLY_TOKEN"]

# api calls
REST_API_DOMAIN = "invest-public-api.tbank.ru"
BOND_REST = "/rest/tinkoff.public.invest.api.contract.v1.InstrumentsService/Bonds"
BOND_BY_REST = "/rest/tinkoff.public.invest.api.contract.v1.InstrumentsService/BondBy"
GET_CANDLES_REST = "/rest/tinkoff.public.invest.api.contract.v1.MarketDataService/GetCandles"

# actives idxs
YDEX_ISIN = "RU000A107T19"
YDEX_TICKER = "YDEX_TQBR"

# cache
CACHE_FPATH = os.path.join(ROOT_DIR, "cache")
BONDS_DATA_FPATH = os.path.join(CACHE_FPATH, "all_bonds_data.json")

