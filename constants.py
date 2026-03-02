import os
from dotenv import load_dotenv
load_dotenv()


READ_ONLY_TOKEN = os.environ["READ_ONLY_TOKEN"]

REST_API_DOMAIN = "invest-public-api.tbank.ru"
BOND_REST = "/rest/tinkoff.public.invest.api.contract.v1.InstrumentsService/Bonds"
BOND_BY_REST = "/rest/tinkoff.public.invest.api.contract.v1.InstrumentsService/BondBy"
GET_CANDLES_REST = "/rest/tinkoff.public.invest.api.contract.v1.MarketDataService/GetCandles"


YDEX_ISIN = "RU000A107T19"
YDEX_TICKER = "YDEX_TQBR"

