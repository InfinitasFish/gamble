import os
from dotenv import load_dotenv
load_dotenv()

# tokens
READ_ONLY_TOKEN = os.environ["READ_ONLY_TOKEN"]

# she bounces on my root until I peak
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

# models
CANDLES_MULTI_TRAINING_FEATURES = ["open", "close", "high", "low"]
CANDLES_MULTI_TARGET_FEATURES = ["open", "close", "high", "low"]
CANDLES_UNI_TARGET_FEATURE = ["close"]
MAX_ITER = 2000
TS_MAX_SEQUENCE_LEN = 10
TS_MIN_SEQUENCE_LEN = 5
TEST_SIZE = 0.33
CV_FOLDS = 10
RANDOM_STATE = 59

# api calls
REST_API_DOMAIN = "invest-public-api.tbank.ru"
BOND_REST = "/rest/tinkoff.public.invest.api.contract.v1.InstrumentsService/Bonds"
BOND_BY_REST = "/rest/tinkoff.public.invest.api.contract.v1.InstrumentsService/BondBy"
GET_CANDLES_REST = "/rest/tinkoff.public.invest.api.contract.v1.MarketDataService/GetCandles"

# actives idxs
YDEX_ISIN = "RU000A107T19"
YDEX_TICKER = "YDEX_TQBR"
X5_TICKER = "X5_TQBR"
VTBR_TICKER = "VTBR_TQBR"
T_TICKER = "T_TQBR"
SBER_TICKER = "SBER_TQBR"

# cache
CACHE_DIR_FPATH = os.path.join(ROOT_DIR, "cache")
BONDS_DATA_FPATH = os.path.join(CACHE_DIR_FPATH, "all_bonds_data.json")

