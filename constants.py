import os
from dotenv import load_dotenv

load_dotenv()
from datetime import timedelta

# tokens
READ_ONLY_TOKEN = os.environ["READ_ONLY_TOKEN"]
EODHD_API_TOKEN = os.environ["EODHD_API_TOKEN"]

# she bounces on my root until I peak
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
MAX_WORKERS = 4

# models
CANDLES_MULTI_TRAINING_FEATURES = ["open", "close", "high", "low"]
CANDLES_MULTI_TARGET_FEATURES = ["open", "close", "high", "low"]
CANDLES_UNI_TARGET_FEATURE = ["close"]
MAX_ITER = 2000
TS_MAX_SEQUENCE_LEN = 10
TS_MIN_SEQUENCE_LEN = 1
TEST_SIZE = 0.33
CV_FOLDS = 10
RANDOM_STATE = 59

# api calls
REST_API_DOMAIN = "invest-public-api.tbank.ru"
BOND_REST = "/rest/tinkoff.public.invest.api.contract.v1.InstrumentsService/Bonds"
BOND_BY_REST = "/rest/tinkoff.public.invest.api.contract.v1.InstrumentsService/BondBy"
GET_CANDLES_REST = "/rest/tinkoff.public.invest.api.contract.v1.MarketDataService/GetCandles"
FROM_ISO = "2025-01-01"
TO_ISO = "2026-01-01"

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

# utils
REG_METRICS_USER = ["R2 Score", "Root Mean Squared Error", "Mean Absolute Error", "Mean Absolute Percentage Error",
               "Median Absolute Error",]
REG_METRICS_SKLEARN = ["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error", "neg_mean_absolute_percentage_error",
                       "neg_median_absolute_error"]
REG_METRICS_TO_SKLEARN = {k: v for k, v in zip(REG_METRICS_USER, REG_METRICS_SKLEARN)}

TINK_INTERVALS = ["CANDLE_INTERVAL_5_MIN", "CANDLE_INTERVAL_10_MIN", "CANDLE_INTERVAL_15_MIN", "CANDLE_INTERVAL_30_MIN",
                  "CANDLE_INTERVAL_HOUR", "CANDLE_INTERVAL_2_HOUR", "CANDLE_INTERVAL_4_HOUR", "CANDLE_INTERVAL_DAY",
                  "CANDLE_INTERVAL_WEEK", "CANDLE_INTERVAL_MONTH"]

TINK_TIME_PERIODS_IN_DAYS = [timedelta(days=1), timedelta(days=1), timedelta(days=1), timedelta(days=2), timedelta(days=7),
                             timedelta(days=30), timedelta(days=30), timedelta(days=365), timedelta(days=365 * 2),
                             timedelta(days=365 * 10)]

INTERVAL_TO_MAX_PERIOD = {k: v for k, v in zip(TINK_INTERVALS, TINK_TIME_PERIODS_IN_DAYS)}
