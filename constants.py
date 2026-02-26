import os
from dotenv import load_dotenv
load_dotenv()



READ_ONLY_TOKEN = os.environ['READ_ONLY_TOKEN']

# TINK_GRPC = 'invest-public-api.tbank.ru:443'
BOND_BY = 'https://invest-public-api.tbank.ru/rest/tinkoff.public.invest.api.contract.v1.InstrumentsService/BondBy'

YDEX_ISIN = 'RU000A107T19'

