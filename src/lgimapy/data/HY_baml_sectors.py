from collections import defaultdict

from lgimapy.data import Database, groupby
from lgimapy.utils import dump_json

# %%

dates = [
    '6/1/2020',
    '8/25/2020',
    '1/4/2021',
    '3/1/2021',
    '6/1/2021',
]

baml_cols = ['BAMLSector', 'BAMLTopLevelSector']
db = Database()
d = defaultdict(dict)
for date in dates:
    db.load_market_data(date)
    df = db.build_market_index().df
    df = df[(~df['BAMLSector'].isna()) | (~df['BAMLTopLevelSector'].isna())]
    ticker_df = groupby(df[['Ticker', *baml_cols]], 'Ticker')
    for col in baml_cols:
        # Update ticker: sector combination, overwriting older dates
        # with newer ones.
        d[col] = {**d[col], **ticker_df[col].to_dict()}

for col, ticker_sector_d in d.items():
    dump_json(ticker_sector_d, f"HY/ticker_{col}_map")
