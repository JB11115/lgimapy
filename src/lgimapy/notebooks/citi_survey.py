from collections import defaultdict
from datetime import timedelta

import numpy as np
import pandas as pd

from lgimapy.data import Database, Index
from lgimapy.utils import load_json, root, Time

# %%
positions = {}
db = Database()

# Get sector kwargs.
raw_kwargs = load_json("indexes")
kwargs = {}
for key, val in raw_kwargs.items():
    kwargs[key] = {
        k: val[k] for k in set(val.keys()) - set(["in_stats_index"])
    }


def get_position(oad=None, dts=None):
    """Return position on scale of -2 to 2 given current OAD overweight"""
    if oad is not None:
        val = oad
        levels = [0.2, 0.1, -0.1, -0.2]
    elif dts is not None:
        val = dts
        levels = [1, 0.98, 0.94, 0.92]

    if val > levels[0]:
        return '+2'
    elif val > levels[1]:
        return '+1'
    elif val > levels[2]:
        return '0'
    elif val > levels[3]:
        return '-1'
    else:
        return '-2'


# Get position of P-LD account sectors.
pld_sectors = [
    "TELECOM",
    "CONSUMER_CYCLICAL",
    "CONSUMER_NON_CYCLICAL",
    "UTILITY",
    "INDUSTRIALS",
    "ENERGY",
    "BANKS_SR",
    "BANKS_SUB",
    "EMERGING_MARKETS",
    "INSURANCE",
    "MUNIS",
]
df = db.load_portfolio(
    accounts="P-LD", market_cols=True, drop_treasuries=False, drop_cash=False
)
db.load_market_data(data=df)
for sector in pld_sectors:
    ix = db.build_market_index(**kwargs[sector])
    oad = np.sum(ix.df["OAD_Diff"])
    positions[kwargs[sector]['name']] = get_position(oad=oad)

# Get positions for overall credit.
dts = np.sum(df['P_DTS']) / np.sum(df['BM_DTS'])
positions['Overall High grade corp'] = get_position(dts=dts)
positions['Overall in credit'] = get_position(dts=dts)

# Get position for ABS/CMBS.
df = db.load_portfolio(accounts="P-MC", market_cols=True)
ix = db.build_market_index(**kwargs['ABS_CMBS'])
oad = np.sum(ix.df["OAD_Diff"])
positions[kwargs['ABS_CMBS']['name']] = get_position(oad=oad)

# Add Nans.
positions['Hybrids'] = 'N/A'
positions['Overall High Yield'] = 'N/A'
positions['Index/ETF Options'] = 'N/A'

# Print result.
sorted_ix = [
    'Telcom',
    'Consumer Cyclical',
    'Consumer Non-Cyclical',
    'Utilities',
    'Industrials',
    'Energy',
    'Banks (Sr)',
    'Banks (Sub)',
    'Emergin Markets',
    'Hybrids',
    'Insurance',
    'Overall High grade corp',
    'Overall High Yield',
    'Index/ETF Options',
    'Munis',
    'ABS/CMBS/Non-Agency RMBS',
    'Overall in credit',
]

print(pd.Series(positions).loc[sorted_ix])
