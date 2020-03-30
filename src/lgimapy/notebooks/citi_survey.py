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
    kwargs[key] = {k: val[k] for k in set(val.keys()) - set(["in_stats_index"])}


def get_position(oad=None, dts=None):
    """Return position on scale of -2 to 2 given current OAD overweight"""
    if oad is not None:
        val = oad
        levels = [0.24, 0.08, -0.08, -0.24]
    elif dts is not None:
        val = dts
        levels = [1, 0.975, 0.945, 0.92]

    if val > levels[0]:
        return "+2"
    elif val > levels[1]:
        return "+1"
    elif val > levels[2]:
        return "0"
    elif val > levels[3]:
        return "-1"
    else:
        return "-2"


# Get position of P-LD account sectors.
pld_sectors = [
    "TELECOM",
    "CYCLICAL_FOR_CITI_SURVEY",
    "CONSUMER_NON_CYCLICAL",
    "UTILITY",
    "INDUSTRIALS_FOR_CITI_SURVEY",
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
    positions[kwargs[sector]["name"]] = get_position(oad=oad)

# Get positions for overall credit.
dts = np.sum(df["P_DTS"]) / np.sum(df["BM_DTS"])
positions["Overall High Grade Corp"] = get_position(dts=dts)
positions["Overall in Credit"] = get_position(dts=dts)

# Get position for ABS/CMBS.
df = db.load_portfolio(accounts="P-MC", market_cols=True)
ix = db.build_market_index(**kwargs["ABS_CMBS"])
oad = np.sum(ix.df["OAD_Diff"])
positions[kwargs["ABS_CMBS"]["name"]] = get_position(oad=oad)

# Add Nans.
positions["Hybrids"] = "N/A"
positions["Overall High Yield"] = "N/A"
positions["Index/ETF Options"] = "N/A"

# Print result.
sorted_ix = [
    "Telecom",
    "Cyclical (Citi)",
    "Consumer Non-Cyclical",
    "Utilities",
    "Industrials (Citi)",
    "Energy",
    "Banks (Sr)",
    "Banks (Sub)",
    "Emerging Markets",
    "Hybrids",
    "Insurance",
    "Overall High Grade Corp",
    "Overall High Yield",
    "Index/ETF Options",
    "Munis",
    "ABS/CMBS/Non-Agency RMBS",
    "Overall in Credit",
]

print(pd.Series(positions).reindex(sorted_ix))
