from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from lgimapy.data import Database

# %%
db = Database()

d = defaultdict(list)
accounts = ["FLD", "CHLD", "GTMC", "CITMC"]
dates = db.trade_dates(start=db.date("YTD"))
for date in tqdm(dates):
    for account in accounts:
        acnt = db.load_portfolio(account=account, date=date)
        d[account].append(acnt.dts("pct"))


# %%
df = pd.DataFrame(d, index=dates)

# %%
db.load_market_data(start=dates[0])
ix = db.build_market_index(in_returns_index=True)
lc_ix = ix.subset(maturity=(10, None))
df["Market_Credit_OAS"] = ix.OAS()
df["Long_Credit_OAS"] = lc_ix.OAS()

df.to_csv("YTD_account_DTS.csv")
