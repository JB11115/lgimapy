from collections import defaultdict

import pandas as pd

from lgimapy.data import Database

# %%
ticker = 'COLOM'
date = '7/1/2021'


d = defaultdict(list)
rep_accounts = "CHLD CITMC USBGC JOYLA AEELG GSKLC".split()
for account in rep_accounts:
    port = db.load_portfolio(account=account, date=date)
    d["Strategy"].append(db._account_to_strategy[account])
    d["Rep Account"].append(account)
    ticker_ix = port.subset(ticker=ticker)
    d["Port CTD"].append(ticker_ix.df["P_OAD"].sum())
    d["BM CTD"].append(ticker_ix.df["BM_OAD"].sum())
    d["CTD Difference"].append(ticker_ix.df["OAD_Diff"].sum())
    d["Port MV Weight (%)"].append(100 * ticker_ix.df["P_Weight"].sum())
    d["BM MV Weight (%)"].append(100 * ticker_ix.df["BM_Weight"].sum())
    d["MV Weight (%) Difference"].append(
        100 * ticker_ix.df["Weight_Diff"].sum()
    )

df = pd.DataFrame(d)
df.to_csv(f"{ticker}_downgrade_rep_account_holdings.csv")
