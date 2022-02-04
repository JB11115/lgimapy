from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from lgimapy.data import Database
from lgimapy.utils import root, load_json

# %%
db = Database()

fid = root("data/NFP Tickers.xlsx")
tickers = pd.read_excel(fid).squeeze().values

# %%

accounts = list(load_json("account_strategy").keys())
account_holdings = defaultdict(float)
account = accounts[5]
scraped_accounts = []
for account in tqdm(accounts):
    try:
        acnt = db.load_portfolio(account=account)
    except Exception:
        continue

    if not len(acnt.df):
        continue

    scraped_accounts.append(account)
    for ticker in tickers:
        df = acnt.df[acnt.df["Ticker"] == ticker]
        account_holdings[ticker] += df["P_AssetValue"].sum()


# %%
scraped_accounts_df = pd.Series(scraped_accounts).to_frame()
exposure_df = pd.Series(account_holdings).to_frame()
exposure_df.columns = ["Exposure"]

scraped_accounts_df.to_csv("scraped_accounts.csv")
exposure_df.to_csv("NFP_ticker_exposure.csv")
