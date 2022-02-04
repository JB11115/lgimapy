import pandas as pd

from lgimapy.data import Database

# %%
account = "P-LD"
db = Database()
acnt = db.load_portfolio(account=account, universe="stats")

# %%
df = (
    acnt.df.groupby("Ticker", observed=True)
    .sum()
    .sort_values("BM_Weight", ascending=False)
    .iloc[:50, :]
)
df.sum()[["BM_Weight", "BM_OAD", "BM_DTS", "P_Weight", "P_OAD", "P_DTS"]]


acnt.df["BM_OAD"].sum()
col = "OAD_Diff"
acnt.df[col].abs().sum()
df[col].abs().sum()
(df[col] ** 2).sum()
(acnt.df[col] ** 2).sum()


df["DTS_Diff"]

pd.Series(df.index).to_frame().to_csv("top50_tickers.csv")
