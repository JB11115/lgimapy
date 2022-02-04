import pandas as pd

from lgimapy.data import Database

# %%
db = Database()
port = db.load_portfolio(account="CITMC")
bm_df = port.df[port.df["BM_Weight"] > 0]

# %%
bm_df
bm_df["MaturityYears"]
bm_df.head()
tickers_over_10 = set()
for _, row in bm_df.iterrows():
    if row["MaturityYears"] > 10:
        tickers_over_10.add(row["Ticker"])


# %%
tickers_under_10 = set(bm_df["Ticker"].unique()) - tickers_over_10
u10_df = bm_df[bm_df["Ticker"].isin(tickers_under_10)]
issuer_pct = len(u10_df) / len(bm_df)
mv_pct = u10_df["BM_Weight"].sum()
print(f"{issuer_pct:.1%} of issuers and {mv_pct:.1%} of Market Value")

# %%
port_ytd = db.load_portfolio(account="CITMC", date=db.date("ytd"))
ytd_df = port_ytd.ticker_df
ye_oas = ytd_df[ytd_df.index.isin(tickers_under_10)]["OAS"].rename("YE OAS")

curr_df = port.ticker_df
curr_oas = curr_df[curr_df.index.isin(tickers_under_10)]["OAS"].rename(
    "Current OAS"
)

ticker_df = pd.concat((curr_oas, ye_oas), axis=1).rename_axis(None)
ticker_df["YTD Chg OAS"] = ticker_df["Current OAS"] - ticker_df["YE OAS"]
ticker_df.to_csv("under_10y_only_spread_moves.csv")
