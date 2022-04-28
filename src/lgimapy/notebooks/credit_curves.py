from collections import defaultdict

import numpy as np
import pandas as pd

from lgimapy.data import Database

# %%

db = Database()
ust_df = db.load_bbg_data(
    ["UST_10Y", "UST_2Y"], "YTW", start="2000", end="2002"
)
ust_df["2s10s"] = ust_df["UST_10Y"] - ust_df["UST_2Y"]
peak_inversion = ust_df.iloc[ust_df["2s10s"].argmin()].name
ust_df[ust_df["2s10s"] < 0]

db.load_market_data(date=peak_inversion)
ix = db.build_market_index(in_stats_index=True, financial_flag=0)

# %%
mat_ix_d = {}
maturites = [5, 10, 30]
issuer_mv = defaultdict(int)
issuer_appearance_set_list = []

for maturity in maturites:
    mat_ix = ix.subset_on_the_runs(maturity)
    issuer_appearance_set_list.append(set(mat_ix.tickers))
    for ticker, mv in zip(mat_ix.df["Ticker"], mat_ix.df["MarketValue"]):
        issuer_mv[ticker] += mv
    mat_ix_d[maturity] = mat_ix

tickers = list(
    issuer_appearance_set_list[0].intersection(*issuer_appearance_set_list)
)
weights = (
    pd.Series(issuer_mv)
    .loc[tickers]
    .sort_values(ascending=False)
    .iloc[:15]
    .round(0)
)

# %%

spread_df = weights.to_frame()
yield_df = weights.to_frame()
spread_df.columns = ["TotalMarketValue"]
yield_df.columns = ["TotalMarketValue"]
for maturity in maturites:
    mat_ix = mat_ix_d[maturity]
    ticker_df = mat_ix.df[mat_ix.df["Ticker"].isin(weights.index)]
    spread_df[f"OAS_{maturity}y"] = ticker_df.set_index("Ticker")["OAS"]
    yield_df[f"YTW_{maturity}y"] = ticker_df.set_index("Ticker")["YieldToWorst"]

spread_df["5s10s"] = spread_df["OAS_10y"] - spread_df["OAS_5y"]
spread_df["10s30s"] = spread_df["OAS_30y"] - spread_df["OAS_10y"]
yield_df["5s10s"] = yield_df["YTW_10y"] - yield_df["YTW_5y"]
yield_df["10s30s"] = yield_df["YTW_30y"] - yield_df["YTW_10y"]


def add_total(df):
    total = pd.Series(dtype=float, index=df.columns, name="Total")
    for col in total.index:
        total[col] = (df["TotalMarketValue"] * df[col]).sum() / df[
            "TotalMarketValue"
        ].sum()
    total["TotalMarketValue"] = np.nan
    return pd.concat((df, total.to_frame().T))


spread_df = add_total(spread_df).round(0)
yield_df = add_total(yield_df).round(2)


yield_df.to_csv(f"credit_curve_YTW_{peak_inversion:%Y-%m-%d}.csv")
spread_df.to_csv(f"credit_curve_OAS_{peak_inversion:%Y-%m-%d}.csv")
