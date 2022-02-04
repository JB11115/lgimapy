from collections import defaultdict

import numpy as np
import pandas as pd

from lgimapy.data import Database, groupby

# %%
account = "P-LD"

db = Database()
port = db.load_portfolio(account=account)

# %%
col_map = {
    "DTS_Diff": "DTS OW",
    "OAD_Diff": "OAD OW",
    "IssueYears": "Years Since Issuance",
    "P_Weight": "Port Weight (%)",
    "BM_Weight": "BM Weight (%)",
}


otr_30y_cusips = (
    port.subset(maturity=(27, 32))
    .df[["Ticker", "OAS", "IssueDate", "CUSIP"]]
    .set_index("CUSIP")
    .groupby("Ticker", observed=True)
    .idxmax()["IssueDate"]
    .dropna()
)
otr_30y_oas = (
    port.subset(cusip=otr_30y_cusips.values)
    .df.set_index("Ticker")["OAS"]
    .to_dict()
)

energy_cusips = port.subset(**db.index_kwargs("ENERGY")).df["CUSIP"]
sifi_cusips = port.subset(**db.index_kwargs("SIFI_BANKS")).df["CUSIP"]
bad_cusips = set(energy_cusips) | set(sifi_cusips)

plus_ones_df = port.df[port.df["AnalystRating"] > 0]
plus_ones_df = plus_ones_df[~plus_ones_df["CUSIP"].isin(bad_cusips)]
plus_ones_ticker_df = groupby(plus_ones_df, "Ticker").rename_axis(None)

hospitals_df = port.subset(**db.index_kwargs("HOSPITALS")).df
hospitals_ticker_df = groupby(hospitals_df, "Ticker").rename_axis(None)
hospitals_sector_df = groupby(hospitals_df, "Sector").rename_axis(None)
hospitals_sector_df.index = ["Hospitals"]
hospitals_df = hospitals_df[hospitals_df["P_AssetValue"] > 0]

munis_df = port.subset(**db.index_kwargs("MUNIS")).df
munis_ticker_df = groupby(munis_df, "Ticker").rename_axis(None)
munis_sector_df = groupby(munis_df, "Sector").rename_axis(None)
munis_sector_df.index = ["Munis"]
munis_df = munis_df[munis_df["P_AssetValue"] > 0]

universities_df = port.subset(**db.index_kwargs("UNIVERSITY")).df
universities_ticker_df = groupby(universities_df, "Ticker").rename_axis(None)
universities_sector_df = groupby(universities_df, "Sector").rename_axis(None)
universities_sector_df.index = ["Universities"]
universities_df = universities_df[universities_df["P_AssetValue"] > 0]

sovs_df = port.subset(**db.index_kwargs("SOVEREIGN")).df
sovs_ticker_df = groupby(sovs_df, "Ticker").rename_axis(None)
sovs_sector_df = groupby(sovs_df, "Sector").rename_axis(None)
sovs_sector_df.index = ["Sovs"]
sovs_df = port.subset(ticker=["INDON", "PHILIP", "ISRAEL"]).df

# %%

raw_ticker_df = pd.concat(
    (
        plus_ones_ticker_df,
        hospitals_sector_df,
        munis_sector_df,
        universities_sector_df,
        sovs_sector_df,
    )
).sort_values("DTS_Diff", ascending=False)
# raw_ticker_df = pd.concat(
#     (
#         plus_ones_ticker_df,
#         hospitals_ticker_df,
#         munis_ticker_df,
#         universities_ticker_df,
#         sovs_ticker_df,
#     )
# ).sort_values("DTS_Diff", ascending=False)
ticker_cols = [
    "LGIMASector",
    "DTS_Diff",
    "OAD_Diff",
    "P_Weight",
    "BM_Weight",
    "AmountOutstanding",
    "MarketValue",
    "AnalystRating",
]

ticker_df = raw_ticker_df[ticker_cols].copy()
ticker_df["Rating"] = db.convert_numeric_ratings(
    raw_ticker_df["NumericRating"].round()
)
ticker_df["OAS"] = raw_ticker_df["OAS"]
ticker_df["OTR 30y OAS"] = ticker_df.index.map(otr_30y_oas)
ticker_df
for col in ["P_Weight", "BM_Weight"]:
    ticker_df[col] *= 100
ticker_df.columns = [col_map.get(col, col) for col in ticker_df.columns]

non_corp_tickers = [
    "INDON",
    "ISRAEL",
    "PHILIP",
    "GTOWNU",
    "USCTRJ",
    "PORTRN",
    "UNVHGR",
    "KPERM",
    "NYPRES",
]
non_corp_df = ticker_df.reindex(non_corp_tickers)
# non_corp_df.to_csv('Potential_Buys_Tickers_non_corp.csv')
ticker_df.to_csv("Potential_Buys_Tickers.csv")


# %%
raw_bond_df = pd.concat(
    (
        plus_ones_df,
        hospitals_df,
        munis_df,
        universities_df,
        sovs_df,
    )
)
bond_cols = [
    "Ticker",
    "CouponRate",
    "MaturityDate",
    "CUSIP",
    "LGIMASector",
    "DTS_Diff",
    "OAD_Diff",
    "P_Weight",
    "BM_Weight",
    "AmountOutstanding",
    "MarketValue",
    "OAS",
    "OAD",
    "IssueYears",
    "AnalystRating",
]
bond_df = raw_bond_df[bond_cols].copy()
bond_df["Rating"] = db.convert_numeric_ratings(
    raw_bond_df["NumericRating"].round()
)
for col in ["P_Weight", "BM_Weight"]:
    bond_df[col] *= 100
bond_df.columns = [col_map.get(col, col) for col in bond_df.columns]
bond_df["sort"] = bond_df["Ticker"].astype(str)
for sector in ["Sovereigns", "Munis", "Universities", "Hospitals"]:
    bond_df.loc[bond_df["LGIMASector"] == sector, "sort"] = sector


# Sort bonds by ticker OW, and then bond OW within ticker.
def sorter(col):
    cat = pd.Categorical(col, categories=ticker_df.index, ordered=True)
    return pd.Series(cat)


bond_df.sort_values("sort", key=sorter, inplace=True)
unique_tickers = set()
sorted_tickers = []
for ticker in bond_df["Ticker"]:
    if ticker in unique_tickers:
        continue
    else:
        unique_tickers.add(ticker)
        sorted_tickers.append(ticker)

gdfs = {ticker: df for ticker, df in bond_df.groupby("Ticker")}
bond_df_list = []
for ticker in sorted_tickers:
    bond_df_list.append(gdfs[ticker].sort_values("DTS OW", ascending=False))


final_bond_df = (
    pd.concat(bond_df_list)
    .set_index("Ticker")
    .drop("sort", axis=1)
    .rename_axis(None)
)
final_bond_df.to_csv("Potential_Buys_Bonds.csv")
