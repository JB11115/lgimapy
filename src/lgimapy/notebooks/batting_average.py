from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from lgimapy import vis
from lgimapy.data import Database
from lgimapy.latex import Document
from lgimapy.portfolios import PerformancePortfolio
from lgimapy.utils import load_json

vis.style()

# %%
db = Database()

account = "P-LD"
bm_index = "STATS_US_IG_10+"
start_date = db.date("ytd", "1/5/2020")
end_date = db.date("ytd", "1/5/2021")
n_tickers = 15
n_sectors = 10
ignored_sectors = {"ENERGY"}

fid = f"{account}_batting_average_{start_date:%Y-%m-%d}_to_{end_date:%Y-%m-%d}"
fid
# %%

bm_index_name = db.index_kwargs(bm_index)["name"]

# Get ticker stats
db.load_market_data(start=start_date, end=end_date)
ix = db.build_market_index(maturity=(10, None), rating=(None, "BBB-"))
start_port = db.load_portfolio(account=account, date=start_date)
start_port_ticker_df = start_port.ticker_df
start_ticker_bm_weight = start_port_ticker_df["BM_Weight"]
end_port = db.load_portfolio(account=account, date=end_date)
end_port_ticker_df = end_port.ticker_df
end_ticker_bm_weight = end_port_ticker_df["BM_Weight"]

ticker_bm_weight = pd.concat(
    (start_ticker_bm_weight, end_ticker_bm_weight), axis=1
).mean(axis=1)


raw_ticker_mv = (
    ix.df[["Ticker", "MarketValue"]]
    .groupby("Ticker", observed=True)
    .sum()
    .squeeze()
)
total_mv = raw_ticker_mv.sum()
ticker_mv = raw_ticker_mv / total_mv


ticker_d = defaultdict(dict)
for ticker in tqdm(ticker_mv.index):
    ticker_ix = ix.subset(ticker=ticker)
    ticker_d["xsret"][ticker] = ticker_ix.aggregate_excess_returns()
    oas = ticker_ix.OAS()
    ticker_d["oas"][ticker] = oas.iloc[-1] - oas.iloc[0]
    try:
        ticker_d["bm_weight"][ticker] = ticker_bm_weight.loc[ticker]
    except KeyError:
        ticker_d["bm_weight"][ticker] = 0

ticker_impact = (
    (pd.Series(ticker_d["xsret"]) * ticker_mv)
    .abs()
    .sort_values(ascending=False)
    .dropna()
)
tickers = ticker_impact.index


sector_d = defaultdict(dict)
sectors = [
    sector
    for sector in db.IG_sectors(drop_chevrons=True)
    if sector not in ignored_sectors
]
sectors.append(bm_index)
for sector in tqdm(sectors):
    kwargs = db.index_kwargs(sector)
    name = kwargs["name"]
    sector_ix = ix.subset(**kwargs)
    sector_d["mv"][name] = sector_ix.df["MarketValue"].sum() / total_mv
    sector_d["xsret"][name] = sector_ix.aggregate_excess_returns()


sector_xsret = (
    pd.Series(sector_d["xsret"]).sort_values(ascending=False).dropna()
)


p_weight_dates = db.date("MONTH_STARTS", start=start_date)[1:]
p_weight_s_list = []
for date in tqdm(p_weight_dates):
    port = db.load_portfolio(account=account, date=date)
    p_weight_s_list.append(port.ticker_df["P_Weight"])

p_weights = pd.concat(p_weight_s_list, axis=1).fillna(0).sum(axis=1)
p_weights /= p_weights.sum()

# %%
pp = PerformancePortfolio("P-LD", start=start_date, end=end_date, pbar=True)
sector_performance = pp.sectors()
strat_performance = pp.total()
# %%
ticker_df = pd.DataFrame(index=tickers)
ticker_df["Benchmark*Weight"] = pd.Series(ticker_d["bm_weight"]).loc[tickers]
ticker_df["$\\Delta$OAS"] = pd.Series(ticker_d["oas"]).loc[tickers]
ticker_df["Benchmark Return Impact"] = 100 * ticker_impact
ticker_df["LGIMA Performance (bp)"] = pp.tickers(tickers=tickers)
ticker_df["Sector"] = (
    pd.concat(
        (
            start_port_ticker_df["LGIMASector"],
            end_port_ticker_df["LGIMASector"],
        ),
        axis=1,
    )
    .fillna(method="ffill", axis=1)
    .iloc[:, -1]
)
bm_ticker_df = ticker_df[ticker_df["Benchmark*Weight"] > 0].copy()
bm_ticker_df["Benchmark Return Impact"] /= bm_ticker_df[
    "Benchmark Return Impact"
].sum()
ticker_table = bm_ticker_df.iloc[:n_tickers, :-1]


sector_df = pd.DataFrame()
sector_df["Excess Return"] = sector_xsret
sector_df["LGIMA Performance (bp)"] = sector_performance
sector_df.loc[bm_index_name, "LGIMA Performance (bp)"] = strat_performance
sector_df_ex_strat = sector_df[sector_df.index != bm_index_name]
sector_table = pd.concat(
    (
        sector_df_ex_strat.iloc[:n_sectors],
        sector_df.loc[bm_index_name].to_frame().T,
        sector_df_ex_strat.iloc[-n_sectors:],
    )
)
# %%
doc = Document(fid=fid, path="reports/batting_average")
doc.add_preamble(
    margin={
        "left": 1,
        "right": 1,
        "top": 1.5,
        "bottom": 1,
    },
    header=doc.header(
        left=f"Batting Averages in {account}",
        right=f"{start_date:%b %#d, %Y} - {end_date:%b %#d, %Y}",
        height=0.5,
    ),
    table_caption_justification="c",
)
doc.add_table(
    ticker_table,
    col_fmt="lcrrr",
    caption="Ticker Batting Averages",
    multi_row_header=True,
    prec={
        "Benchmark*Weight": "1%",
        "$\\Delta$OAS": "0f",
        "Benchmark Return Impact": "1%",
        "LGIMA Performance (bp)": "1f",
    },
    div_bar_col=["Benchmark Return Impact", "LGIMA Performance (bp)"],
    div_bar_kws={
        "Benchmark Return Impact": {"cmax": "navy"},
        "LGIMA Performance (bp)": {"cmax": "steelblue", "cmin": "firebrick"},
    },
)
doc.add_table(
    sector_table,
    col_fmt="lrr",
    caption="Sector Batting Averages",
    multi_row_header=True,
    midrule_locs=[
        sector_table.index[n_sectors],
        sector_table.index[n_sectors + 1],
    ],
    prec={
        "Excess Return": "1%",
        "LGIMA Performance (bp)": "1f",
    },
    div_bar_col=["Excess Return", "LGIMA Performance (bp)"],
    div_bar_kws={
        "cmax": "steelblue",
        "cmin": "firebrick",
    },
)
doc.save()
# ticker_df.to_csv(f"{fid}_issuers.csv")
# sector_df.to_csv(f"{fid}_sectors.csv")
