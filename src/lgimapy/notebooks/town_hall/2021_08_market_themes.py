from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from lgimapy import vis
from lgimapy.data import Database
from lgimapy.latex import Document
from lgimapy.portfolios import PerformancePortfolio

vis.style()

# %%
account = "P-LD"
pdf_fid = "2021_10_market_themes_w_performance"
db = Database()
start_date = db.date("ytd")
end_date = db.date("today")
# end_date = db.date("month_end", "9/15/2021")


db.load_market_data(start=start_date, end=end_date)
ix = db.build_market_index(in_returns_index=True, maturity=(10, None))
start_port = db.load_portfolio(account=account, date=start_date)
start_port_ticker_df = start_port.ticker_df
start_ticker_bm_weight = start_port_ticker_df["BM_Weight"]
end_port = db.load_portfolio(account=account, date=end_date)
end_port_ticker_df = end_port.ticker_df
end_ticker_bm_weight = end_port_ticker_df["BM_Weight"]

ticker_bm_weight = pd.concat(
    (start_ticker_bm_weight, end_ticker_bm_weight), axis=1
).mean(axis=1)


# %%
raw_ticker_mv = (
    ix.df[["Ticker", "MarketValue"]]
    .groupby("Ticker", observed=True)
    .sum()
    .squeeze()
)
total_mv = raw_ticker_mv.sum()
ticker_mv = raw_ticker_mv / total_mv

sector_d = defaultdict(dict)
sectors = db.IG_sectors(drop_chevrons=True)
sectors.append("STATS_US_IG_10+")
for sector in tqdm(sectors):
    kwargs = db.index_kwargs(sector)
    name = kwargs["name"]
    sector_ix = ix.subset(**kwargs)
    sector_d["mv"][name] = sector_ix.df["MarketValue"].sum() / total_mv
    sector_d["xsret"][name] = sector_ix.aggregate_excess_returns()

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

sector_xsret = (
    pd.Series(sector_d["xsret"]).sort_values(ascending=False).dropna()
)
dropped = {"Energy", "US Long Credit Index"}
sector_xsret_dropped = sector_xsret[~sector_xsret.index.isin(dropped)]
n = 10
top_tickers = ticker_impact.iloc[:15]
raw_top_sectors = sector_xsret_dropped.iloc[:n].append(
    sector_xsret_dropped.iloc[-n:]
)
order = list(raw_top_sectors.index)
order.insert(n, "US Long Credit Index")
top_sectors = sector_xsret.loc[order]

# %%
# fig, ax = vis.subplots(figsize=(6, 8))
# n = 25
# top_impact = impact.iloc[:n].append(impact.iloc[-n:])
# top_impact.plot.barh(ax=ax, color="navy")
# ax.grid(False, axis="y")
# ax.set_yticklabels(top_impact.index, fontsize=10)
# vis.format_xaxis(ax, xtickfmt="{x:.2%}")
# ax.set_xlabel("Benchmark Returns Impact")
# vis.savefig("bm_returns_impact")

# %%
pp = PerformancePortfolio("P-LD", start=start_date, end=end_date)
sector_performance = pp.sectors()

# %%

ticker_df = pd.DataFrame(index=top_tickers.index)
ticker_df["Benchmark*Weight"] = pd.Series(ticker_d["bm_weight"]).loc[
    top_tickers.index
]
ticker_df["YTD*$\\Delta$OAS"] = pd.Series(ticker_d["oas"]).loc[
    top_tickers.index
]
ticker_df["Benchmark Return Impact"] = 100 * top_tickers
ticker_df["LGIMA Performance (bp)"] = pp.tickers(tickers=top_tickers.index)


sector_df = pd.DataFrame(index=top_sectors.index)
sector_df["YTD Excess Return"] = top_sectors
sector_performance["US Long Credit Index"] = 19.3
sector_df["LGIMA Performance (bp)"] = sector_performance.loc[top_sectors.index]

# %%
doc = Document(fid=pfd_fid, path="reports/town_hall")
doc.add_preamble(margin={"paperwidth": 11, "paperheight": 18})
doc.add_table(
    ticker_df,
    col_fmt="lcrrr",
    multi_row_header=True,
    adjust=True,
    prec={
        "Benchmark*Weight": "1%",
        "YTD*$\\Delta$OAS": "0f",
        "Benchmark Return Impact": "3f",
        "LGIMA Performance (bp)": "1f",
    },
    div_bar_col=["Benchmark Return Impact", "LGIMA Performance (bp)"],
    div_bar_kws={
        "Benchmark Return Impact": {"cmax": "navy"},
        "LGIMA Performance (bp)": {"cmax": "steelblue", "cmin": "firebrick"},
    },
)

doc.add_table(
    sector_df,
    col_fmt="lrr",
    adjust=True,
    multi_row_header=True,
    midrule_locs=[sector_df.index[n], sector_df.index[n + 1]],
    prec={
        "YTD Excess Return": "1%",
        "LGIMA Performance (bp)": "1f",
    },
    div_bar_col=["YTD Excess Return", "LGIMA Performance (bp)"],
    div_bar_kws={
        "cmax": "steelblue",
        "cmin": "firebrick",
    },
)
doc.save(save_tex=True)

# %%

ticker_df.to_csv(f"{account}_issuer_performance_insights.csv")
sector_df.to_csv(f"{account}_sector_performance_insights.csv")
