from collections import defaultdict
from datetime import timedelta
from itertools import chain

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import linregress
from tqdm import tqdm

import lgimapy.vis as vis
from lgimapy.data import Database
from lgimapy.latex import Document
from lgimapy.utils import root

vis.style()
# %matplotlib qt
# %%


db = Database()

for month in range(8, 9):
    pd.to_datetime(f"1/1/2019")
    while start_date < pd.to_datetime("1/28/2020"):
        try:
            port_df = db.load_portfolio(start_date, accounts="FLD")
        except ValueError:
            start_date += timedelta(1)
            continue
        else:
            break
    print(start_date)
    # get_major_positions(port_df, 2.5, [5])


holding_periods = (1, 2, 3, 4, 5, 6, 9, 12)
threshold = 2.5
holding_period = 1


# pos, cusips = get_major_positions(
#     port_df, threshold=2.5, holding_periods=holding_periods
# )


# %%
# %matplotlib qt
df = db.load_portfolio(accounts="FLD").set_index("CUSIP")
df = df[df["Sector"] != "TREASURIES"].copy()
replacement_tickers = {"TWC": "CHTR"}
df["Ticker"] = [replacement_tickers.get(t, t) for t in df["Ticker"]]
db.load_market_data(local=True)
ix = db.build_market_index(cusip=df.index)
df = pd.concat([df, ix.df["MarketValue"]], axis=1, sort=False)

ticker_df = (
    df[["Ticker", "OAD_Diff", "MarketValue"]]
    .groupby(["Ticker"], observed=True)
    .sum()
    .reset_index()
    .set_index("Ticker")
)

thresh_ticker_df = ticker_df[np.abs(ticker_df["OAD_Diff"]) > threshold]
# %%
med = np.median(ticker_df["OAD_Diff"])
fig, ax = vis.subplots()
vis.plot_hist(ax, ticker_df["OAD_Diff"], bins=60, normed=False)
ax.axvline(
    med, color="firebrick", ls="--", lw=1.5, label=f"Median: {med:.3f} yr"
)
ax.set_xlabel("OAD Difference (yr)")
ax.legend()
vis.show()
# %%


def weighted_median(a, weights):
    df = pd.DataFrame({"a": a, "weights": weights})
    df.sort_values("a", inplace=True)
    cumsum = np.cumsum(df["a"])
    cutoff = np.sum(df["a"]) / 2
    return df[cumsum >= cutoff]["a"].iloc[0]


np.sum(ticker_df.eval("OAD_Diff * MarketValue")) / np.sum(
    ticker_df["MarketValue"]
)


med = weighted_median(ticker_df["OAD_Diff"], ticker_df["MarketValue"]) / 1000
fig, ax = vis.subplots()
vis.plot_hist(
    ax,
    ticker_df["OAD_Diff"],
    bins=40,
    weights=ticker_df["MarketValue"] / 1000,
    normed=False,
)
ax.axvline(
    med, color="firebrick", ls="--", lw=1.5, label=f"Median: {med:.3f} yr"
)
vis.format_yaxis(ax, "${x:.0f}M")
ax.set_xlabel("OAD Difference (yr)")
ax.legend()
vis.show()

# %%
fig, ax = vis.subplots()
vis.plot_hist(ax, thresh_ticker_df.values, bins=20)
ax.set_xlabel("OAD Difference (bp)")
vis.show()

# %%
def get_major_positions(df, threshold, holding_periods):
    # Find position in tickers.
    df = df[df["Sector"] != "TREASURIES"].copy()
    replacement_tickers = {"TWC": "CHTR"}
    df["Ticker"] = [replacement_tickers.get(t, t) for t in df["Ticker"]]
    ticker_df = (
        df[["Ticker", "OAD_Diff"]]
        .groupby(["Ticker"], observed=True)
        .sum()
        .reset_index()
        .set_index("Ticker")
    )["OAD_Diff"] * 100
    pos = ticker_df[np.abs(ticker_df) > threshold].copy()

    # Find underlying bonds for each ticker.
    pos_tickers = set(pos.index)
    cusips_d = defaultdict(list)
    for _, bond in df.iterrows():
        ticker = bond["Ticker"]
        if ticker in pos_tickers:
            cusips_d[ticker].append(bond["CUSIP"])
    all_cusips = list(chain(*[cusips for cusips in cusips_d.values()]))

    # Load all data needed for perfomance calculations.
    db_2 = Database()
    start = df["Date"][0]
    end = db_2.date(f"+{max(holding_periods)}m", reference_date=start)
    db_2.load_market_data(start=start, end=end, cusips=all_cusips, local=True)
    performance_ix = db_2.build_market_index()

    # Track performance of underlying bonds for each holding period.
    df_list = [pos]

    for holding_period in holding_periods:
        # print(f"Holding Period: {holding_period}")
        end = db_2.date(f"+{holding_period}m", reference_date=start)
        ix = performance_ix.subset(start=start, end=end)
        # Track performance for bonds of each ticker.
        d = defaultdict(list)
        for ticker, cusips in tqdm(cusips_d.items()):
            ix_ticker = ix.subset(cusip=cusips)
            oas = ix_ticker.get_synthetic_differenced_plot_history("OAS")
            d["Ticker"].append(ticker)
            d["OAS_Change"].append(oas[-1] - oas[0])
        s_hp = pd.Series(
            d["OAS_Change"], index=d["Ticker"], name=f"{holding_period}m"
        )
        df_list.append(s_hp)

    df = pd.concat(df_list, axis=1, sort=True)
    df["start"] = start
    df.to_csv(f"ticker_ou_performance_{start.strftime('%Y%m%d')}.csv")


# %%


# %%
df = db.load_portfolio(accounts="P-LD", market_cols="all")


def get_issuer_df(df, non_fin=False):
    """Create issuer DataFrame."""
    if non_fin:
        bad_sectors = {
            "P_AND_C",
            "LIFE",
            "APARTMENT_REITS",
            "BANKING",
            "BROKERAGE_ASSETMANAGERS_EXCHANGES",
            "RETAIL_REITS",
            "HEALTHCARE_REITS",
            "OTHER_REITS",
            "FINANCIAL_OTHER",
            "FINANCE_COMPANIES",
            "OFFICE_REITS",
            "TREASURIES",
            "SOVEREIGN",
            "SUPRANATIONAL",
            "INDUSTRIAL_OTHER",
            "GOVERNMENT_GUARANTEE",
            "OWNED_NO_GUARANTEE",
        }
        bad_subsectors = {"Medical-HMO", "Medical-Hospitals"}
        bad_tickers = {"KPERM"}
        df = df[
            (~df["Sector"].isin(bad_sectors))
            & (~df["Subsector"].isin(bad_subsectors))
            & (~df["Ticker"].isin(bad_tickers))
        ].copy()

    replacement_tickers = {"TWC": "CHTR"}
    df["Ticker"] = [replacement_tickers.get(t, t) for t in df["Ticker"]]

    return (
        df[["Ticker", "OAD_Diff", "BM_OAD"]]
        .groupby(["Ticker"], observed=True)
        .sum()
        .reset_index()
        .set_index("Ticker")
    )


# %%


# %%
def make_current_position_table(account, non_fin):
    """Find table of current positioning."""
    port_df = db.load_portfolio(accounts=account, market_cols=True)
    ticker_df = get_issuer_df(port_df, non_fin=non_fin)

    position_threshold = 0.05
    thresholds_list = [(0.02, 0.05), (0.05, 0.10), (0.1, 0.15), (0.15, 1)]
    d = defaultdict(list)
    for thresholds in thresholds_list:
        thresh_df = ticker_df[
            (np.abs(ticker_df["BM_OAD"]) > thresholds[0])
            & (np.abs(ticker_df["BM_OAD"]) <= thresholds[1])
        ]
        count = defaultdict(int)
        for oad in thresh_df["OAD_Diff"]:
            if oad < -position_threshold:
                count["uw"] += 1
            elif oad < position_threshold:
                count["neutral"] += 1
            else:
                count["ow"] += 1
        d["Benchmark OAD"].append(thresholds)
        d[f"# in Benchmark"].append(len(thresh_df))
        d[f"# Overweight"].append(count["ow"])
        d[f"# Underweight"].append(count["uw"])
        d[f"# Neutral"].append(count["neutral"])

    return pd.DataFrame(d)


def make_ow_uw_plot(fid, day_dfs, non_fin):
    dates = list(year_port_df["Date"].unique())

    thresholds_list = [(0.02, 0.05), (0.05, 0.10), (0.1, 0.15), (0.15, 1)]
    df = day_dfs[0]
    thresholds = thresholds_list[0]
    d = defaultdict(list)
    for df in day_dfs:
        for thresholds in thresholds_list:
            ticker_df = get_issuer_df(df, non_fin=non_fin)
            thresh_df = ticker_df[
                (ticker_df["OAD_Diff"] > thresholds[0])
                & (ticker_df["OAD_Diff"] <= thresholds[1])
            ]
            d[f"ow_{thresholds}"].append(len(thresh_df))
            thresh_df = ticker_df[
                (ticker_df["OAD_Diff"] < -thresholds[0])
                & (ticker_df["OAD_Diff"] >= -thresholds[1])
            ]
            d[f"uw_{thresholds}"].append(len(thresh_df))

    position_df = pd.DataFrame(d, index=dates)
    position_df = position_df[position_df.index != "7/9/2019"]
    overs = [col for col in position_df.columns if "ow" in col]

    fig, axes = vis.subplots(1, 2, figsize=(14, 5), sharey=True)
    overs = [col for col in position_df.columns if "ow" in col]
    ow_df = position_df[overs]
    cols = [col[3:] for col in ow_df.columns]
    cols[-1] = "> 0.15"
    ow_df.columns = cols
    vis.plot_multiple_timeseries(
        ow_df,
        title="Overweights",
        ylabel="# of Issuers",
        ax=axes[0],
        legend=False,
    )

    unders = [col for col in position_df.columns if "uw" in col]
    uw_df = position_df[unders]
    cols = [col[3:] for col in uw_df.columns]
    cols[-1] = "> 0.15"
    uw_df.columns = cols
    vis.plot_multiple_timeseries(
        uw_df, title="Underweights", ax=axes[1], legend=False
    )
    axes[0].legend(loc="upper left", shadow=True, fancybox=True)
    vis.savefig(fid)
    vis.close()


# %%
def main():
    # %%
    account = "JOYLA"
    path = root("latex/PM_Meetings/2020-01/")

    # Load data.
    year_port_df = db.load_portfolio(
        accounts=account, start="1/1/2019", market_cols=True
    )
    day_dfs = [df for date, df in year_port_df.groupby("Date")]
    doc = Document(account, path=path)

    # Add account section.
    doc.add_preamble(margin={"top": 1, "left": 1, "right": 1})
    doc.add_section(account)
    fid = f"{account}_ou_uw"
    make_ow_uw_plot(path / fid, day_dfs, non_fin=False)
    doc.add_figure(fid, width=0.95)
    table = make_current_position_table(account, non_fin=False)
    doc.add_table(table, hide_index=True)

    # Add Non-fin section.
    doc.add_section("Non-Fin")
    fid = f"{account}_nonfin_ou_uw"
    make_ow_uw_plot(path / fid, day_dfs, non_fin=True)
    doc.add_figure(fid, width=0.95)
    table = make_current_position_table(account, non_fin=True)
    doc.add_table(table, hide_index=True)

    doc.save()
    # %%


def largest_30_market_values(account):
    # %%
    db = Database()
    df = db.load_portfolio(accounts=account, market_cols=True)
    replacement_tickers = {"TWC": "CHTR"}
    df["Ticker"] = [replacement_tickers.get(t, t) for t in df["Ticker"]]

    ticker_df = (
        df.groupby(["Ticker"], observed=True)
        .sum()
        .reset_index()
        .set_index("Ticker")
        .sort_values("BM_Weight", ascending=False)
        .reset_index()
    )

    cols = ["Ticker", "P_Weight", "BM_Weight", "P_OAD", "BM_OAD", "OAD_Diff"]
    table = ticker_df[cols][:30]
    cols = [
        "{}",
        "Portfolio MV",
        "Benchmark MV",
        "Portfolio OAD",
        "Benchmark OAD",
        "OAD Overweight",
    ]
    table.columns = cols
    # %%
    path = root("latex/PM_Meetings/2020-01/")
    doc = Document(fid=f"{account}_top_30", path=path)
    doc.add_section(f"{account} Top 30 Issuers")
    doc.add_table(
        table,
        col_fmt="lrccccr",
        hide_index=True,
        prec={
            "Portfolio MV": "1%",
            "Benchmark MV": "1%",
            "Portfolio OAD": "2f",
            "Benchmark OAD": "2f",
            "OAD Overweight": "2f",
        },
        midrule_locs=np.arange(5, len(table), 5),
        gradient_cell_col="OAD Overweight",
        gradient_cell_kws={"cmin": "firebrick", "cmax": "steelblue"},
        adjust=True,
    )
    doc.save()
