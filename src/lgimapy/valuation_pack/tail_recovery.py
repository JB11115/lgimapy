from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from lgimapy import vis
from lgimapy.data import Database, spread_diff
from lgimapy.latex import Document
from lgimapy.utils import EventDistance

vis.style()

# %%


def update_lc_tail(fid):
    doc = Document(
        fid, path="reports/valuation_pack", fig_dir=True, load_tex=True
    )
    crises_xsrets, crises_spreads = get_crises_tail_recovery_dfs()
    plot_xsret_crises_recoveries(crises_xsrets, doc.fig_dir)
    plot_spread_crises_recoveries(crises_spreads, doc.fig_dir)
    tail_df, curr_dt, prev_dt = get_tail_df()
    plot_just_tail(tail_df, curr_dt, prev_dt, doc.fig_dir)
    make_tail_position_table(tail_df, doc)
    doc.save_tex()


def get_tail_df():
    db = Database()

    current_date = db.date("today")
    prev_date = db.nearest_date("12/1/2019")
    kwargs = {"in_stats_index": True, "maturity": (10, None)}
    # kwargs = {"in_hy_stats_index": True, "rating": ("BB+", "BB-")}

    curr_dt = current_date.strftime("%m/%d/%Y")
    prev_dt = prev_date.strftime("%m/%d/%Y")

    # Get current spreads.
    db.load_market_data(current_date, local=True)
    ix = db.build_market_index(**kwargs)
    ix.calc_dollar_adjusted_spreads()
    curr_df = ix.ticker_df

    # Get previous spreads and combine.
    db.load_market_data(prev_date, local=True)
    ix = db.build_market_index(**kwargs)
    ix.calc_dollar_adjusted_spreads()
    prev_df = ix.ticker_df
    prev_df["Prev_PX_Adj_OAS"] = prev_df["PX_Adj_OAS"]
    df = pd.concat((curr_df, prev_df["Prev_PX_Adj_OAS"]), axis=1, join="inner")

    # Find change in spreads for each ticker and get cumulative market value.
    df["spread_chg"] = df["PX_Adj_OAS"] - df["Prev_PX_Adj_OAS"]
    df.sort_values("spread_chg", inplace=True)
    df["MV%"] = df["MarketValue"] / df["MarketValue"].sum()
    df["Cum MV"] = np.cumsum(df["MV%"])

    # sorted(df["Sector"].dropna().unique())
    cmap = {
        "AEROSPACE_DEFENSE": "goldenrod",
        "AIRLINES": "goldenrod",
        "APARTMENT_REITS": "firebrick",
        "DIVERSIFIED_MANUFACTURING": "goldenrod",
        "HEALTHCARE_REITS": "firebrick",
        "INDEPENDENT": "darkgreen",
        "INTEGRATED": "darkgreen",
        "MEDIA_ENTERTAINMENT": "darkorchid",
        "MIDSTREAM": "darkgreen",
        "OFFICE_REITS": "firebrick",
        "OIL_FIELD_SERVICES": "darkgreen",
        "OTHER_REITS": "firebrick",
        "REFINING": "darkgreen",
        "RETAIL_REITS": "firebrick",
        "SOVEREIGN": "steelblue",
    }
    df["color"] = df["Sector"].map(cmap)
    df["color"].fillna("grey", inplace=True)

    bad_tickers = {"AAL", "DALSCD"}
    tail_df = df[(df["Cum MV"] > 0.8) & ~df.index.isin(bad_tickers)].copy()
    return tail_df, curr_dt, prev_dt


def plot_just_tail(df_tail, curr_dt, prev_dt, path):
    x = np.concatenate([[0.8], df_tail["Cum MV"].values])
    bin_width = np.diff(x)
    df_tail["x"] = df_tail["Cum MV"] - bin_width / 2
    y = df_tail["spread_chg"]
    fig, ax = vis.subplots(figsize=(12, 8))
    ax.bar(
        x[:-1],
        y,
        width=bin_width,
        color=df_tail["color"],
        align="edge",
        alpha=0.5,
    )
    colors = {
        "Industrials": "goldenrod",
        "Energy": "darkgreen",
        "Sovs": "steelblue",
        "REITs": "firebrick",
        "Media": "darkorchid",
        "Other": "grey",
    }
    leg = [Line2D([0], [0], color=c, alpha=0.5, lw=7) for c in colors.values()]

    ax.legend(leg, colors.keys(), loc="upper left", fancybox=True, shadow=True)
    vis.format_xaxis(ax, xtickfmt="{x:.0%}")
    ax.set_ylabel(
        f"PX Adjusted Spread Change (bp)\nfrom {prev_dt} to {curr_dt}"
    )
    ax.set_xlabel(f"Market Value in Index as of {curr_dt}")
    ax.set_title("Long Credit 20% Tail")
    ax.yaxis.set_minor_locator(plt.MultipleLocator(50))
    ax.grid(True, which="minor", ls="--")
    df_tail_label = df_tail[df_tail["MarketValue"] > 5000]
    for ticker, row in df_tail_label.iterrows():
        spread_chg = row["spread_chg"]
        neg = spread_chg < 0
        ax.annotate(
            text=f"{ticker}  " if neg else f"  {ticker}",
            xy=(row["x"], spread_chg),
            rotation=90,
            va="top" if neg else "bottom",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    vis.savefig("lc_index_tail", path=path)
    vis.close()


def make_tail_position_table(df_tail, doc):
    db = Database()
    lc = db.load_portfolio(strategy="US Long Credit")

    ticker_df = lc.ticker_df
    lc_pos = lc.ticker_overweights().rename("overweight")

    df_tail_pos = pd.concat(
        (df_tail, ticker_df[["LGIMASector", "P_OAD"]], lc_pos),
        axis=1,
        join="inner",
    )

    table_cols = {
        "Issuer": "Issuer",
        "LGIMASector": "Sector",
        "MV%": "Market Value*% of Index",
        "spread_chg": "Spread Change*Since 2019",
        "PX_Adj_OAS": "PX Adj*Spread",
        "overweight": "LGIMA*Overweight",
    }
    df_table = (
        df_tail_pos[list(table_cols.keys())][(df_tail_pos["MV%"] > 0.001)]
        .sort_values("spread_chg", ascending=False)
        .rename(columns=table_cols)
        .rename_axis(None)
        .iloc[:20]
    )
    total = df_table.sum(numeric_only=True)
    cols = ["Spread Change*Since 2019", "PX Adj*Spread"]
    for col in cols:
        mv = "Market Value*% of Index"
        total.loc[col] = (df_table[col] * df_table[mv]).sum() / df_table[
            mv
        ].sum()
    table = df_table.append(pd.Series(total, name="Total"))
    footnote = (
        "\\scriptsize The \\textit{tail} for each crisis was constructed by "
        "finding issuers (comprising 10\\% of the index by market value) with "
        "the largest price-adjusted spread change from 3 months before each "
        "crisis wides to the wides. Price adjustment was 1 bp tighter per \\$ "
        "above par for `BBB's and 0.5 bp/\\$ for `A's."
    )
    doc.start_edit("lc_tail_table")
    doc.add_table(
        table,
        caption="Long Duration Positioning",
        table_notes=footnote,
        multi_row_header=True,
        adjust=True,
        col_fmt="lll|ccc|c",
        prec={
            "Market Value*% of Index": "2%",
            "Spread Change*Since 2019": "0f",
            "PX Adj*Spread": "0f",
            "LGIMA*Overweight": "2f",
        },
        midrule_locs=["Total"],
        gradient_cell_col="LGIMA*Overweight",
        gradient_cell_kws={"cmax": "army", "cmin": "rose"},
        alternating_colors=("lightgray", None),
    )
    doc.end_edit()


def get_crises_tail_recovery_dfs():
    db = Database()
    events = {
        "GFC": "12/04/2008",
        "Sov Debt Crisis": "10/04/2011",
        "Energy Crisis": "2/11/2016",
        "Covid-19": "3/23/2020",
    }
    d = defaultdict(list)
    for event, date in events.items():
        d["wides"].append(db.nearest_date(date))
        d["start"].append(db.date("3m", date))
        try:
            end = db.date("+2y", date)
        except IndexError:
            end = db.date("today")
        d["end"].append(end)

    event_df = pd.DataFrame(d, index=events.keys())
    tail_size = 10
    xsret_d, spread_d = defaultdict(list), defaultdict(list)
    for name, event in event_df.iterrows():
        db.load_market_data(start=event["start"], end=event["end"], local=True)

        # Find tail isins.
        ix = db.build_market_index(in_stats_index=True, maturity=(10, None))
        ix.calc_dollar_adjusted_spreads()
        start_df = ix.subset(date=event["start"]).df
        wides_df = ix.subset(date=event["wides"]).df
        start_df["start_oas"] = start_df["PX_Adj_OAS"]
        chg_df = pd.concat(
            (wides_df, start_df["start_oas"]), axis=1, join="inner"
        )
        chg_df["diff"] = chg_df["PX_Adj_OAS"] - chg_df["start_oas"]
        chg_df.sort_values("diff", inplace=True)
        chg_df["cum_mv"] = (
            np.cumsum(chg_df["MarketValue"]) / chg_df["MarketValue"].sum()
        )
        thresh = 1 - tail_size / 100
        tail = chg_df[chg_df["cum_mv"] > thresh]
        tail_isins = tail["ISIN"].values

        # Get index ex tail, tail of the index, and tail overall.
        # index = ix.subset(start=event["wides"])
        index = ix.subset(start=event["start"])
        index_ex_tail_ix = index.subset(isin=tail_isins, special_rules="~ISIN")
        tail_in_index_ix = index.subset(isin=tail_isins)
        tail_ix = db.build_market_index(isin=tail_isins, start=event["start"])

        # Find excess returns for each from the wides date.
        def centered_xsrets(ix):
            xsrets = ix.cumulative_excess_returns()
            xsrets.index = [
                (d - event["wides"]).days / 365 for d in xsrets.index
            ]
            return xsrets

        def centered_spreads(ix):
            spreads = ix.market_value_weight("PX_Adj_OAS")
            spreads.index = [
                (d - event["wides"]).days / 365 for d in spreads.index
            ]
            return spreads

        spread_d["in_index_tail"].append(
            centered_spreads(tail_in_index_ix).rename(name)
        )
        spread_d["index_ex_tail"].append(
            centered_spreads(index_ex_tail_ix).rename(name)
        )

        xsret_d["index_ex_tail"].append(
            centered_xsrets(index_ex_tail_ix).rename(name)
        )
        xsret_d["in_index_tail"].append(
            centered_xsrets(tail_in_index_ix).rename(name)
        )
        xsret_d["tail"].append(centered_xsrets(tail_ix).rename(name))

    return xsret_d, spread_d


def plot_xsret_crises_recoveries(xsret_dict, path):
    index_ex_tail_df = pd.concat(xsret_dict["index_ex_tail"], axis=1).fillna(
        method="bfill"
    )
    in_index_tail_df = pd.concat(xsret_dict["in_index_tail"], axis=1).fillna(
        method="bfill"
    )
    xsret_df_in_index = in_index_tail_df - index_ex_tail_df

    tail_df = pd.concat(xsret_dict["tail"], axis=1).fillna(method="bfill")
    in_index_diff = in_index_tail_df - index_ex_tail_df
    fallen_angel_diff = tail_df - index_ex_tail_df
    current_date = in_index_diff["Covid-19"].dropna().index[-1]
    current_in_ix_df = in_index_diff[in_index_diff.index <= current_date].iloc[
        -1
    ]
    current_fa_df = fallen_angel_diff[
        fallen_angel_diff.index <= current_date
    ].iloc[-1]

    fig, ax = vis.subplots(figsize=(14, 9.5))
    colors = ["k", "navy", "darkgreen", "darkorchid"]
    for col, color in zip(in_index_diff.columns, colors):
        ax.plot(
            in_index_diff[col],
            color=color,
            label=f"Selling Fallen Angels: {current_in_ix_df.loc[col]:.1%}",
            alpha=0.7,
            lw=2,
        )
        ax.plot(
            fallen_angel_diff[col],
            color=color,
            ls="--",
            label=f"Holding Fallen Angels: {current_fa_df.loc[col]:.1%}",
            alpha=0.7,
            lw=2,
        )
    vis.format_yaxis(ax, ytickfmt="{x:.0%}")
    ax.legend(
        fancybox=True,
        shadow=True,
        fontsize=16,
        title=f"Cumulative XSRets {current_date:.2f} yrs after Wides",
    )
    ax.set_ylabel("Cumulative Excess Returns\nfrom 3m Prior to Wides")
    ax.set_xlabel("Years from Crisis Wides")
    ax.set_title(
        "Long Credit Crisis XSRet Recovery\n"
        "Difference between Tail and Ex-Tail"
    )
    vis.savefig(f"xsret_crises_recovery", path=path)
    vis.close()


def plot_spread_crises_recoveries(spread_dict, path):
    index_ex_tail_df = pd.concat(spread_dict["index_ex_tail"], axis=1).fillna(
        method="bfill"
    )
    in_index_tail_df = pd.concat(spread_dict["in_index_tail"], axis=1).fillna(
        method="bfill"
    )
    in_index_diff = in_index_tail_df - index_ex_tail_df
    current_date = in_index_diff["Covid-19"].dropna().index[-1]
    current_df = in_index_diff[in_index_diff.index <= current_date].iloc[-1]
    fig, ax = vis.subplots(figsize=(14, 9.5))
    colors = ["k", "navy", "darkgreen", "darkorchid"]
    for col, color in zip(in_index_diff.columns, colors):
        ax.plot(
            in_index_diff[col],
            color=color,
            label=f"{col}: {current_df.loc[col]:.0f} bp",
            alpha=0.7,
            lw=2,
        )
    ax.legend(
        fancybox=True,
        shadow=True,
        fontsize=20,
        title=f"Spreads {current_date:.2f} yrs after Wides",
    )
    ax.set_ylabel("PX Adjusted OAS (bp)")
    ax.set_xlabel("Years from Crisis Wides")
    ax.set_title(
        "Long Credit Crisis Spread Recovery\n"
        "Difference between Tail and Ex-Tail"
    )
    vis.savefig(f"spread_crises_recovery", path=path)
    vis.close()
