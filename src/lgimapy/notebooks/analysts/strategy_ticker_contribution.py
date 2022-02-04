from collections import defaultdict
from datetime import datetime as dt

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import lgimapy.vis as vis
from lgimapy.bloomberg import bdh
from lgimapy.data import Database
from lgimapy.utils import load_json, first_unique, mkdir, root


# %%


def main():
    # %%
    fid = f"{dt.today().strftime('%Y-%m-%d')}_TMT"
    fig_dir = root("reports/analyst_sectors")
    mkdir(fig_dir)
    use_all_strategies = False
    sector_list = [
        "WIRELESS",
        "WIRELINES",
        "CABLE_SATELLITE",
        "MEDIA_ENTERTAINMENT",
        "TECHNOLOGY",
    ]
    threshold = None  # default is `None`
    sectors = sector_list
    all_strategies = use_all_strategies
    df = get_strategy_positions(sector_list, use_all_strategies, threshold)

    # df.to_csv(f"{fig_dir / fid}.csv")
    # %%
    vis.style()
    plot_overweight_heatmap(df, figsize=(12, 11))
    # vis.show()
    vis.savefig(fig_dir / fid, dpi=500)
    # %%


# %%
def plot_strategy_ticker_overweights(
    sector_list,
    fid,
    use_all_strategies=False,
    threshold=None,
):
    fig_dir = root("reports/analyst_sectors")
    mkdir(fig_dir)

    df = get_strategy_positions(sector_list, use_all_strategies, threshold)

    plot_overweight_heatmap(df, figsize=(12, 6))
    vis.savefig(fig_dir / fid)


def get_strategy_positions(
    sectors, all_strategies=False, threshold=None, index="Ticker"
):
    """
    Get strategy positions for given sectors.
    """
    if all_strategies:
        strategies = [
            "Liability Aware Long Duration Credit",
            "Barclays-Russell LDI Custom - DE",
            "US Long A+ Credit",
            "US Credit A or better",
            "INKA",
            "80% US A or Better LC/20% US BBB LC",
            "US Long Corporate A or better",
            "US Intermediate Credit",
            "US Intermediate Credit A or better",
            "Global Agg USD Corp",
            "US Corporate IG",
            "US Corporate 1% Issuer Cap",
            "US Credit",
            "Custom RBS",
            "GM_Blend",
            "US Long Corporate",
            "US Long Corp 2% Cap",
            "US Long Credit Ex Emerging Market",
            "US Corporate 7+ Years",
            "US Long Credit",
            "US Long Government/Credit",
            "US Long GC 70/30",
            "US Long GC 75/25",
            "US Long GC 80/20",
        ]
    else:
        strategies = [
            "Liability Aware Long Duration Credit",
            "US Long A+ Credit",
            "US Credit A or better",
            "80% US A or Better LC/20% US BBB LC",
            "US Long Corporate A or better",
            "US Intermediate Credit A or better",
            "US Credit",
            "US Long Corporate",
            "US Long Corp 2% Cap",
            "US Long Credit Ex Emerging Market",
            "US Corporate 7+ Years",
            "US Long Credit",
            "US Long Government/Credit",
            "US Long GC 70/30",
            "US Long GC 75/25",
            "US Long GC 80/20",
        ]
    # Get OAD Contribution of each ticker by strategy.
    strategy_accounts = load_json("strategy_accounts")
    db = Database()
    db.display_all_columns()
    df_list = []
    index = "Ticker"
    strategy = "US Corporate 7+ Years"
    for strategy in tqdm(strategies):
        strat = db.load_portfolio(strategy=strategy, universe="stats")
        strat_sectors = strat.subset(sector=sectors)
        df_list.append(strat_sectors.ticker_overweights().rename(strategy))
    df = pd.concat(df_list, join="outer", axis=1, sort=False)

    # Get map from ticker to sector.
    db.load_market_data(local=True)
    ix = db.build_market_index(sector=sectors)
    sectors_df = ix.df.groupby(index, observed=True).nth(0)
    sector_dict = {
        ticker: sector for ticker, sector in sectors_df["Sector"].items()
    }
    # Drop Tickers that aren't in database
    while True:
        try:
            df["Sector"] = [sector_dict[ticker] for ticker in df.index]
        except KeyError as e:
            ticker = e.args[0]
            print(f"Warnging, {ticker} is not in Database.")
            df.drop(ticker, inplace=True)
        else:
            break

    # Sort DataFrame by sector and strategy.
    sorter = dict(zip(sectors, range(len(sectors))))
    df["sector_sort"] = df["Sector"].map(sorter)
    df.sort_values(
        [
            "sector_sort",
            "US Long Credit",
            "US Credit",
            "US Intermediate Credit A or better",
        ],
        ascending=[True, False, False, False],
        inplace=True,
    )

    # Make sector names bold in index.
    _, sector_names = first_unique(df["Sector"])
    new_ix = []
    for ticker, sector in zip(df.index, sector_names):
        if sector == "":
            new_ix.append(ticker)
        else:
            bold_sector = ""
            for word in sector.replace("_", " ").title().split():
                bold_sector += f"$\\bf{{{word}}}$ "
            new_ix.append(bold_sector + ticker)

    df.drop(["Sector", "sector_sort"], axis=1, inplace=True)
    df.index = new_ix

    if threshold is not None:
        if not isinstance(threshold, (tuple, list)):
            threshold = (-threshold, threshold)
        met_thresh = np.sum((df <= threshold[0]) | (df >= threshold[1]), axis=1)
        keep_rows = [True if met > 0 else False for met in met_thresh]
        df = df[keep_rows]
    return df.dropna(how="all")


# %%
def plot_overweight_heatmap(df, figsize=(16, 9)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    df = df[df.abs().max(axis=1) >= 0.005]
    sns.heatmap(
        df,
        cmap="coolwarm_r",
        center=0,
        linewidths=0.2,
        annot=True,
        annot_kws={"fontsize": 7},
        fmt=".2f",
        cbar=True,
        cbar_kws={"label": "Strategy Overweight (OAD)"},
        ax=ax,
    )
    ax.xaxis.tick_top()
    ax.set_xticklabels(df.columns, rotation=45, ha="left", fontsize=8)
    ax.set_yticklabels(df.index, ha="right", fontsize=7, va="center")

    # Mask labels below threshold over +/- 0.01 OAD contribution.
    threshold = 0.01
    annot_mask = []
    for row in range(len(df)):
        row_annot = []
        for val in df.iloc[row, :]:
            if val is None or np.isnan(val):
                continue
            if np.abs(val) >= threshold:
                row_annot.append(True)
            else:
                row_annot.append(False)
        annot_mask.append(row_annot)

    for text, show_annot in zip(
        ax.texts, (element for row in annot_mask for element in row)
    ):
        text.set_visible(show_annot)


# plot_overweight_heatmap(df, figsize=(12, 5))
# vis.show()
# %%
if __name__ == "__main__":
    main()
