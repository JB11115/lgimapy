from collections import defaultdict

import numpy as np
import pandas as pd

from lgimapy import vis
from lgimapy.data import Database, Index, groupby
from lgimapy.utils import root, Time

vis.style()
# %%

# %%
def main():
    df = load_data()
    date = df.index[-1]
    print(date)


def load_data():
    """Load data, updated and save if required."""
    fid = root("data/IG_correlation.parquet")
    try:
        old_df = pd.read_parquet(fid)
    except (FileNotFoundError, OSError):
        # Create file from scratch.
        df = compute_correlation(start="1/1/2000", end="3/1/2000")
    else:
        # Update data if required.
        last_date = old_df.index[-1]
        dates_to_compute = Database().trade_dates(exclusive_start=last_date)
        if dates_to_compute:
            new_df = compute_correlation(start=dates_to_compute[0])
            df = pd.concat((old_df, new_df))
        else:
            df = old_df.copy()

    # Save Data.
    df.to_parquet(fid)
    return df


def compute_correlation(start, end=None):
    lookback = "3m"
    db = Database()
    db.load_market_data(start=db.date(lookback, start), end=end)
    full_ix = db.build_market_index(in_stats_index=True, maturity=(10, None))
    ix_d = {
        "IG": full_ix.copy(),
        "A": full_ix.subset(rating=("A+", "A-")),
        "BBB": full_ix.subset(rating=("BBB+", "BBB-")),
    }
    dates_to_compute = db.trade_dates(start=start, end=end)
    df_list = []
    for name, ix in ix_d.items():
        corr = compute_single_index_corr(ix, lookback, dates_to_compute, db)
        df_list.append(corr.rename(name))
    return pd.concat(df_list, axis=1)


def compute_single_index_corr(ix, lookback, dates, db):
    cols = [
        "Ticker",
        "Issuer",
        "CollateralType",
        "Date",
        "XSRet",
        "PrevMarketValue",
    ]
    re_df = groupby(ix.df[cols].reset_index(), cols=cols[:4]).reset_index()
    re_df["RiskEntity"] = (
        re_df["Ticker"].astype(str)
        + "|"
        + re_df["Issuer"].astype(str)
        + "|"
        + re_df["CollateralType"].astype(str)
    )
    re_df["CUSIP"] = re_df["RiskEntity"]
    re_ix = Index(re_df, index="RiskEntity")
    xsret_df = re_ix.get_value_history("XSRet")

    corr = np.zeros(len(dates))
    for i, date in enumerate(dates):
        start = db.date(lookback, date)
        df_date = xsret_df[(xsret_df.index >= start) & (xsret_df.index <= date)]
        corr[i] = avg_corr(df_date)

    return pd.Series(corr, index=dates)


def avg_corr(df):
    corr_matrix = df.dropna(axis=1).corr()
    # Remove nans if any are present in the matrix.
    while corr_matrix.isna().sum().sum():
        # Remove them one risk entity at a time.
        bad_re = corr_matrix.isna().sum().sort_values().index[-1]
        corr_matrix = corr_matrix.drop(bad_re).drop(columns=bad_re)

    # Find average of lower triangular portion of matrix
    # less the 1's diagonal.
    n = corr_matrix.shape[0]
    return (np.tril(corr_matrix).sum() - n) / np.arange(n).sum()


def plot_correlation(df):
    # %%
    fid = root("data/IG_correlation.parquet")
    df = pd.read_parquet(fid)
    # %%
    colors = ["k", "#875053", "#80A1C1"]
    fig, ax = vis.subplots(figsize=(14, 6))
    for col, color in zip(df.columns, colors):
        corr = df[col]
        med = np.median(corr)
        label = f"{col}, Median: {med:.2f}"
        vis.plot_timeseries(
            df[col],
            color=color,
            lw=1.5,
            alpha=0.9,
            median_line=True,
            median_line_kws={"color": color, "label": "_nolegend_"},
            label=label,
            ax=ax,
        )
        ax.legend(fancybox=True, shadow=True, fontsize=16)
    vis.savefig("IG_correlation_by_rating")
    # vis.show()
    # %%
    fig, ax = vis.subplots(figsize=(14, 6))
    vis.plot_timeseries(
        df["IG"],
        lw=2,
        color="navy",
        median_line=True,
        median_line_kws={"prec": 2},
        ax=ax,
    )
    ax.legend(fancybox=True, shadow=True, fontsize=16)
    vis.savefig("IG_correlation")
    # vis.show()

    # %%

    # vis.savefig("IG_correlation")


# %%
if __name__ == "__main__":
    main()
