from bisect import bisect_left
from collections import Counter
from datetime import timedelta
from functools import partial

import numpy as np
import pandas as pd

import lgimapy.vis as vis
from lgimapy.bloomberg import bdp
from lgimapy.data import Database
from lgimapy.utils import fill_dt_index, mkdir, root

vis.style()
# %%
# Load Data and Find risk entities.
def update_default_data():
    data = {}
    index_ratings = {"HY": ("BB+", "D")}
    db = Database()
    for index, ratings in index_ratings.items():
        saved_df, start_date = read_saved_data(index)
        db.load_market_data(start=start_date)
        ix = db.build_market_index(
            rating=ratings,
            in_hy_stats_index=True,
            amount_outstanding=(0.01, None),
        )
        new_df = get_default_data(ix, start_date)


def read_saved_data(name):
    path = root("data/HY/defaults")
    mkdir(path)
    fid = path / f"{name}.parquet"
    try:
        df = pd.read_parquet(fid)
    except (OSError, FileNotFoundError):
        df = pd.DataFrame()
        db = Database()
        last_observed_date = db.date("+1M", db.date("MARKET_START"))
    else:
        # Drop last 2 weeks of data in case new defaults occurred.
        df = df.iloc[:-10].copy()
        last_observed_date = df.index[-1]

    return df, last_observed_date


# %%
def get_default_data(ix, start_date):
    """
    Find number of daily defaults, MV of daily defaults,
    number of tickers in the index, and MV of the index.

    Returns
    -------
    pd.DataFrame:
        DataFrame of data after the last observed date.
    """

    dates = ix.dates
    isins = ix.isins

    # Scrape default dates for each isin.
    default_dates = pd.to_datetime(
        bdp(isins, "Corp", fields="DEFAULT_DATE").squeeze().dropna()
    ).sort_values()

    # Find the ticker and the amount outstanding on the last
    # day in the database for defaulting.
    last_observed_df = (
        ix.df.sort_values("Date", ascending=False)
        .groupby("ISIN", observed=True)
        .nth(0)
    )
    bond_default_df = pd.DataFrame(default_dates)
    for col in ["Ticker", "AmountOutstanding"]:
        bond_default_df[col] = last_observed_df[col]
    # Convert to the amount outstanding defaulted each day.
    daily_mv_defaults = (
        bond_default_df.groupby(["DEFAULT_DATE"]).sum().squeeze()
    )

    # Find the first day each ticker defaults.
    issuer_defaults = (
        bond_default_df.groupby("Ticker", observed=True)
        .nth(0)["DEFAULT_DATE"]
        .sort_values()
    )
    # Convert to the number of tickers defualting each day.
    daily_issuer_defaults = pd.Series(Counter(issuer_defaults))

    # Find total size in MV and number of tickers for the index.
    ix_mv = ix.df[["Date", "AmountOutstanding"]].groupby("Date").sum().squeeze()
    n_tickers = ix.df[["Date", "Ticker"]].groupby("Date").nunique()["Ticker"]

    # Combine together.
    df = pd.concat(
        (
            fill_dt_index(daily_issuer_defaults, 0).rename("n_defaults"),
            fill_dt_index(daily_mv_defaults, 0).rename("mv_defaults"),
            fill_dt_index(ix_mv, method="ffill").rename("mv_ix"),
            fill_dt_index(n_tickers, method="ffill").rename("n_ix_tickers"),
        ),
        axis=1,
    )
    # Do not include start date which is date of last observed data.
    return df[df.index > start_date].fillna(0)


df = get_default_data(ix, last_observed_date)
df
# %%


# Plot issuer default rate.
args = (dates, defaults, n_re)
ix_args = (dates, defaults_ix, n_re_ix)
default_rates = {
    "1 Year HY": find_issuer_default_rate(*args, 252),
    "1 Year Index Elgible HY": find_issuer_default_rate(*ix_args, 252),
    "3 Year HY": find_issuer_default_rate(*args, 756),
    "3 Year Index Elgible HY": find_issuer_default_rate(*ix_args, 756),
    "5 Year HY": find_issuer_default_rate(*args, 1260),
    "5 Year Index Elgible HY": find_issuer_default_rate(*ix_args, 1260),
}

default_rates = {k: v.rename(k) for k, v in default_rates.items()}

# %%
figsize = (16, 12)
vis.plot_multiple_timeseries(
    [default_rates[key] for key in sorted(default_rates.keys())[::-1]],
    ytickfmt="{x:.0%}",
    ylabel="Issuer Weighted Cumulative Default Rate",
    c_list=[
        "darkorchid",
        "darkorchid",
        "darkorange",
        "darkorange",
        "navy",
        "navy",
    ],
    ls_list=["-", ":"] * 3,
    figsize=figsize,
)
# vis.show()
vis.savefig("HY_issuer_default_rates")

# %%
# Plot market value default rate.
args = (dates, mv_df)
ix_args = (dates, mv_ix_df)
default_rates = {
    "1 Year HY": find_mv_default_rate(*args, 252),
    "1 Year Index Elgible HY": find_mv_default_rate(*ix_args, 252),
    "3 Year HY": find_mv_default_rate(*args, 756),
    "3 Year Index Elgible HY": find_mv_default_rate(*ix_args, 756),
    "5 Year HY": find_mv_default_rate(*args, 1260),
    "5 Year Index Elgible HY": find_mv_default_rate(*ix_args, 1260),
}

default_rates = {k: v.rename(k) for k, v in default_rates.items()}


# %%
vis.plot_multiple_timeseries(
    [default_rates[key] for key in sorted(default_rates.keys())[::-1]],
    ytickfmt="{x:.0%}",
    ylabel="Market Value Weighted Cumulative Default Rate",
    c_list=[
        "darkorchid",
        "darkorchid",
        "darkorange",
        "darkorange",
        "navy",
        "navy",
    ],
    ls_list=["-", ":"] * 3,
    figsize=figsize,
)
# vis.show()
vis.savefig("HY_market_value_default_rates")
