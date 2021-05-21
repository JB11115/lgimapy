import pandas as pd

from lgimapy.data import Database
from lgimapy.utils import load_json, dump_json

# %%
def update_top_30_tickers(date):
    """
    Update the top 30 tickers used for the
    BBB Top 30 Non-Fin indexes.
    """
    db = Database()
    db.load_market_data(local=True, date=date)
    fid = "index_kwargs/bloomberg"
    indexes = load_json(fid)

    # Update tickers for long credit.
    ix = db.build_market_index(
        rating=("BBB+", "BBB-"),
        maturity=(10, None),
        financial_flag=False,
        sector="LOCAL_AUTHORITIES",
        in_stats_index=True,
        special_rules="~Sector",
    )
    df = (
        ix.df.groupby(["Ticker"], observed=True)
        .sum()
        .sort_values("MarketValue", ascending=False)
    )
    top_30_tickers = list(df.index[:31])
    indexes["BBB_NON_FIN_TOP_30_10+"]["ticker"] = top_30_tickers
    indexes["BBB_NON_FIN_EX_TOP_30_10+"]["ticker"] = top_30_tickers

    # Update tickers for market credit.
    ix = db.build_market_index(
        rating=("BBB+", "BBB-"),
        financial_flag=False,
        sector="LOCAL_AUTHORITIES",
        in_stats_index=True,
        special_rules="~Sector",
    )
    df = (
        ix.df.groupby(["Ticker"], observed=True)
        .sum()
        .sort_values("MarketValue", ascending=False)
    )
    top_30_tickers = list(df.index[:31])
    indexes["BBB_NON_FIN_TOP_30"]["ticker"] = top_30_tickers
    indexes["BBB_NON_FIN_EX_TOP_30"]["ticker"] = top_30_tickers

    # Update tickers for long credit.
    ix = db.build_market_index(
        rating=("A+", "A-"),
        maturity=(10, None),
        financial_flag=False,
        sector="LOCAL_AUTHORITIES",
        in_stats_index=True,
        special_rules="~Sector",
    )
    df = (
        ix.df.groupby(["Ticker"], observed=True)
        .sum()
        .sort_values("MarketValue", ascending=False)
    )
    top_30_tickers = list(df.index[:31])
    indexes["A_NON_FIN_TOP_30_10+"]["ticker"] = top_30_tickers
    indexes["A_NON_FIN_EX_TOP_30_10+"]["ticker"] = top_30_tickers

    # Update tickers for market credit.
    ix = db.build_market_index(
        rating=("A+", "A-"),
        financial_flag=False,
        sector="LOCAL_AUTHORITIES",
        in_stats_index=True,
        special_rules="~Sector",
    )
    df = (
        ix.df.groupby(["Ticker"], observed=True)
        .sum()
        .sort_values("MarketValue", ascending=False)
    )
    top_30_tickers = list(df.index[:31])
    indexes["A_NON_FIN_TOP_30"]["ticker"] = top_30_tickers
    indexes["A_NON_FIN_EX_TOP_30"]["ticker"] = top_30_tickers
    # Save changes.
    dump_json(indexes, fid)


# %%
db = Database()
today = db.date("today")
update_top_30_tickers(today)
