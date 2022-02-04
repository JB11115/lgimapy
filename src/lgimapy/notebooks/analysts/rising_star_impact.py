from collections import defaultdict

import numpy as np
import pandas as pd

from lgimapy.data import Database, groupby, BondBasket

# %%
ticker = "KHC"
sector = "FOOD_AND_BEVERAGE"
strategies = ["Long Credit", "Market Credit"]
previous_date = "2/1/2020"

db = Database()
prev_date = db.nearest_date(previous_date, before=False)

date = prev_date
def get_previous_stats(strategy, sector, ticker, date, db):
    d = {}
    account = {"Long Credit": "P-LD", "Market Credit": "CITMC"}[strategy]
    port = db.load_portfolio(account=account, date=date)
    bm_df = port.ticker_df
    d["BM MV"] = f"${bm_df.loc[ticker, 'MarketValue']/1e3:,.1f}B"
    d["BM MV Weight"] = f"{bm_df.loc[ticker, 'BM_Weight']:.2%}"

    ticker_oad = bm_df.loc[ticker, "BM_OAD"]
    d["BM OAD"] = f"{ticker_oad:.3f} yrs"

    ticker_dts = bm_df.loc[ticker, "BM_DTS"]
    bm_dts = port.df["BM_DTS"].sum()
    d["BM DTS Weight"] = f"{ticker_dts / bm_dts:.2%}"

    strat_kwargs = {
        "Long Credit": {"maturity": (10, None)},
        "Market Credit": {},
    }
    sector_kwargs = db.index_kwargs(
        sector,
        **strat_kwargs[strategy],
        unused_constraints=["OAS", "in_stats_index"],
    )
    sector_port = port.subset(**sector_kwargs)
    sector_df = sector_port.ticker_df

    sector_bm_weight = sector_port.df["BM_Weight"].sum()
    sector_ticker_weight = sector_df.loc[ticker, "BM_Weight"]

    d["Sector MV Weight"] = f"{sector_ticker_weight / sector_bm_weight:.2%}"
    sector_ticker_dts = sector_df.loc[ticker, "BM_DTS"]
    sector_bm_dts = sector_port.df["BM_DTS"].sum()
    d["Sector DTS Weight"] = f"{sector_ticker_dts / sector_bm_dts:.2%}"

    return pd.Series(d).rename(date)

def get_expected_stats(strategy, sector, ticker, db, date=None):
    date = db.date('today') if date is None else db.nearest_date(date)
    d = {}
    strat_kwargs = {
        "Long Credit": {"maturity": (10, None), 'ticker': ticker},
        "Market Credit": {'ticker': ticker},
    }[strategy]
    db.load_market_data(date=date)
    ticker_ix = db.build_market_index(**strat_kwargs)
    account = {"Long Credit": "P-LD", "Market Credit": "CITMC"}[strategy]
    port = db.load_portfolio(account=account, universe='stats')

    # Add Rising Star bonds to the benchmark
    raw_bm_df = port.df[port.df['BM_Weight'] > 0]
    bm_cusip_df = pd.concat((raw_bm_df, ticker_ix.df))

    bm_cusip_df["BM_DTS"] = (
        bm_cusip_df["DTS"]
        * bm_cusip_df["MarketValue"]
        / np.sum(bm_cusip_df["MarketValue"])
    )
    bm_cusip_df["BM_OAD"] = (
        bm_cusip_df["OAD"]
        * bm_cusip_df["MarketValue"]
        / np.sum(bm_cusip_df["MarketValue"])
    )
    bm_cusip_df["BM_Weight"] = (
        bm_cusip_df["MarketValue"]
        / np.sum(bm_cusip_df["MarketValue"])
    )
    bm_df = groupby(bm_cusip_df, 'Ticker')

    d["BM MV"] = f"${bm_df.loc[ticker, 'MarketValue']/1e3:,.1f}B"
    d["BM MV Weight"] = f"{bm_df.loc[ticker, 'BM_Weight']:.2%}"

    ticker_oad = bm_df.loc[ticker, "BM_OAD"]
    d["BM OAD"] = f"{ticker_oad:.3f} yrs"

    ticker_dts = bm_df.loc[ticker, "BM_DTS"]
    bm_dts = port.df["BM_DTS"].sum()
    d["BM DTS Weight"] = f"{ticker_dts / bm_dts:.2%}"


    sector_kwargs = db.index_kwargs(
        sector,
        unused_constraints=["OAS", "in_stats_index", "maturity", 'ticker'],
    )
    bm_sector_ix = BondBasket(bm_cusip_df).subset(**sector_kwargs)
    sector_df = groupby(bm_sector_ix.df, 'Ticker')
    sector_bm_weight = sector_df["BM_Weight"].sum()
    sector_ticker_weight = sector_df.loc[ticker, "BM_Weight"]

    d["Sector MV Weight"] = f"{sector_ticker_weight / sector_bm_weight:.2%}"
    sector_ticker_dts = sector_df.loc[ticker, "BM_DTS"]
    sector_bm_dts = sector_df["BM_DTS"].sum()
    d["Sector DTS Weight"] = f"{sector_ticker_dts / sector_bm_dts:.2%}"
    return pd.Series(d).rename(date)


d = {}
for strategy in strategies:
    prev = get_previous_stats(strategy, sector, ticker, prev_date, db)
    new = get_expected_stats(strategy, sector, ticker, db)
    d[strategy] = pd.concat((new, prev), axis=1)
    fid = f"{ticker}_{strategy.replace(' ', '_')}_Rising_Star_analysis.csv"
    d[strategy].to_csv(fid)

# %%
d['Long Credit']
