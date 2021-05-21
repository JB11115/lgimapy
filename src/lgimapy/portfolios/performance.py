from collections import defaultdict
from functools import lru_cache

import numpy as np
import pandas as pd
from tqdm import tqdm

from lgimapy.data import (
    concat_index_dfs,
    Database,
    IG_sectors,
    IG_market_segments,
    Index,
)
from lgimapy.utils import to_datetime, to_set
from lgimapy.data import credit_sectors, HY_sectors

# %%


class PerformancePortfolio:
    """
    Class for measuring PnL performance over time for a
    given portfolio.

    Parameters
    ----------
    start: datetime, optional
        Starting date for scrape.
    end: datetime, optional
        Ending date for scrape.
    account: str or List[str], optional
        Account(s) to include in scrape.
    strategy: str or List[str], optional
        Strategy(s) to include in scrape.
    **kwargs: Keyword arguments for :meth:`Database.load_portfolio`.

    Attributes
    ----------
    ix: :class:`Index`
        Index with ``PnL`` column in :attr:`Index.df` with daily
        performance in bp.
    """

    def __init__(
        self, account=None, strategy=None, start=None, end=None, **kwargs
    ):
        if start is None:
            raise ValueError("Must provide a start date.")

        self.account = account
        self.strategy = strategy
        self.name = self.account or self.strategy

        self._db = Database()
        self._dates = self._db.trade_dates(start=start, end=end)
        self.start, *_, self.end = self._dates
        self.ix = self._load_data(**kwargs)

    def __repr__(self):
        dates = f"{self.start:%m/%d/%Y} - {self.end:%m/%d/%Y}"
        if self.account is not None:
            return f"PerformanceIndex(account={self.account}, {dates})"
        elif self.strategy is not None:
            return f"PerformanceIndex(strategy={self.strategy}, {dates})"

    def _load_data(self, **kwargs):
        self._db.load_market_data(start=self.start, end=self.end)
        self._market_ix = self._db.build_market_index()

        df_list = []
        prev_acnt = self._db.load_portfolio(
            account=self.account,
            strategy=self.strategy,
            date=self.start,
            market_cols=False,
            **kwargs,
        )
        for date in self._dates[1:]:
            curr_acnt = self._db.load_portfolio(
                account=self.account,
                strategy=self.strategy,
                date=date,
                market_cols=False,
                **kwargs,
            )
            isins = set(prev_acnt.df["ISIN"]) & set(curr_acnt.df["ISIN"])
            curr_df = curr_acnt.df[curr_acnt.df["ISIN"].isin(isins)].set_index(
                "ISIN"
            )
            prev_df = prev_acnt.df[prev_acnt.df["ISIN"].isin(isins)].set_index(
                "ISIN"
            )
            df = self._market_ix.day(date)
            df = df[df["ISIN"].isin(isins)].set_index("ISIN", drop=False)

            oas_chg = curr_df["OAS"] - prev_df["OAS"]
            oad_avg = (curr_df["OAD_Diff"] + prev_df["OAD_Diff"]) / 2
            df["PnL"] = -oas_chg * oad_avg
            df_list.append(df)

            prev_acnt = curr_acnt

        return Index(concat_index_dfs(df_list))

    def tickers(self, tickers=None, start=None, end=None):
        """Find PnL of tickers."""
        df = self.ix.subset(start=start, end=end).df
        ticker_pnl = (
            df[["Ticker", "PnL"]]
            .groupby("Ticker", observed=True)
            .sum()
            .squeeze()
            .sort_values()
            .rename_axis(None)
        )
        ticker_pnl.index = ticker_pnl.index.astype(str)

        if tickers is not None:
            ticker_set = to_set(tickers, dtype=str)
            subset_pnl = ticker_pnl[ticker_pnl.index.isin(ticker_set)]
            missing_tickers = ticker_set - set(subset_pnl.index)
            missing_pnl = pd.Series(
                np.full(len(missing_tickers), np.nan), index=missing_tickers
            )
            return pd.concat((subset_pnl, missing_pnl))
        else:
            return ticker_pnl

    def sectors(self, sectors=None, start=None, end=None):
        """Find PnL of sectors."""
        ix = self.ix.subset(start=start, end=end)
        d = defaultdict(list)
        sectors = IG_sectors() if sectors is None else sectors
        for sector in sectors:
            kwargs = self._db.index_kwargs(
                sector, unused_constraints="in_stats_index"
            )
            sector_ix = ix.subset(**kwargs)
            d["sector"].append(kwargs["name"])
            d["PnL"].append(sector_ix.df["PnL"].sum())
        return (
            pd.DataFrame(d)
            .set_index("sector")
            .squeeze()
            .sort_values()
            .rename_axis(None)
        )

    def market_segments(self, segments=None, start=None, end=None):
        """Find PnL of sectors."""
        segments = IG_market_segments() if segments is None else segments
        return self.sectors(sectors=segments, start=start, end=end)

    def isins(self, isins=None, start=None, end=None):
        """Find PnL of isins."""
        df = self.ix.subset(start=start, end=end).df
        isin_pnl = (
            df[["ISIN", "PnL"]]
            .groupby("ISIN", observed=True)
            .sum()
            .squeeze()
            .sort_values()
            .rename_axis(None)
        )
        isin_pnl.index = isin_pnl.index.astype(str)

        if isins is not None:
            isin_set = to_set(isins, dtype=str)
            subset_pnl = isin_pnl[isin_pnl.index.isin(isin_set)]
            missing_isins = isin_set - set(subset_pnl.index)
            missing_pnl = pd.Series(
                np.full(len(missing_isins), np.nan), index=missing_isins
            )
            return pd.concat((subset_pnl, missing_pnl))
        else:
            return isin_pnl


# accounts = ["CARGLC", "SRPLC"]
# performance_ports = {
#     account: PerformancePortfolio(account, start=Database().date("YTD"))
#     for account in accounts
# }
# performance_ports
# db = Database()
# pp = PerformancePortfolio("P-LD", start=db.date("yesterday"))
# # %%
# pp.tickers()
#
# pp.sectors()
#
#
# pp.market_segments()
#

# %%


class PerformanceComp:
    """
    Comapare performance between accounts/straegies.

    Parameters
    ----------

    """

    def __init__(self, *perf_ports):
        self.perf_ports = list(*perf_ports)
        self.port_names = [pp.name for pp in self.perf_ports]

    def tickers(self, n=10, start=None, end=None):
        df = pd.concat(
            (
                pp.tickers(start=start, end=end).rename(pp.name)
                for pp in self.perf_ports
            ),
            axis=1,
        )
        return self._largest_variation(df, n)

    def sectors(self, sectors=None, start=None, end=None, n=10):
        df = pd.concat(
            (
                pp.sectors(sectors, start=start, end=end).rename(pp.name)
                for pp in self.perf_ports
            ),
            axis=1,
        )
        return self._largest_variation(df, n)

    def market_segments(self, segments=None, start=None, end=None, n=10):
        df = pd.concat(
            (
                pp.market_segments(segments, start=start, end=end).rename(
                    pp.name
                )
                for pp in self.perf_ports
            ),
            axis=1,
        )
        return self._largest_variation(df, n)

    def _largest_variation(self, df, n):
        """
        pd.DatFrame:
            Find top ``n`` rows with largest variation between portfolios.
        """
        # Find index of rows with most variation among portfolios.
        avg = df.fillna(0).mean(axis=1)
        std = df.fillna(0).std(axis=1).sort_values()
        if n is not None:
            idx = std[-n:].index[::-1]
        else:
            idx = std.index[::-1]
        table = df.loc[idx]
        table["Avg"] = avg
        cols = list(table.columns[-1:]) + list(table.columns[:-1])
        return table[cols]


# %%
# db = Database()
# self = PerformanceComp(performance_ports.values())
#
# # %%
# self.tickers(n=20)
# self.tickers(n=10, start=db.date("1m"))
#
#
# self.sectors(sectors=credit_sectors(), n=10)
# self.sectors(n=10, start=db.date("1m"))
#
#
# self.market_segments()
# self.market_segments(start=db.date("1m"))
#
#
# # %%
#
# carglc = performance_ports["CARGLC"]
# carglc.sectors(start="3/1/2021", end="3/31/2021")
