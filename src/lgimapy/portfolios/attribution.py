import argparse
from collections import defaultdict
from functools import lru_cache, cached_property

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
from lgimapy.utils import mkdir, replace_multiple, to_datetime, to_set, pprint
from lgimapy.data import credit_sectors, HY_sectors, IG_sectors

# %%


class AttributionIndex:
    """
    Class for measuring PnL performance over time for a
    given portfolio.

    Parameters
    ----------
    account: str or List[str], optional
        Account(s) to include in scrape.
    start: datetime, optional
        Starting date for scrape.
    end: datetime, optional
        Ending date for scrape.
    pbar: bool, default=False
        If ``True``, show a progress bar while loading data.
    **portfolio_kws: Keyword arguments for :meth:`Database.load_portfolio`.

    Attributes
    ----------
    ix: :class:`Index`
        Index with ``PnL`` column in :attr:`Index.df` with daily
        performance in bp.
    """

    def __init__(
        self,
        account=None,
        strategy=None,
        date=None,
        start=None,
        end=None,
        ix=None,
        db=None,
        pbar=False,
    ):
        self.account = account
        self.strategy = strategy
        if self.strategy is not None:
            raise NotImplementedError("Strategies are not implemented.")

        self.name = self.account or self.strategy
        self._pbar = pbar
        self._db = Database() if db is None else db

        # Load data for the portfolio and create an index with PnL.
        if date is not None:
            start = end = date
        self.ix = self._load_PnL_ix(start=start, end=end, ix=ix)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"{self.portfolio_type}='{self.account}', "
            f"start='{self.start:%m/%d/%Y}', "
            f"end='{self.end:%m/%d/%Y}')"
        )

    @cached_property
    def portfolio_type(self):
        if self.account is not None:
            return "Account"
        elif self.strategy is not None:
            return "Strategy"

    @cached_property
    def _local_PnL_fid(self):
        data_dir = self._db.local(
            f"portfolio_performance/{self.portfolio_type}"
        )
        mkdir(data_dir)
        # Clean portfolio name.
        repl = {" ": "_", "/": "_", "%": "pct"}
        fid = replace_multiple(self.name, repl)
        return data_dir / f"{fid}.parquet"

    @cached_property
    def _local_PnL_df(self):
        try:
            return pd.read_parquet(self._local_PnL_fid).T
        except FileNotFoundError:
            return None

    @cached_property
    def _local_PnL_dates(self):
        if self._local_PnL_df is None:
            return set()
        else:
            return set(pd.to_datetime(self._local_PnL_df.columns))

    def _load_PnL_ix(self, start=None, end=None, ix=None):
        """
        Load portfolio data for given :attr:`account` or :attr:`strategy`.
        If portoflio is listed under in :attr:`_locally_stored_accounts` then
        the data will try to be loaded locally, and any data loaded
        from the database will be stored upon loading for future use.

        Returns
        -------
        pd.DataFrame:
            Data for each portfolio loaded with "PnL" column
            appended showing estimate daily performance (in bp
            vs the respective benchmark.
        """
        # Load market data.
        if ix is None:
            self._db.load_market_data(start=start, end=end)
            ix = self._db.build_market_index()
            self.start = to_datetime(start)
            self.end = (
                self._db.date("today") if end is None else to_datetime(end)
            )
        else:
            self.end = ix.end_date
            self.start = ix.start_date

        # Add PnL as new column, mapped by date and ISIN.
        PnL_dict = self._get_PnL_df().fillna(0).to_dict()
        date_isin_s = ix.df[["ISIN", "Date"]].set_index("Date").squeeze()
        PnL = np.zeros(len(date_isin_s))
        for i, (date, isin) in enumerate(date_isin_s.items()):
            PnL[i] = PnL_dict[date].get(isin, 0)

        ix.df["PnL"] = PnL
        return ix

    def _get_PnL_df(self):
        """
        Scrape portoflio data and calculate PnL if required. Save any
        scraped data locally for future use.

        Returns
        -------
        pd.DataFrame:
            DataFrame of PnL data for current portfolio, with dates
            as columns and ISINs as index.
        """
        trade_dates = self._db.trade_dates(self.start, self.end)
        dates_to_scrape = sorted(list(set(trade_dates) - self._local_PnL_dates))
        last_scraped_date = None
        PnL_date_df_list = []
        for date in tqdm(dates_to_scrape, disable=(not self._pbar)):
            # Find preceding trade date, and scrape the account data
            # if necessary
            prev_date = self._db.trade_dates(exclusive_end=date)[-1]
            if prev_date == last_scraped_date:
                prev_acnt = curr_acnt
            else:
                prev_acnt = self._db.load_portfolio(
                    account=self.account,
                    strategy=self.strategy,
                    date=prev_date,
                )
            # Scrape the current account data.
            curr_acnt = self._db.load_portfolio(
                account=self.account,
                strategy=self.strategy,
                date=date,
            )
            last_scraped_date = date

            # Subset both accounts to only the bonds that exist in each.
            isins = set(prev_acnt.df["ISIN"]) & set(curr_acnt.df["ISIN"])
            curr_df = curr_acnt.df[curr_acnt.df["ISIN"].isin(isins)].set_index(
                "ISIN"
            )
            prev_df = prev_acnt.df[prev_acnt.df["ISIN"].isin(isins)].set_index(
                "ISIN"
            )

            # Calculate PnL using simple back of the envolope method
            # of multiplying change in spread by duration overweight
            # (average OW over the day by open and close values).
            oas_chg = curr_df["OAS"] - prev_df["OAS"]
            oad_ow_avg = (curr_df["OAD_Diff"] + prev_df["OAD_Diff"]) / 2
            pnl = -oas_chg * oad_ow_avg
            PnL_date_df_list.append(pnl.rename(date))

        if PnL_date_df_list:
            # New data was computed.
            new_PnL_df = pd.concat(PnL_date_df_list, axis=1).rename_axis(None)
            if self._local_PnL_df is not None:
                # Old PnL data exists, combine, save and return.
                full_PnL_df = pd.concat(
                    (self._local_PnL_df, new_PnL_df), axis=1
                )
                full_PnL_df.T.to_parquet(self._local_PnL_fid)
                return full_PnL_df
            else:
                # No old PnL data exists, save and return new data.
                new_PnL_df.T.to_parquet(self._local_PnL_fid)
                return new_PnL_df
        else:
            # No new PnL data needed to be computed.
            return self._local_PnL_df

        # return concat_index_dfs(df_list)

    def total(self, start=None, end=None):
        """Find total PnL of portfolio."""
        return self.ix.subset(start=start, end=end).df["PnL"].sum()

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
        sectors = IG_sectors(unique=True) if sectors is None else sectors
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

    def best_worst_df(self, s, n=10, prec=1):
        worst = s.iloc[:n]
        best = s.iloc[-n:][::-1]
        df = pd.DataFrame()
        df["Best"] = best.index
        df["PnL"] = best.values.round(prec)
        df["Worst"] = worst.index
        df["PnL "] = worst.values.round(prec)
        df.index += 1
        return df


# accounts = ["CARGLC", "SRPLC"]
# performance_ports = {
#     account: AttributionIndex(account, start=Database().date("YTD"))
#     for account in accounts
# }
# performance_ports
# db = Database()
# account = "P-LD"
# start = db.date("YTD")
# self = AttributionIndex(account, start=start)
# self.tickers()
# self.sectors()
#
# self.best_worst_df(self.tickers())

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


def parse_args():
    """Collect settings from command line and set defaults."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--portfolio", help="Portfolio")
    parser.add_argument("-s", "--start", help="Start Date")
    parser.set_defaults(portfolio="P-LD", start="yesterday")
    return parser.parse_args()


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


class temp:
    # start = "yesterday"
    # portfolio = "P-LD"
    db = Database()
    portfolio = "P-LD"
    start = "yesterday"
    end = ""


# args = temp()
#
# db = Database()
# pp = AttributionIndex(args.portfolio, start=db.date(args.start))
# pp.ix.df["PnL"].sum()
# %%


def main():
    # %%
    args = parse_args()
    # args = temp()

    db = Database()
    pp = AttributionIndex(args.portfolio, start=db.date(args.start))

    # %%
    print(f"\nTotal: {pp.total():+.2f} bp")

    print("\nTickers")
    pprint(get_best_worst_df(pp.tickers(), n=10))

    print("\n\nSectors")
    pprint(get_best_worst_df(pp.sectors(), n=10))

    print("\n\nMarket Segments")
    pprint(get_best_worst_df(pp.market_segments()))
    # %%


def creat_performance_file():
    # %%
    port_name = "JNJLC"
    fid = f"{port_name}_2019.xlsx"
    start = "12/31/2018"
    end = "12/31/2019"
    pp = AttributionIndex("JNJLC", start=start, end=end)
    excel = pd.ExcelWriter(fid)
    get_best_worst_df(pp.tickers(), n=40).to_excel(excel, sheet_name="Tickers")
    get_best_worst_df(pp.sectors(), n=20).to_excel(excel, sheet_name="Sectors")
    get_best_worst_df(pp.market_segments(), n=15).to_excel(
        excel, sheet_name="Market Segments"
    )
    excel.save()

    # %%


# %%


if __name__ == "__main__":
    main()


# %%
