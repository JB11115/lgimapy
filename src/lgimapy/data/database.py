import json
import os
import pickle
import sys
import warnings
from bisect import bisect_left
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from functools import lru_cache, partial, cached_property
from glob import glob
from inspect import cleandoc, getfullargspec
from itertools import chain
from pathlib import Path

import boto3
import datetime as dt
import fuzzywuzzy
import numpy as np
import pandas as pd
import pyodbc

from lgimapy.bloomberg import (
    bdp,
    get_bloomberg_subsector,
    update_issuer_business_strucure_json,
)
from lgimapy.data import (
    clean_dtypes,
    concat_index_dfs,
    convert_sectors_to_fin_flags,
    credit_sectors,
    groupby,
    HY_sectors,
    HY_market_segments,
    IG_sectors,
    IG_market_segments,
    Index,
    new_issue_mask,
    Account,
    Strategy,
)
from lgimapy.utils import (
    check_market,
    current_platform,
    dump_json,
    load_json,
    nearest_date,
    replace_multiple,
    root,
    sep_str_int,
    S_drive,
    to_datetime,
    to_int,
    to_list,
    to_set,
    X_drive,
)


# %%


def get_basys_fids(market):
    """
    Get list of BASys fids for given market.

    Parameters
    ----------
    market: ``{"US", "EUR", "GBP"}``, default="US"
        Market to get trade dates for.

    Returns
    -------
    pd.Series:
        Series of BASys fids with datetime index.
    """
    market = check_market(market)
    if sys.platform == "win32":
        dir = S_drive(f"FrontOffice/Bonds/BASys/CSVFiles/MarkIT/{market}/")
    elif sys.platform == "linux":
        dir = S_drive(f"FrontOffice/Bonds/BASys/CSVFiles/MarkIT/{market}/")
    fids = dir.glob("*")
    files = {}
    for fid in fids:
        if len(fid.stem) != 21:
            # Bad file.
            continue
        date = pd.to_datetime(fid.stem[-10:])
        files[date] = fid
    return pd.Series(files, name="fid").sort_index()


class Database:
    """
    Class to pull and clean index data from SQL database.

    Parameters
    ----------
    server: {'LIVE', 'DEV'}, default='LIVE'
        Choice between live and development servers.

    Attributes
    ----------
    df: pd.DataFrame
        Full DataFrame of all cusips over loaded period.
    """

    def __init__(self, server="LIVE", market="US"):
        self.server = server
        self.market = check_market(market)

    @cached_property
    def _datamart_conn(self):
        if sys.platform == "win32":
            return pyodbc.connect(
                "Driver={SQL Server};"
                f"SERVER=l00-lgimadatamart-sql,14333;"
                "DATABASE=LGIMADatamart;"
                "Trusted_Connection=yes;"
                f"UID=inv\{os.environ['USERNAME']};"
                f"PWD={os.environ['USERPASS']};"
            )
        elif sys.platform == "linux":
            return pyodbc.connect(
                "Driver={ODBC Driver 17 for SQL Server};"
                f"SERVER=l00-lgimadatamart-sql,14333;"
                "DATABASE=LGIMADatamart;"
                # f"UID=inv\{os.environ['USERNAME']};"
                # f"PWD={os.environ['USERPASS']};"
                "DOMAIN=inv;"
                "UID=quantslive_svc;"
                "PWD=Srv0ql092020;"
            )
        else:
            raise OSError(f"Unknown platform: {sys.platform}")

    @cached_property
    def current_env(self):
        return current_platform()

    @staticmethod
    def local(fid=""):
        return root(f"data/{fid}")

    @staticmethod
    def X_drive(fid=""):
        return X_drive(fid)

    @staticmethod
    def S_drive(fid=""):
        return S_drive(fid)

    def load_json(self, filename=None, empty_on_error=False, full_fid=None):
        return load_json(
            filename=filename, empty_on_error=empty_on_error, full_fid=full_fid
        )

    def dump_json(self, d, filename=None, full_fid=None, **kwargs):
        dump_json(d=d, filename=filename, full_fid=full_fid, **kwargs)

    def query_datamart(self, sql, **kwargs):
        """
        Query datamart using an SQL query.

        Parameters
        ----------
        sql: str
            SQL query to be executed or a table name.
        kwargs:
            Keyword arguments for ``pandas.read_sql()``.

        Returns
        -------
            pd.DataFrame or Iterator[pd.DataFrame]
        """
        with warnings.catch_warnings():
            # ignore warning for non-SQLAlchemy Connecton
            # see github.com/pandas-dev/pandas/issues/45660
            warnings.simplefilter("ignore", UserWarning)
            return pd.read_sql(sql, self._datamart_conn, **kwargs)

    @cached_property
    def _passwords(self):
        if sys.platform == "linux":
            pswd_fid = Path.home() / "pswd.json"
        elif sys.platform == "win32":
            pswd_fid = Path("P:/pswd.json")
        else:
            raise OSError(f"Unknown platform: {sys.platform}")
        return load_json(full_fid=pswd_fid)

    @cached_property
    def _3PDH_sess(self):
        """Dict[str: int]: Memoized rating map."""
        import awswrangler as wr
        import boto3

        return boto3.Session(
            aws_access_key_id=self._passwords["AWS"]["prod_access_key"],
            aws_secret_access_key=self._passwords["AWS"][
                "prod_secret_access_key"
            ],
            region_name="us-east-2",
        )

    @cached_property
    def _ratings(self):
        """Dict[str: int]: Memoized rating map."""
        ratings_map = load_json("ratings")
        for i in range(23):
            ratings_map[i] = i
        return ratings_map

    @cached_property
    def all_dates(self):
        """List[datetime]: Memoized list of all dates in DataBase."""
        return list(self._trade_date_df.index)

    @cached_property
    def all_dates(self):
        """List[datetime]: Memoized list of all dates in DataBase."""
        return list(self._trade_date_df.index)

    def trade_dates(
        self,
        start=None,
        end=None,
        exclusive_start=None,
        exclusive_end=None,
        market=None,
    ):
        """
        List of trade dates in database.

        Parameters
        ----------
        start: datetime, optional
            Inclusive starting date for trade dates.
        end: datetime, optional
            Inclusive end date for trade dates.
        exclusive_start: datetime, optional
            Exclusive starting date for trade dates.
        exclusive_end: datetime, optional
            Exclusive end date for trade dates.
        market: ``{"US", "EUR", "GBP"}``, optional
            Market to get trade dates for.
            Defaults to :attr:`Database.market`.

        Returns
        -------
        List[datetime]:
            Trade dates in specified range.
        """
        market = self.market if market is None else check_market(market)
        df = self._trade_date_df(market)
        trade_dates = df[df["holiday"] == 0]
        if start is not None:
            trade_dates = trade_dates[trade_dates.index >= to_datetime(start)]
        if exclusive_start is not None:
            trade_dates = trade_dates[
                trade_dates.index > to_datetime(exclusive_start)
            ]
        if end is not None:
            trade_dates = trade_dates[trade_dates.index <= to_datetime(end)]
        if exclusive_end is not None:
            trade_dates = trade_dates[
                trade_dates.index < to_datetime(exclusive_end)
            ]
        return list(trade_dates.index)

    @lru_cache(maxsize=None)
    def holiday_dates(self, market=None):
        """List[datetime]: Memoized list of holiday dates."""
        market = self.market if market is None else check_market(market)
        df = self._trade_date_df(market)
        holidays = df[df["holiday"] == 1]
        return list(holidays.index)

    @lru_cache(maxsize=None)
    def _trade_date_df(self, market):
        """pd.DataFrame: Memoized trade date boolean series for holidays."""
        if market == "US":
            fid = self.local(f"{market}/trade_dates.parquet")
        else:
            fid = self.local(f"{market}/trade_dates_{sys.platform}.parquet")
        return pd.read_parquet(fid)

    @lru_cache(maxsize=None)
    def _index_kwargs_dict(self, source=None):
        """dict[str: dict]: keyword arguments for saved indexes."""
        if source is None:
            source = "bloomberg" if self.market == "US" else "iboxx"
        source = source.lower()
        allowable_sources = {"bloomberg", "iboxx", "baml", "bloomberg_js"}
        if source in allowable_sources:
            return load_json(f"index_kwargs/{source}")
        else:
            raise ValueError(
                f"'{source}' not in allowable sources. "
                f"Please select one of {allowable_sources}."
            )

    @cached_property
    def _rating_changes_df(self):
        """pd.DataFrame: Rating change history."""
        fid = self.local("rating_changes.parquet")
        df = pd.read_parquet(fid)
        for col in ["Date_NEW", "Date_PREV"]:
            df[col] = pd.to_datetime(df[col])
        return df

    @cached_property
    def _ticker_changes(self):
        """Dict[str: str]: Memoized map of hisorical ticker changes."""
        return load_json("ticker_changes")

    @cached_property
    def _utility_business_structure(self):
        """Dict[str: str]: Memoized map of US states to state codes."""
        return load_json("utility_business_structure")

    @cached_property
    def _BAMLSector_Ticker_map(self):
        """Dict[str: str]: Memoized map of tickers to BAML sectors."""
        return load_json("HY/sector_maps/BAML_Ticker_L4_map")

    @cached_property
    def _BAMLTopLevelSector_Ticker_map(self):
        """Dict[str: str]: Memoized map of tickers to BAML top level sectors."""
        return load_json("HY/sector_maps/BAML_Ticker_L3_map")

    @cached_property
    def _BAMLSector_Issuer_map(self):
        """Dict[str: str]: Memoized map of tickers to BAML sectors."""
        return load_json("HY/sector_maps/BAML_Issuer_L4_map")

    @cached_property
    def _BAMLTopLevelSector_Issuer_map(self):
        """Dict[str: str]: Memoized map of tickers to BAML top level sectors."""
        return load_json("HY/sector_maps/BAML_Issuer_L3_map")

    @cached_property
    def state_codes(self):
        """Dict[str: str]: Memoized map of US states to state codes."""
        return load_json("state_codes")

    @cached_property
    def state_populations(self):
        """Dict[str: int]: Memoized map of US states to their populations."""
        return load_json("state_populations")

    @cached_property
    def long_corp_sectors(self):
        """List[str]: list of sectors in Long Corp Benchmark."""
        fid = self.local("long_corp_sectors.parquet")
        sector_df = pd.read_parquet(fid)
        return sorted(sector_df["Sector"].values)

    @staticmethod
    def credit_sectors(*args, **kwargs):
        return credit_sectors(*args, **kwargs)

    @staticmethod
    def HY_sectors(*args, **kwargs):
        return HY_sectors(*args, **kwargs)

    @staticmethod
    def HY_market_segments(*args, **kwargs):
        return HY_market_segments(*args, **kwargs)

    @staticmethod
    def IG_sectors(*args, **kwargs):
        return IG_sectors(*args, **kwargs)

    @staticmethod
    def IG_market_segments(*args, **kwargs):
        return IG_market_segments(*args, **kwargs)

    @staticmethod
    def fid_safe_str(s):
        repl = {" ": "_", "/": "_", "%": "pct"}
        return replace_multiple(s, repl)

    @cached_property
    def _defaults_df(self):
        fid = self.local("defaults.csv")
        df = pd.read_csv(fid, index_col=0)
        df.columns = ["Ticker", "Date", "ISIN"]
        bad_dates = {"#N/A Invalid Security"}
        bad_isins = {"#N/A Requesting Data...", "#N/A Field Not Applicable"}
        df = df[
            ~((df["Date"].isin(bad_dates)) | (df["ISIN"].isin(bad_isins)))
        ].copy()
        df["Date"] = pd.to_datetime(df["Date"])
        return df.sort_values("Date").reset_index(drop=True).rename_axis(None)

    def defaults(self, start=None, end=None, isins=None):
        """
        Get history of defaults.

        Parameters
        ----------
        start: datetime, optional
            Start date for rating changes.
        end: datetime, optional
            End date for rating changes.
        isins: List[str], optional
            List of ISINs to subset.

        Returns
        -------
        df: pd.DataFrame
            Rating changes within specified dates.
        """
        df = self._defaults_df.copy()
        if start is not None:
            df = df[df["Date"] >= pd.to_datetime(start)].copy()
        if end is not None:
            df = df[df["Date"] <= pd.to_datetime(end)].copy()
        if isins is not None:
            df = df[df["ISIN"].isin(isins)].copy()
        return df

    def rating_changes(
        self,
        start=None,
        end=None,
        fallen_angels=False,
        rising_stars=False,
        upgrades=False,
        downgrades=False,
        isins=None,
    ):
        """
        Get history of rating changes.

        Parameters
        ----------
        start: datetime, optional
            Start date for rating changes.
        end: datetime, optional
            End date for rating changes.
        fallen_angels: bool, default=False
            If ``True`` include fallen angels only. If ``rising_stars``
            is also ``True`` include both.
        rising_stars: bool, default=False
            If ``True`` include fallen angels only. If ``fallen_angels``
            is also ``True`` include both.
        upgrades: bool, default=False
            If ``True`` subset to upgrades from any agency or
            upgrade to composite rating.
        downgrades: bool, default=False
            If ``True`` subset to downgrades from any agency or
            downgrade to composite rating.
        isins: List[str], optional
            List of ISINs to subset.

        Returns
        -------
        df: pd.DataFrame
            Rating changes within specified dates.
        """
        df = self._rating_changes_df.copy()
        if start is not None:
            df = df[df["Date_PREV"] >= pd.to_datetime(start)].copy()
        if end is not None:
            df = df[df["Date_PREV"] <= pd.to_datetime(end)].copy()

        if fallen_angels or rising_stars:
            is_fallen_angel = (
                df[f"NumericRating_PREV"] <= self.convert_letter_ratings("BBB-")
            ) & (df[f"NumericRating_NEW"] >= self.convert_letter_ratings("BB+"))
            is_rising_star = (
                df[f"NumericRating_PREV"] >= self.convert_letter_ratings("BB+")
            ) & (
                df[f"NumericRating_NEW"] <= self.convert_letter_ratings("BBB-")
            )
            if fallen_angels and rising_stars:
                df = df[is_fallen_angel | is_rising_star].copy()
            elif fallen_angels:
                df = df[is_fallen_angel].copy()
            else:
                df = df[is_rising_star].copy()

        if upgrades:
            df = df[
                (df["NumericRating_CHANGE"] > 0)
                | (df["SPRating_CHANGE"] > 0)
                | (df["MoodyRating_CHANGE"] > 0)
                | (df["FitchRating_CHANGE"] > 0)
            ].copy()

        if downgrades:
            df = df[
                (df["NumericRating_CHANGE"] < 0)
                | (df["SPRating_CHANGE"] < 0)
                | (df["MoodyRating_CHANGE"] < 0)
                | (df["FitchRating_CHANGE"] < 0)
            ].copy()

        if isins is not None:
            df = df[df["ISIN"].isin(isins)].copy()

        return df

    def index_kwargs(self, key, unused_constraints=None, source=None, **kwargs):
        """
        Index keyword arguments for saved indexes,
        with ability to override/add/remove arguments.

        Parameters
        ----------
        key: str
            Key of stored index in `indexes.json`.
        unused_constraints: str or List[str], optional
            Constraintes to remove from kwargs list if present.
        source: ``{"bloomberg", "iboxx", "baml"}``, optional
            Source for index kwargs. Defaults based on current
            :attr:`market`.
        kwargs:
            Keyword arguments to override or add to index.

        Returns
        -------
        dict:
            Keyword arguments and respective constraints
            for specified index.
        """
        if source is None:
            source = "bloomberg" if self.market == "US" else "iboxx"
        try:
            d = self._index_kwargs_dict(source)[key].copy()
        except KeyError:
            raise KeyError(f"{key} is not a stored Index.")

        if unused_constraints is not None:
            unused_cons = to_set(unused_constraints, dtype=str)
            d = {k: v for k, v in d.items() if k not in unused_cons}

        d.update(**kwargs)
        return d

    def _subset_date_series(self, dates, start=None, end=None):
        if start is not None:
            dates = dates[dates.index >= to_datetime(start)]
        if end is not None:
            dates = dates[dates.index <= to_datetime(end)]
        return dates

    def date(
        self,
        date_delta,
        reference_date=None,
        trade_dates=None,
        market=None,
        fcast=False,
        start=None,
        end=None,
        **kwargs,
    ):
        """
        Find date relative to a specified date.

        Parameters
        ----------
        date_delta: str, ``{'today', 'WTD', 'MTD', 'YTD', '3d', '5y', etc}``
            Difference between reference date and target date.
        reference_date: datetime, optional
            Reference date to use, if None use most recent trade date.
        trade_dates: List[datetime], optional
            Trade dates to use in finding date, defaults to IG trade
            dates for current ``market``.
        market: ``{"US", "EUR", "GBP"}``, optional
            Market to get trade dates for.
            Defaults to :attr:`Database.market`.
        fcast: bool, default=False
            If ``True`` forecast a future date past what is available
            in the Database. This does not account for weekends or holidays.
        kwargs:
            Keyword arguments for :func:`nearest_date`.

        Returns
        -------
        datetime:
            Target date from reference and delta.

        See Also
        --------
        :func:`nearest_date`
        """
        market = self.market if market is None else check_market(market)
        date_delta = date_delta.upper()
        if trade_dates is None:
            trade_dates = self.trade_dates(market=market)

        if date_delta == "PORTFOLIO_START":
            return pd.to_datetime("9/1/2018")
        elif date_delta == "HY_START":
            return pd.to_datetime("6/1/2020")
        elif date_delta == "MARKET_START":
            return {
                "US": pd.to_datetime("2/2/1998"),
                "EUR": pd.to_datetime("12/1/2014"),
                "GBP": pd.to_datetime("12/1/2014"),
            }[market]
        elif date_delta == "MONTH_STARTS":
            all_dates = pd.Series(trade_dates, trade_dates)
            dates = all_dates.groupby(pd.Grouper(freq="M")).first().iloc[1:]
            return list(self._subset_date_series(dates, start, end))
        elif date_delta == "MONTH_ENDS":
            all_dates = pd.Series(trade_dates, trade_dates)
            dates = all_dates.groupby(pd.Grouper(freq="M")).last().iloc[1:]
            return list(self._subset_date_series(dates, start, end))
        elif date_delta == "YEAR_STARTS":
            all_dates = pd.Series(trade_dates, trade_dates)
            dates = all_dates.groupby(pd.Grouper(freq="Y")).first().iloc[1:]
            return list(self._subset_date_series(dates, start, end))
        elif date_delta == "YEAR_ENDS":
            all_dates = pd.Series(trade_dates, trade_dates)
            dates = all_dates.groupby(pd.Grouper(freq="Y")).last().iloc[1:]
            return list(self._subset_date_series(dates, start, end))

        # Use today's date as reference date if none is provided,
        # otherwise convert the provided date to a datetime object.
        if reference_date is None:
            ref_date = trade_dates[-1]
        elif isinstance(reference_date, int) and 1980 < reference_date < 2100:
            # Convert year to datetime object for the first of that year.
            ref_date = pd.to_datetime(str(reference_date))
        else:
            ref_date = to_datetime(reference_date)

        last_trade = partial(bisect_left, trade_dates)
        if date_delta == "TODAY":
            return ref_date
        elif date_delta in {"YESTERDAY", "DAILY", "1D", "1DAY", "1DAYS"}:
            return self.nearest_date(
                ref_date, market=market, inclusive=False, after=False
            )
        elif date_delta in {"WTD", "LAST_WEEK_END"}:
            return trade_dates[
                last_trade(ref_date - timedelta(ref_date.weekday() + 1)) - 1
            ]
        elif date_delta == "MONTH_START":
            return self.nearest_date(
                ref_date.replace(day=1), market=market, before=False
            )
            return trade_dates[last_trade(ref_date.replace(day=1))]
        elif date_delta == "NEXT_MONTH_START":
            # Find first trade date that is on or after the 1st
            # of the next month.
            next_month = 1 if ref_date.month == 12 else ref_date.month + 1
            next_year = (
                ref_date.year + 1 if ref_date.month == 12 else ref_date.year
            )
            next_month_start = pd.to_datetime(f"{next_month}/1/{next_year}")
            return self.nearest_date(
                next_month_start, market=market, before=False
            )
        elif date_delta == "MONTH_END":
            next_month = self.date("NEXT_MONTH_START", reference_date=ref_date)
            return self.date("MTD", reference_date=next_month)
        elif date_delta in {"MTD", "LAST_MONTH_END"}:
            return trade_dates[last_trade(ref_date.replace(day=1)) - 1]
        elif date_delta == "YEAR_START":
            return self.nearest_date(
                ref_date.replace(day=1, month=1), market=market, before=False
            )
        elif date_delta == "NEXT_YEAR_START":
            next_year = ref_date.year + 1
            next_year_start = pd.to_datetime(f"1/1/{next_year}")
            return self.nearest_date(
                next_year_start, market=market, before=False
            )
        elif date_delta == "YEAR_END":
            next_year = self.date("NEXT_YEAR_START", reference_date=ref_date)
            return self.date("YTD", reference_date=next_year)
        elif date_delta in {"YTD", "LAST_YEAR_END"}:
            return trade_dates[last_trade(ref_date.replace(month=1, day=1)) - 1]
        else:
            # Assume value-unit specification.
            positive = "+" in date_delta
            val, unit = sep_str_int(date_delta.strip("+"))
            reverse_kwarg_map = {
                "days": ["D", "DAY", "DAYS"],
                "weeks": ["W", "WK", "WEEK", "WEEKS"],
                "months": ["M", "MO", "MONTH", "MONTHS"],
                "years": ["Y", "YR", "YEAR", "YEARS"],
            }
            dt_kwarg_map = {
                key: kwarg
                for kwarg, keys in reverse_kwarg_map.items()
                for key in keys
            }
            dt_kwargs = {dt_kwarg_map[unit]: val}
            if positive:
                if fcast:
                    return ref_date + relativedelta(**dt_kwargs)
                else:
                    return self.nearest_date(
                        ref_date + relativedelta(**dt_kwargs),
                        market=market,
                        **kwargs,
                    )
            else:
                return self.nearest_date(
                    ref_date - relativedelta(**dt_kwargs),
                    market=market,
                    **kwargs,
                )

    def get_account_market_values(self):
        sql = f"""
            SELECT AV.DateKey, AV.MarketValue, DA.BloombergID
            FROM dbo.AccountValue AV
            JOIN dbo.DimAccount DA ON AV.AccountKey = DA.AccountKey
            WHERE AV.DateKey >= {self.date("PORTFOLIO_START"):%Y%m%d}
            """

        df = self.query_datamart(sql)
        df["DateKey"] = pd.to_datetime(df["DateKey"], format="%Y%m%d")
        return (
            pd.pivot_table(
                df,
                values="MarketValue",
                columns=["BloombergID"],
                index="DateKey",
            )
            / 1e6
        ).round(6)

    @cached_property
    def account_market_values(self):
        fid = self.local("account_market_values.parquet")
        return pd.read_parquet(fid)

    @cached_property
    def _strategy_benchmark_map(self):
        return load_json("strategy_benchmarks")

    @lru_cache(maxsize=None)
    def _account_strategy_map(self, date=None):
        """Dict[str: str]: Account name to strategy map."""
        date = self.date("today") if date is None else to_datetime(date)
        fid = self.local(f"portfolios/account_strategy_maps/{date:%Y-%m-%d}")
        return load_json(full_fid=fid)

    @lru_cache(maxsize=None)
    def _strategy_account_map(self, date=None):
        """Dict[str: str]: Respective accounts for each strategy."""
        date = self.date("today") if date is None else to_datetime(date)
        fid = self.local(f"portfolios/strategy_account_maps/{date:%Y-%m-%d}")
        return load_json(full_fid=fid)

    def strategies(self, date=None):
        return sorted(list(self._strategy_account_map(date).keys()))

    def accounts(self, date=None):
        return sorted(list(self._account_strategy_map(date).keys()))

    @cached_property
    def _manager_to_accounts(self):
        """Dict[str: str]: Respective accounts for each PM."""
        return load_json("manager_accounts")

    @cached_property
    def DM_countries(self, region=None):
        """Set[str]: DM Country codes."""
        fid = self.local("DM_countries.parquet")
        return set(pd.read_parquet(fid).squeeze())

    @cached_property
    def _country_codes(self, region=None):
        """Dict[str: str]: Country codes file."""
        return load_json("country_codes")

    def country_codes(self, region=None):
        """Dict[str: str]: Developed market country codes."""
        codes = self._country_codes
        if region is None:
            d = {}
            for region_d in codes.values():
                d = {**d, **region_d}
            return d
        else:
            return codes[region.upper()]

    @lru_cache(maxsize=None)
    def _hy_index_flags(self, index):
        """pd.DataFrame: Index flag boolean for ISIN vs date."""
        fid = self.local(f"index_members/{index}.parquet")
        return pd.read_parquet(fid)

    def load_trade_dates(self):
        """List[datetime]: Dates with credit data."""
        sql = "select distinct effectivedatekey from \
            dbo.InstrumentAnalytics order by effectivedatekey"
        return list(
            pd.to_datetime(
                self.query_datamart(sql).values.ravel(),
                format="%Y%m%d",
            )
        )

    @cached_property
    def _numeric_to_letter_ratings(self):
        """Dict[int, str]: Numeric rating to letter map."""
        rating_dict = load_json("numeric_to_SP_ratings")
        # Convert input keys to int.
        return {int(k): v for k, v in rating_dict.items()}

    def convert_numeric_ratings(self, ratings):
        """
        Convert numeric ratings to S&P letter ratings.

        Parameters
        ----------
        rating: int, Iterable[int], or pd.Series
            Numeric rating(s) to convert.

        Returns
        -------
        str, List[str], or pd.Series
            S&P letter ratings for input values.
        """

        if isinstance(ratings, pd.Series):
            return ratings.map(self._numeric_to_letter_ratings)
        elif isinstance(ratings, int):
            return self._numeric_to_letter_ratings[ratings]
        elif isinstance(ratings, float):
            if ratings.is_integer():
                return self._numeric_to_letter_ratings[int(ratings)]
            else:
                raise ValueError(f"Input must be an integer.")
        else:
            return [self._numeric_to_letter_ratings[r] for r in ratings]

    def convert_letter_ratings(self, ratings):
        """
        Convert letter ratings to numeric values.

        Parameters
        ----------
        rating: str, Iterable[str], or pd.Series
            Letter rating(s) to convert.

        Returns
        -------
        int, List[int], or pd.Series
            Numeric values for input letter ratings.
        """
        if isinstance(ratings, pd.Series):
            return ratings.map(self._ratings)
        elif isinstance(ratings, str):
            return self._ratings[ratings]
        else:
            return [self._ratings[r] for r in ratings]

    def display_all_columns(self):
        """Set DataFrames to display all columnns in IPython."""
        pd.set_option("display.max_columns", 999)

    def display_all_rows(self, n=500):
        """Set DataFrames to display all columnns in IPython."""
        pd.set_option("display.max_rows", n)
        pd.set_option("display.min_rows", n)

    def nearest_date(self, date, market=None, **kwargs):
        """
        Return trade date nearest to input date.

        Parameters
        ----------
        date: datetime object
            Input date.
        market: ``{"US", "EUR", "GBP"}``, optional
            Market to get trade dates for.
            Defaults to :attr:`Database.market`.
        inclusive: bool, default=True
            Whether to include to specified reference date in
            the searchable list.
        before: bool, default=True
            Whether to include dates before the reference date.
        after: bool, default=True
            Whether to include dates after the reference date.

        Returns
        -------
        datetime:
            Trade date nearest to input date.

        See Also
        --------
        :func:`nearest_date`
        """
        market = self.market if market is None else check_market(market)
        return nearest_date(date, self.trade_dates(market=market), **kwargs)

    def _standardize_cusips(self, df):
        """
        Standardize CUSIPs, converting cusips which changed name
        to most recent cusip value for full history.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with cusip index.

        Returns
        -------
        df: pd.DataFrame
            DataFrame with all CUSIPs updated to current values.
        """
        dates = self.loaded_dates[::-1]
        df.set_index("CUSIP", inplace=True)

        # Build CUSIP map of CUSIPs which change.
        # TODO: add 'Issuer' back to multi_ix
        multi_ix = [  # index for identifying CUSIP changes
            "Ticker",
            "CouponRate",
            "MaturityDate",
            "IssueDate",
        ]
        cusip_map = {}
        for i in range(len(dates) - 1):
            df_today = df[df["Date"] == dates[i]].copy()
            df_yesterday = df[df["Date"] == dates[i + 1]].copy()

            # Find CUSIPs that were dropped from yesterday to today
            # and CUSIPs that were newly added today.
            df_dropped = df_yesterday[
                ~df_yesterday.index.isin(df_today.index)
            ].copy()
            df_new = df_today[~df_today.index.isin(df_yesterday.index)].copy()

            # Store CUSIPs, and set index to change identifier.
            df_dropped["CUSIP_old"] = df_dropped.index
            df_new["CUSIP_new"] = df_new.index
            df_dropped.set_index(multi_ix, inplace=True)
            df_new.set_index(multi_ix, inplace=True)

            # Store instances where CUSIPs change in cusip_map dict.
            df_new[df_new.index.isin(df_dropped.index)]
            df_change = df_new[["CUSIP_new"]].join(
                df_dropped[["CUSIP_old"]], how="inner"
            )
            for _, row in df_change.iterrows():
                cusip_map[row["CUSIP_old"]] = row["CUSIP_new"]

        # Update CUSIP map to account for CUSIPs which change multiple times.
        rev_CUSIPs = list(cusip_map.keys())[::-1]
        for i, key in enumerate(rev_CUSIPs):
            for k in rev_CUSIPs[i:]:
                if cusip_map[k] == key:
                    cusip_map[k] = cusip_map[key]

        # Map old CUSIPs to new CUSIPs and reset index.
        df.index = [cusip_map.get(ix, ix) for ix in df.index]
        df["CUSIP"] = df.index.astype("category")
        df.reset_index(inplace=True, drop=True)
        return df

    def _preprocess_market_data(self, df):
        """
        Convert dtypes for columns in sql_df to save memory.
        Column names are corrected and values in categorical
        columns which are identical are converted to a single
        set of standard values to reduce number of categories.

        Parameters
        ----------
        df: pd.DataFrame
            Raw DataFrame from SQL query.

        Returns
        -------
        pd.DataFrame
            DataFrame with correct column dtypes.
        """
        # Fix bad column names.
        col_map = {
            "Price": "CleanPrice",
            "IndustryClassification4": "Sector",
            "RVRecommendation_USD": "AnalystRating",
            "MLFI_Classification3": "BAMLTopLevelSector",
            "MLFI_Classification4": "BAMLSector",
            "MLFI_ModDurtoMat": "ModDurtoMat",
            "MLFI_ModDurtoWorst": "ModDurtoWorst",
            "YieldAndSpreadWorkoutDate": "WorkoutDate",
        }
        df.rename(columns=col_map, inplace=True)

        # Make flag columns -1 for nonexistent data to
        # mitigate memory usage by allwoing int dtype.
        df["Eligibility144AFlag"].fillna(-1, inplace=True)

        # Convert str time to datetime.
        date_cols = [
            "Date",
            "MaturityDate",
            "IssueDate",
            "NextCallDate",
            "WorkoutDate",
        ]
        for date_col in date_cols:
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(
                    df[date_col], format="%Y-%m-%d", errors="coerce"
                )

        # Capitalize market of issue, sector, and issuer.
        for col in ["MarketOfIssue", "Sector", "Issuer"]:
            df[col] = df[col].str.upper()

        # Clean tickers.
        df["Ticker"] = self._clean_tickers(df["Ticker"])

        # Map sectors to standard values.
        # For BAML columns, populate sectors based on ticker maps.
        df["Sector"] = self._clean_sectors(df["Sector"])
        self._populate_BAML_sectors(df)
        sector_map = {"OTHER_FINANCIAL": "FINANCIAL_OTHER"}
        for old_val, sector in sector_map.items():
            df.loc[df["Sector"] == old_val, "Sector"] = sector

        # Map coupon types to standard values.
        coupon_map = {
            "Fixed Coupon": "FIXED",
            "FIXED, OID": "FIXED",
            "Step-up Multiple": "STEP CPN",
            "Step-up Once": "STEP CPN",
            "Fixed Multiple Step-up w Scheduled Payments": "STEP CPN",
            "ZERO": "ZERO COUPON",
            "Hybrid w Scheduled Payments": "HYBRID VARIABLE",
            "Hybrid Fixed-to-Float": "HYBRID VARIABLE",
            "Fixed Rate w Partial Pay-In-Kind": "FIXED PIK",
        }
        for old_val, c_type in coupon_map.items():
            df.loc[df["CouponType"] == old_val, "CouponType"] = c_type

        # Map call types to standard values.
        calltype_map = {
            "Callable, Never Refundable": "CALL/NR",
            "Callable with Exception, Never Refundable": "CALL/NR",
            "European Call (Callable Only on Scheduled Dates)": "EUROCAL",
            "Noncallable": "NONCALL",
            "Noncallable with Exception, Refundable": "NONCALL",
            "NCX/NR": "NONCALL",
            "Callable, Refundable": "CALL/RF",
            "Make Whole (Call is ignored by option models)": "MKWHOLE",
        }
        for old_val, c_type in calltype_map.items():
            df.loc[df["CallType"] == old_val, "CallType"] = c_type

        # Map collateral types to standard values.
        equiv_ranks = {
            "UNSECURED": ["BONDS", "SR UNSECURED", "NOTES", "COMPANY GUARNT"],
            "SECURED": ["SR SECURED"],
            "1ST MORTGAGE": ["1ST REF MORT", "GENL REF MORT"],
        }
        for key, val in equiv_ranks.items():
            df.loc[df["CollateralType"].isin(val), "CollateralType"] = key

        # Fix collateral type for treasuries and local authorities.
        collateral_map = {
            "TREASURIES": "TREASURY_UNSECURED",
            "LOCAL_AUTHORITIES": "STATE",
        }
        for sector, c_type in collateral_map.items():
            df.loc[df["Sector"] == sector, "CollateralType"] = c_type

        # Add AAA ratings to treasury strips which have no ratings.
        cols = ["MoodyRating", "SPRating", "FitchRating"]
        strip_mask = df["Ticker"] == "SP"
        for col in cols:
            df.loc[strip_mask, col] = "AAA"
        # Fill NaNs for rating categories.
        df[cols] = df[cols].fillna("NR")

        # Make missing analyst ratings NaN.
        df["AnalystRating"] = df["AnalystRating"].replace("NR", np.nan)

        # Fix treasury strip coupon to zero coupon.
        df.loc[strip_mask, "CouponType"] = "ZERO COUPON"

        # Calculate DTS.
        df["DTS"] = df["OAS"] * df["OASD"]

        # Add financial flag column.
        df["FinancialFlag"] = convert_sectors_to_fin_flags(df["Sector"])

        # Add bloomberg subsector.
        df["Subsector"] = get_bloomberg_subsector(df["CUSIP"].values)

        # Add Opco/Holdco business structure for utilities.
        # utility_sectors = {"NATURAL_GAS", "ELECTRIC", "UTILITY_OTHER"}
        # utility_loc = df["Sector"].isin(utility_sectors)
        # update_issuer_business_strucure_json(df[utility_loc], db=self)
        # df.loc[utility_loc, "Subsector"] = df.loc[utility_loc, "Issuer"].map(
        #     self._business_structure_map
        # )

        return clean_dtypes(df)

    def _clean_sectors(self, sector):
        """
        Clean sector names so they can be saved as a fid.

        Parameters
        ----------
        sector: pd.Series
            Raw sector names.
        Returns
        -------
        pd.Series:
            Cleaned sector names.
        """
        pattern = "|".join([" - ", "-", " / ", "/", " "])
        s = (
            sector.str.upper()
            .str.replace("P&C", "P_AND_C", regex=True)
            .str.replace("&", "AND", regex=True)
            .str.replace(",", "", regex=True)
            .str.replace(pattern, "_", regex=True)
        )
        sector_map = {
            "GOVERNMENT_GUARANTEED": "GOVERNMENT_GUARANTEE",
            "GOVERNMENT_OWNED_NO_GUARANTEE": "OWNED_NO_GUARANTEE",
            "CONSUMER_CYC_SERVICES": "CONSUMER_CYCLICAL_SERVICES",
            "CONSUMER_CYC_SERVICES": "CONSUMER_CYCLICAL_SERVICES",
            "OTHER_INDUSTRIAL": "INDUSTRIAL_OTHER",
        }
        loc = s[s.isin(sector_map.keys())].index
        s.loc[loc] = s.loc[loc].map(sector_map)
        return s

    def _clean_tickers(self, tickers):
        """
        Clean tickers correcting historical tickers
        when they change but maintain the same credit
        risk.

        Parameters
        ----------
        tickers: pd.Series
            Series of tickers.

        Returns
        -------
        clean_tickers: pd.Series:
            Cleaned series of tickers.
        """
        clean_tickers = tickers.copy()
        loc = clean_tickers.isin(self._ticker_changes)
        clean_tickers.loc[loc] = clean_tickers[loc].map(self._ticker_changes)
        return clean_tickers

    def _populate_BAML_sectors(self, df):
        """
        Use stored BAML maps to populate missing
        BAML sectors and overwrite erronious ones.

        When applying this fix, First use the ticker map to fill in the
        most frequent sector for each ticker, then use the issuer map to
        add a more granular mapping for tickers with multiple sectors.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with necessary columns.
        """
        for col in ["BAMLSector", "BAMLTopLevelSector"]:
            for id in ["Ticker", "Issuer"]:
                map = getattr(self, f"_{col}_{id}_map")
                loc = df[id].isin(map)
                df.loc[loc, col] = df.loc[loc, id].map(map)

    def _drop_duplicate_columns(self, df, keep_col, drop_col):
        if keep_col in df.columns and drop_col in df.columns:
            return df.drop(drop_col, axis=1)
        else:
            return df

    def _preprocess_basys_data(self, df):
        # Fill missing sectors and subsectors.
        sector_cols = [f"Level {i}" for i in range(7)]
        df[sector_cols] = (
            df[sector_cols]
            .replace("*", np.nan)
            .fillna(method="ffill", axis=1)
            .copy()
        )

        # Fill missing collateral types.
        collat_cols = [f"Seniority Level {i+1}" for i in range(3)]
        df[collat_cols] = (
            df[collat_cols]
            .replace("*", np.nan)
            .fillna(method="ffill", axis=1)
            .copy()
        )

        # Drop potential duplicate columns. These columns changed
        # name over time and for a short period both were present.
        # df = self._drop_duplicate_columns(
        #     df, "Z-Spread Over", "Z-Spread Over RFR"
        # )
        # df = self._drop_duplicate_columns(
        #     df,
        #     "Month-to-Date Excess Return over RFR",
        #     "Month-to-date Libor Swap Return",
        # )

        col_map = {
            "Date": "Date",
            "Ticker": "Ticker",
            "Issuer": "Issuer",
            "ISIN": "ISIN",
            "CUSIP": "CUSIP",
            "Coupon": "CouponRate",
            "Final Maturity": "MaturityDate",
            "Workout date": "WorkoutDate",
            "Level 1": "SectorLevel1",
            "Level 2": "SectorLevel2",
            "Level 3": "SectorLevel3",
            "Level 4": "SectorLevel4",
            "Level 5": "SectorLevel5",
            "Level 6": "SectorLevel6",
            "Markit iBoxx Rating": "CompositeRating",
            "Seniority Level 2": "CollateralType",
            "Notional Amount": "AmountOutstanding",
            "Market Value": "MarketValue",
            "Next Call Date": "NextCallDate",
            "Index Price": "CleanPrice",
            "Dirty Index Price": "DirtyPrice",
            "Accrued Interest": "AccruedInterest",
            "Effective OA duration": "OAD",
            "OAS": "OAS",
            "Z-Spread Over Libor": "ZSpread",
            "Z-Spread Over RFR": "ZSpread",
            "Semi-Annual Yield": "YieldToWorst",
            "Semi-Annual Yield to Maturity": "YieldToMat",
            "Semi-Annual Modified Duration": "ModDurationToWorst",
            "Semi-Annual Modified Duration to Maturity": "ModDurationToMat",
            "Month-to-date Sovereign Curve Swap Return": "MTDXSRet",
            "Month-to-Date Sovereign Curve Swap Return": "MTDXSRet",
            "Month-to-Date Libor Swap Return": "MTDLiborXSRet",
            "Month-to-date Libor Swap Return": "MTDLiborXSRet",
            "Month-to-Date Excess Return over RFR": "MTDLiborXSRet",
            "Month-to-date Excess Return over RFR": "MTDLiborXSRet",
        }

        # Get sorted unique columns.
        unique_cols = set()
        cols = []
        for col in col_map.values():
            if col in unique_cols:
                continue
            else:
                cols.append(col)
                unique_cols.add(col)

        df = df.rename(columns=col_map)[cols].copy()

        # Convert str time to datetime.
        for date_col in ["Date", "MaturityDate", "WorkoutDate", "NextCallDate"]:
            dates = pd.to_datetime(
                df[date_col], format="%Y-%m-%d", errors="coerce"
            )
            # Find indexes with NaT dates, and use other format.
            mask = dates.isna()
            clean_dates = pd.to_datetime(
                df.loc[mask, date_col], format="%d/%m/%Y", errors="coerce"
            )
            dates.loc[mask] = clean_dates
            mask = dates.isna()
            clean_dates = pd.to_datetime(
                df.loc[mask, date_col], errors="coerce"
            )
            dates.loc[mask] = clean_dates
            df[date_col] = dates

        # Set perpetuals to a constant date of 1/1/2220.
        df.loc[df["MaturityDate"].isna(), "MaturityDate"] = pd.to_datetime(
            "1/1/2220"
        )

        # Define maturites year.
        day = "timedelta64[D]"
        df["MaturityYears"] = (df["MaturityDate"] - df["Date"]).astype(
            day
        ) / 365

        # Issuers and clean raw sector columns.
        df["Issuer"] = df["Issuer"].str.upper()
        sector_cols = [f"SectorLevel{i}" for i in range(1, 7)]
        for col in sector_cols:
            df[col] = self._clean_sectors(df[col])

        # Put amount outstanding and market value into $M.
        for col in ["AmountOutstanding", "MarketValue"]:
            df[col] /= 1e6

        # Get numeric ratings from composit ratings.
        df["NumericRating"] = self._get_numeric_ratings(df, ["CompositeRating"])

        # Calculate DTS. Approximate OAD ~ OASD.
        df["DTS"] = df["OAD"] * df["OAS"]
        df["ZSpread"]
        df["DTS_Libor"] = df["OAD"] * df["ZSpread"]

        df["CUSIP"].fillna(df["ISIN"], inplace=True)
        df["Sector"] = df["SectorLevel5"].values
        df.loc[df["SectorLevel1"] == "SOVEREIGNS", "Sector"] = "SOVEREIGNS"
        df["Subsector"] = np.nan
        # Add financial flag column.
        df["FinancialFlag"] = convert_sectors_to_fin_flags(df["Sector"])
        return clean_dtypes(df)

    def _get_numeric_ratings(self, df, cols):
        """
        Get numeric ratings using the `ratings.json` conversion
        file to convert all rating agencies to numeric values,
        then apply middle-or-lower methodology to get single
        rating.

        Parameters
        ----------
        df: pd.DataFrame
            Raw DataFrame from SQL querry.

        Notes
        -----
        If unknown value is encountered, user will
        be prompted to provide the numeric value, and the
        `ratings.json` file will be updated.

        Returns
        -------
        MOL: [N,] ndarray
            Middle or lower
        """

        # Make temporary matrix of numeric ratings.
        ratings_mat = np.zeros((len(df), len(cols)), dtype="object")
        for i, col in enumerate(cols):
            try:
                agency_col = df[col].cat.add_categories("NR")
            except (AttributeError, ValueError):
                agency_col = df[col]

            ratings_mat[:, i] = agency_col.fillna("NR")

        # Fill matrix with numeric ratings for each rating agency.
        while True:
            try:
                num_ratings = np.vectorize(self._ratings.__getitem__)(
                    ratings_mat
                ).astype(float)
            except KeyError as e:
                # Ask user to provide value for missing key.
                key = e.args[0]
                val = int(
                    input(
                        (
                            f"KeyError: '{key}' is not in `ratings.json`.\n"
                            "Please provide the appropriate numeric value:\n"
                        )
                    )
                )
                # Update `ratings.json` with user provided key:val pair.
                self._ratings[key] = val
                dump_json(self._ratings, "ratings")
            else:
                break

        num_ratings[num_ratings == 0] = np.nan  # json nan value is 0

        # Vectorized implementation of middle-or-lower.
        warnings.simplefilter(action="ignore", category=RuntimeWarning)
        nans = np.sum(np.isnan(num_ratings), axis=1)
        MOL = np.median(num_ratings, axis=1)  # middle
        lower = np.nanmax(num_ratings, axis=1)  # max for lower
        lower_mask = (nans == 1) | (nans == 2)
        MOL[lower_mask] = lower[lower_mask]
        warnings.simplefilter(action="default", category=RuntimeWarning)
        return MOL

    def _clean_market_data(self, df):
        """
        Clean DataFrame from SQL querry.

        Parameters
        ----------
        df: pd.DataFrame
            Raw DataFrame from SQL querry.

        Returns
        -------
        df: pd.DataFrame
            Cleaned DataFrame.
        """
        # List rules for bonds to keep and drop.
        allowed_coupon_types = {
            "FIXED",
            "VARIABLE",
            "STEP CPN",
            "FIXED PIK",
            "ZERO COUPON",
            "ADJUSTABLE",
            "HYBRID VARIABLE",
            "DEFAULTED",
        }
        allowed_call_types = {
            "NONCALL",
            "MKWHOLE",
            "CALL/NR",
            "CALL/RF",
            "EUROCAL",
            np.nan,
        }
        disallowed_sectors = {
            "NON_AGENCY_CMBS",
            "AGENCY_CMBS",
            "ABS_OTHER",
            "STRANDED_UTILITY",
            "GOVERNMENT_SPONSORED",
            "CAR_LOAN",
            "CREDIT_CARD",
            "MTG_NON_PFANDBRIEFE",
            "PFANDBRIEFE_TRADITIONAL_HYPOTHEKEN",
            "PFANDBRIEFE_TRADITIONAL_OEFFENLICHE",
            "PFANDBRIEFE_JUMBO_HYPOTHEKEN",
            "PFANDBRIEFE_JUMBO_OEFFENLICHE",
            "PSLOAN_NON_PFANDBRIEFE",
            "CONVENTIONAL_15_YR",
            "CONVENTIONAL_20_YR",
            "CONVENTIONAL_30_YR",
            "5/1_ARM",
            "7/1_ARM",
            "3/1_ARM",
            "GNMA_15_YR",
            "GNMA_30_YR",
            "NA",
        }
        disallowed_tickers = {"TVA", "FNMA", "FHLMC", "FN", "FR"}
        gcc_tickers = {
            "KSA",
            "QATAR",
            "ARAMCO",
            "QPETRO",
            "BHRAIN",
            "DUGB",
            "ADGB",
            "QUDIB",
            "OMAN",
            "KUWIB",
        }
        disallowed_collateral_types = {
            "CERT OF DEPOSIT",
            "GOVT LIQUID GTD",
            "INSURED",
            "FNCL",
            "FNCN",
            "FNHLCR",
        }

        # Drop rows with the required columns.
        required_cols = [
            "Date",
            "CUSIP",
            "Ticker",
            "CollateralType",
            "CouponRate",
        ]
        df.dropna(subset=required_cols, how="any", inplace=True)

        # Get middle-or-lower numeric rating for each cusip.
        rating_cols = ["MoodyRating", "SPRating", "FitchRating"]
        df["NumericRating"] = self._get_numeric_ratings(df, rating_cols)

        # Set perpetuals to a constant maturity date of 1/1/2220.
        df.loc[df["MaturityDate"].isna(), "MaturityDate"] = pd.to_datetime(
            "1/1/2220"
        )

        # Replace maturities on bonds with variable coupon type and
        # maturity less call below 1 yr (e.g., 11NC10).
        # TODO: Decide if this is best way to deal with these bonds.
        variable_bonds = df["CouponType"] == "VARIABLE"
        day = "timedelta64[D]"
        df.loc[
            variable_bonds
            & ((df["MaturityDate"] - df["NextCallDate"]).astype(day) <= 370),
            "MaturityDate",
        ] = df["NextCallDate"]

        # Define time to maturity and since issuance in years.
        df["MaturityYears"] = (df["MaturityDate"] - df["Date"]).astype(
            day
        ) / 365
        # For variable coupon bonds, make this measure until the next
        # call date, which is the date when the coupon type changes.
        # For fixed variable to float bonds, this will be the relevant
        # date for determining index eligibility.
        df.loc[variable_bonds, "MaturityYears"] = (
            df.loc[variable_bonds, "NextCallDate"]
            - df.loc[variable_bonds, "Date"]
        ).astype(day) / 365
        df["IssueYears"] = (df["Date"] - df["IssueDate"]).astype(day) / 365

        # Add HY Index Flags
        df = self._add_hy_index_flags(df)

        # Find bonds that are in any of the tracked indexes.
        ignored_flag_cols = {
            "FinancialFlag",
            "Eligibility144AFlag",
            "AnyIndexFlag",
            "MLHYFlag",
            "USAggReturnsFlag",
            "USAggStatisticsFlag",
        }
        index_flag_cols = [
            col
            for col in df.columns
            if "flag" in col.lower() and col not in ignored_flag_cols
        ]
        df[index_flag_cols] = df[index_flag_cols].replace(-1, 0)
        df[index_flag_cols]
        is_index_bond = df[index_flag_cols].sum(axis=1).astype(bool)

        # Find columns to keep by ignoring unnecessary columns.
        drop_cols = {
            "Delta",
            "Gamma",
            "Rho",
            "Theta",
            "Vega",
            "OAS_1W",
            "OAS_1M",
            "OAS_3M",
            "OAS_6M",
            "OAS_12M",
            "LiquidityCostScore",
            "LQA",
            "MLFI_Classification1",
            "MLFI_Classification2",
        }
        keep_cols = [c for c in df.columns if c not in drop_cols]

        # Subset DataFrame by specified rules.
        df = df[
            is_index_bond
            | (
                (df["CouponType"].isin(allowed_coupon_types))
                & (df["CallType"].isin(allowed_call_types))
                & (~df["Sector"].isin(disallowed_sectors))
                & (~df["Ticker"].isin(disallowed_tickers))
                & (~df["CollateralType"].isin(disallowed_collateral_types))
                & (
                    df["CountryOfRisk"].isin(self.DM_countries)
                    | df["Ticker"].isin(gcc_tickers)
                )
                & (df["Currency"] == "USD")
                & (df["NumericRating"] <= 22)
            )
        ][keep_cols].copy()

        # Convert outstading from $ to $M.
        df["AmountOutstanding"] /= 1e6

        # Determine original maturity as int.
        # TODO: Think about modifying for a Long 10 which should be 10.5.
        df["OriginalMaturity"] = np.nan_to_num(
            np.round(df["MaturityYears"] + df["IssueYears"])
        ).astype("int8")

        # Calculate dirty price.
        df["DirtyPrice"] = df["CleanPrice"] + df["AccruedInterest"]

        # Use dirty price to calculate market value.
        df["MarketValue"] = df["AmountOutstanding"] * df["DirtyPrice"] / 100

        # Add utility business structure.
        df = self._update_utility_business_structure(df)

        # Make new fields categories and drop duplicates.
        return clean_dtypes(df).drop_duplicates(subset=["CUSIP", "Date"])

    def _update_utility_business_structure(self, df):
        # Map utility subsector to proper business structure
        # from issuer name.
        utility_map = self._utility_business_structure
        utility_sectors = {"ELECTRIC", "NATURAL_GAS", "UTILITY_OTHER"}
        utes = df["Sector"].isin(utility_sectors)
        df["Subsector"] = df["Subsector"].astype(str)
        df.loc[utes, "Subsector"] = df.loc[utes, "Issuer"].map(utility_map)

        # Alert user to missing utilities
        loaded_issuers = set(df.loc[utes]["Issuer"].unique())
        missing_issuers = loaded_issuers - set(utility_map.keys())
        cols = ["Issuer", "Ticker"]
        missing_issuers_df = df[df["Issuer"].isin(missing_issuers)].dropna(
            subset=cols
        )
        if len(missing_issuers_df):
            missing_utes = groupby(missing_issuers_df[cols], "Issuer")["Ticker"]
            print("\nMissing Utility Business Structures:")
            for issuer, ticker in missing_utes.items():
                print(f"    {ticker: <8}: {issuer}")
            print()
        return df

    def _load_local_data(self):
        """Load data from the ``lgimapy/data`` directory."""
        # Find correct dates for current load.
        last_trade_date = self.trade_dates(market=self._market)[-1]
        start = last_trade_date if self._start is None else self._start
        end = last_trade_date if self._end is None else self._end
        start_month = pd.to_datetime(f"{start.month}/1/{start.year}")

        # Load feather files as DataFrames and subset to correct dates.
        fmt = f"%Y_%m.{self._fid_extension}"
        files = pd.date_range(start_month, end, freq="MS").strftime(fmt)
        fid_dir = self.local(f"{self._market}/{self._fid_extension}s")
        fids = [fid_dir / file for file in files]
        read_func = {
            "feather": pd.read_feather,
            "parquet": pd.read_parquet,
        }[self._fid_extension]
        df = concat_index_dfs([clean_dtypes(read_func(f)) for f in fids])
        df = df[(df["Date"] >= start) & (df["Date"] <= end)]
        return df

    def _load_SQL_market_data(self):
        """Load US market data using SQL stored procedure."""
        # Format input dates for SQL query.
        dt_fmt = "%m/%d/%Y"
        start = None if self._start is None else self._start.strftime(dt_fmt)
        end = None if self._end is None else self._end.strftime(dt_fmt)

        # Perform SQL query.
        yesterday = self.trade_dates(market=self._market)[-1].strftime(dt_fmt)
        sql = "exec [LGIMADatamart].[dbo].[sp_AFI_Get_SecurityAnalytics] {}"
        if start is None and end is None:
            query = sql.format(f"'{yesterday}', '{yesterday}'")
        elif start is not None and end is None:
            query = sql.format(f"'{start}', '{yesterday}'")
        else:
            query = sql.format(f"'{start}', '{end}'")
        # Add specific cusips if required.
        if self._cusips is not None:
            cusip_str = f", '{','.join(self._cusips)}'"
            query += cusip_str

        # Load data into a DataFrame form SQL in chunks to
        # reduce memory usage, cleaning them on the fly.
        df_chunk = self.query_datamart(query, chunksize=50_000)
        chunk_list = []
        for chunk in df_chunk:
            # Preprocess chunk.
            if self._preprocess:
                chunk = self._preprocess_market_data(chunk)
            if self._clean:
                chunk = self._clean_market_data(chunk)
            chunk_list.append(chunk)
        if self._clean:
            try:
                df = concat_index_dfs(chunk_list)
            except (IndexError, ValueError):
                raise ValueError("Empty Index, try selecting different dates.")
        else:
            try:
                df = pd.concat(chunk_list, join="outer", sort=False)
            except (IndexError, ValueError):
                raise ValueError("Empty Index, try selecting different dates.")

        # Save unique loaded dates to memory.
        return df

    def _load_BASys_data(self):
        trade_dates = self._trade_date_df(self._market)
        trade_dates = trade_dates[trade_dates["holiday"] == 0]
        if self._start is not None:
            trade_dates = trade_dates[trade_dates.index >= self._start]
        if self._end is not None:
            trade_dates = trade_dates[trade_dates.index <= self._end]
        if self._start is None and self._end is None:
            trade_dates = trade_dates.iloc[-1].to_frame().T
        df_list = []
        for fid in trade_dates["fid"]:
            df_raw = pd.read_csv(
                Path(fid),
                # engine="python",
                encoding="ISO-8859-1",
                dtype={
                    "Next Call Date": str,
                    "Date": str,
                    "Workout date": str,
                    "Final Maturity": str,
                    "Level 1": str,
                    "Level 2": str,
                    "Level 3": str,
                    "Level 4": str,
                    "Level 5": str,
                    "Level 6": str,
                    "ISIN": str,
                    "CUSIP": str,
                    "Ticker": str,
                },
            )
            if self._clean:
                df_list.append(self._preprocess_basys_data(df_raw))
            else:
                df_list.append(df_raw)

        return pd.concat(df_list).sort_values(["Date", "Ticker"])

    def has_dates_loaded(self, dates):
        loaded_dates_set = (
            set() if self.loaded_dates is None else set(self.loaded_dates)
        )
        return loaded_dates_set >= to_set(dates, dtype=pd.Timestamp)

    def load_market_data(
        self,
        date=None,
        start=None,
        end=None,
        cusips=None,
        clean=True,
        preprocess=True,
        ret_df=False,
        local=True,
        local_file_fmt="feather",
        market=None,
        data=None,
    ):
        """
        Load market data from SQL server. If end is not specified
        data is scraped through previous day. If neither start nor
        end are given only the data from previous day is scraped.
        Optionally load from local compressed format for increased
        performance or feed a DataFrame directly.

        Notes
        -----
        If unknown value is encountered in ratings, user will
        be prompted to provide the numeric value, and the
        `ratings.json` file will be updated.

        Parameters
        ----------
        date: datetime, optional
            Single date to scrape.
        start: datetime, optional
            Starting date for scrape.
        end: datetime, optional
            Ending date for scrape.
        cusips: List[str], optional
            List of cusips to specify for the load, by default load all.
        clean: bool, default=True
            If ``True``, apply standard cleaning rules to loaded data.
        preprocess: bool, default=True
            If ``True``, preprocess loaded data.
        ret_df: bool, default=False
            If ``True``, return loaded DataFrame.
        local: bool, default=True
            Load index from local feather file.
        local_file_fmt: ``{"feather", "parquet"}``, default="feather"
            File extension for local files.
        market: ``{"US", "EUR", "GBP"}``, optional
            Market to get trade dates for.
            Defaults to :attr:`Database.market`.
        data: pd.DataFrame, optional
            Pre-loaded data to keep in :class:`Database` memory as
            :attr:`df`.

        Returns
        -------
        df: pd.DataFrame
            DataFrame for specified date's index data if `ret_df` is true.
        """
        self._start = to_datetime(start)
        self._end = to_datetime(end)
        if date is not None:
            self._start = self._end = to_datetime(date)
        self._market = self.market if market is None else check_market(market)
        self._fid_extension = local_file_fmt
        self._cusips = cusips
        self._clean = clean
        self._preprocess = preprocess

        if data is not None:
            # Store provided data.
            self.df = data.copy()
        elif local:
            # Load data from local files.
            self.df = self._load_local_data()
        else:
            if self._market == "US":
                # Load US market data from SQL.
                self.df = self._load_SQL_market_data()
            else:
                # Load EUR/GBP market data from BASys directory.
                self.df = self._load_BASys_data()

        if ret_df:
            return self.df

    @property
    def loaded_dates(self):
        """List[datetime]: List of currently loaded dates."""
        try:
            return list(pd.to_datetime(self.df["Date"].unique()))
        except AttributeError:
            return None

    def _add_category_input(self, input_val, col_name):
        """
        Add inputs from :meth:`Database.build_market_index` function
        with type of either str or List[str] to hash table to
        use when subsetting full DataFrame.

        Parameters
        ----------
        input_val: str, List[str].
            Input variable from :meth:`Database.build_market_index`.
        col_nam: str
            Column name in full DataFrame.
        """
        if input_val is not None:
            self._all_rules.append(col_name)
            self._category_vals[col_name] = set(input_val)

    def _add_range_input(self, input_val, col_name):
        """
        Add inputs from:meth:`Database.build_market_index` function with
        type tuple of ranged float values to hash table to use
        when subsetting full DataFrame. `-np.infty` and `np.infty`
        can be used to drop rows with NaNs.

        Parameters
        ----------
        input_val: Tuple(float, float).
            Input variable from :meth:`Database.build_market_index`.
        col_nam: str
            Column name in full DataFrame.
        """
        i0, i1 = input_val[0], input_val[1]
        if i0 is not None or i1 is not None:
            self._all_rules.append(col_name)
            if col_name == "Date":
                self._range_vals[col_name] = (
                    self.df["Date"].iloc[0] if i0 is None else i0,
                    self.df["Date"].iloc[-1] if i1 is None else i1,
                )
            else:
                self._range_vals[col_name] = (
                    -np.infty if i0 is None else i0,
                    np.infty if i1 is None else i1,
                )

    def _add_flag_input(self, input_val, col_name):
        """
        Add inputs from :meth:`Database.build_market_index` function
        with bool or int type to hash table to use when subsetting
        full DataFrame.

        Parameters
        ----------
        input_val: bool.
            Input variable from :meth:`Database.build_market_index`.
        col_nam: str
            Column name in full DataFrame.
        """
        if input_val is not None:
            self._all_rules.append(col_name)
            self._flags[col_name] = int(input_val)

    def build_market_index(
        self,
        name="",
        date=None,
        start=None,
        end=None,
        rating=(None, None),
        analyst_rating=(None, None),
        currency=None,
        cusip=None,
        isin=None,
        issuer=None,
        ticker=None,
        sector=None,
        subsector=None,
        BAML_top_level_sector=None,
        BAML_sector=None,
        drop_treasuries=True,
        drop_municipals=False,
        maturity=(None, None),
        issue_years=(None, None),
        original_maturity=(None, None),
        clean_price=(None, None),
        dirty_price=(None, None),
        coupon_rate=(None, None),
        coupon_type=None,
        market_of_issue=None,
        country_of_domicile=None,
        country_of_risk=None,
        amount_outstanding=(None, None),
        market_value=(None, None),
        collateral_type=None,
        yield_to_worst=(None, None),
        OAD=(None, None),
        OAS=(None, None),
        OASD=(None, None),
        DTS=(None, None),
        mod_dur_to_worst=(None, None),
        mod_dur_to_mat=(None, None),
        liquidity_score=(None, None),
        in_stats_index=None,
        in_returns_index=None,
        in_agg_stats_index=None,
        in_agg_returns_index=None,
        in_hy_stats_index=None,
        in_hy_returns_index=None,
        in_any_index=None,
        in_H4UN_index=None,
        in_H0A0_index=None,
        in_HC1N_index=None,
        in_HUC2_index=None,
        in_HUC3_index=None,
        is_144A=None,
        financial_flag=None,
        is_new_issue=None,
        special_rules=None,
    ):
        """
        Build index with customized rules from :attr:`Database.df`.

        Parameters
        ----------
        name: str, default=''
            Optional name for returned index.
        date: datetime, optional
            Single date to build index.
        start: datetime, optional
            Start date for index, if None the start date from load is used.
        end: datetime, optional
            End date for index, if None the end date from load is used.
        rating: str , Tuple[str, str], optional
            Bond rating/rating range for index.

            Examples:

            * str: ``'HY'``, ``'IG'``, ``'AAA'``, ``'Aa1'``, etc.
            * Tuple[str, str]: ``('AAA', 'BB')`` uses all bonds in
              specified inclusive range.
        analyst_rating: Tuple[float, float], default=(None, None)
            Range of analyst ratings to include.
        currency: str, List[str], optional
            Currency or list of currencies to include.
        cusip: str, List[str]: optional
            CUSIP or list of CUSIPs to include.
        isin: str, List[str]: optional
            ISIN or list of ISINs to include.
        issuer: str, List[str], optional
            Issuer, or list of issuers to include.
        ticker: str, List[str], optional
            Ticker or list of tickers to include in index, default is all.
        sector: str, List[str], optional
            Sector or list of sectors to include in index.
        subsector: str, List[str], optional
            Subsector or list of subsectors to include in index.
        BAML_top_level_sector: str, List[str], optional
            BAML top level sector or list of sectors to include in index.
        BAML_sector: str, List[str], optional
            BAML sector or list of sectors to include in index.
        drop_treasuries: bool, default=True
            Whether to drop treausuries.
        drop_municipals: bool, default=False
            Whether to drop municipals.
        maturity: Tuple[float, float], {5, 10, 20, 30}, optional
            Maturities to include, if int is specified the following ranges
            are used:

            * 5: 4-6
            * 10: 6-11
            * 20: 11-25
            * 30: 25 - 31
        original_maturity: Tuple[float, float], default=(None, None)
            Range of original bond maturities to include.
        clean_price: Tuple[float, float]), default=(None, None)
            Clean price range of bonds to include, default is all.
        dirty_price: Tuple[float, float]), default=(None, None)
            Dirty price range of bonds to include, default is all.
        coupon_rate: Tuple[float, float]), default=(None, None)
            Coupon rate range of bonds to include, default is all.
        coupon_type: str or List[str], optional
            Coupon types ``{'FIXED', 'ZERO COUPON', 'STEP CPN', etc.)``
            to include in index, default is all.
        market_of_issue: str, List[str], optional
            Markets of issue to include in index, defautl is all.
        country_of_domicile: str, List[str], optional
            Country or list of countries of domicile to include
            in index, default is all.
        country_of_risk: str, List[str], optional
            Country or list of countries wherer risk is centered
            to include in index, default is all.
        amount_outstanding: Tuple[float, float], default=(None, None)
            Range of amount outstanding to include in index (Millions).
        market_value: Tuple[float, float], default=(None, None)
            Range of market values to include in index (Millions).
        issue_years: Tuple[float, float], default=(None, None)
            Range of years since issue to include in index,
            default is all.
        collateral_type: str, List[str], optional
            Collateral type or list of types to include,
            default is all.
        yield_to_worst: Tuple[float, float], default=(None, None)
            Range of yields (to worst) to include, default is all.
        OAD: Tuple[float, float], default=(None, None)
            Range of option adjusted durations to include,
            default is all.
        OAS: Tuple[float, float], default=(None, None)
            Range of option adjusted spreads to include,
            default is all.
        OASD:  Tuple[float, float], default=(None, None)
            Range of option adjusted spread durations,
            default is all.
        DTS: Tuple[float, float], default=(None, None)
            Range of DTS to include, default is all.
        mod_dur_to_worst: Tuple[float, float], default=(None, None)
            Range of modified durations to worst date,
            default is all.
        mod_dur_to_mat: Tuple[float, float], default=(None, None)
            Range of modified durations to maturity date,
            default is all.
        liquidity_score: Tuple[float, float], default=(None, None)
            Range of liquidty scores to use, default is all.
        in_stats_index: bool, optional
            If True, only include bonds in stats index.
            If False, only include bonds out of stats index.
            By defualt include both.
        in_returns_index: bool, optional
            If True, only include bonds in returns index.
            If False, only include bonds out of returns index.
            By defualt include both.
        in_agg_stats_index: bool, optional
            If True, only include bonds in aggregate stats index.
            If False, only include bonds out of aggregate stats index.
            By defualt include both.
        in_agg_returns_index: bool, optional
            If True, only include bonds in aggregate returns index.
            If False, only include bonds out of aggregate returns index.
            By defualt include both.
        in_hy_stats_index: bool, optional
            If True, only include bonds in HY stats index.
            If False, only include bonds out of HY stats index.
            By defualt include both.
        in_hy_returns_index: bool, optional
            If True, only include bonds in HY returns index.
            If False, only include bonds out of HY returns index.
            By defualt include both.
        in_any_index: bool, optional
            If ``True``, only include bonds in any Bloomberg index.
            If ``False``, only include bonds not in any Bloomberg index.
            By defualt include both.
        in_H4UN_index: bool, optional
            If ``True``, only include bonds in iBoxx H4UN index.
            If ``False``, only include bonds not in iBoxx H4UN index.
            By defualt include both.
        in_H0A0_index: bool, optional
            If ``True``, only include bonds in iBoxx H0A0 index.
            If ``False``, only include bonds not in iBoxx H0A0 index.
            By defualt include both.
        in_HC1N_index: bool, optional
            If ``True``, only include bonds in iBoxx HC1N index.
            If ``False``, only include bonds not in iBoxx HC1N index.
            By defualt include both.
        in_HUC2_index: bool, optional
            If ``True``, only include bonds in iBoxx HUC2 index.
            If ``False``, only include bonds not in iBoxx HUC2 index.
            By defualt include both.
        in_HUC3_index: bool, optional
            If ``True``, only include bonds in iBoxx HUC3 index.
            If ``False``, only include bonds not in iBoxx HUC3 index.
            By defualt include both.
        is_144A: bool, optional
            If True, only include 144A bonds.
            If False, only include non 144A bonds.
            By defualt include both.
        is_new_issue: bool, optional
            If True, only include bonds in the month they were issued.
            If False, include all bonds.
            By default include all bonds.
        financial_flag: bool or ``{0, 1, 2}``, optional
            Selection for including fins, non-fins, or other.

            * 0 or ``False``: Non-financial sectors.
            * 1 or ``True``: Financial sectors.
            * 2: Other (Treasuries, Sovs, Govt Ownwed, etc.).
        special_rules: str, List[str] optional
            Special rule(s) for subsetting index using bitwise
            operators. If None, all specified inputs are applied
            independtently of eachother as bitwise &. All rules
            can be stacked using paranthesis to create more
            complex rules.

            Examples:

            * Include specified sectors or subsectors:
              ``special_rules='Sector | Subsector'``
            * Include all but specified sectors:
              ``special_rules='~Sector'``
            * Include either (all but specified currency or specified
              sectors) xor specified maturities:
              ``special_rules='(~Currnecy | Sector) ^ MaturityYears'``

        Returns
        -------
        :class:`Index`:
            :class:`Index` with specified rules.
        """
        # Convert dates to datetime.
        if date is not None:
            start = end = date
        start = None if start is None else pd.to_datetime(start)
        end = None if end is None else pd.to_datetime(end)

        # Convert rating to range of inclusive ratings.
        if rating == (None, None):
            pass
        else:
            rating = self.convert_input_ratings(rating)

        # Convert all category constraints to lists.
        currency = to_list(currency, dtype=str, sort=True)
        ticker = to_list(ticker, dtype=str, sort=True)
        cusip = to_list(cusip, dtype=str, sort=True)
        isin = to_list(isin, dtype=str, sort=True)
        issuer = to_list(issuer, dtype=str, sort=True)
        market_of_issue = to_list(market_of_issue, dtype=str, sort=True)
        country_of_domicile = to_list(country_of_domicile, dtype=str, sort=True)
        country_of_risk = to_list(country_of_risk, dtype=str, sort=True)
        collateral_type = to_list(collateral_type, dtype=str, sort=True)
        coupon_type = to_list(coupon_type, dtype=str, sort=True)
        sector = to_list(sector, dtype=str, sort=True)
        subsector = to_list(subsector, dtype=str, sort=True)
        BAML_sector = to_list(BAML_sector, dtype=str, sort=True)
        BAML_top_level_sector = to_list(
            BAML_top_level_sector, dtype=str, sort=True
        )

        # Convert all flag constraints to int.
        in_returns_index = to_int(in_returns_index)
        in_stats_index = to_int(in_stats_index)
        in_agg_returns_index = to_int(in_agg_returns_index)
        in_agg_stats_index = to_int(in_agg_stats_index)
        in_hy_stats_index = to_int(in_hy_stats_index)
        in_hy_returns_index = to_int(in_hy_returns_index)
        in_any_index = to_int(in_any_index)
        in_H4UN_index = to_int(in_H4UN_index)
        in_H0A0_index = to_int(in_H0A0_index)
        in_HC1N_index = to_int(in_HC1N_index)
        in_HUC2_index = to_int(in_HUC2_index)
        in_HUC3_index = to_int(in_HUC3_index)
        is_144A = to_int(is_144A)
        financial_flag = to_int(financial_flag)
        is_new_issue = to_int(is_new_issue)

        # Save parameter constraints used to build index.
        argspec = getfullargspec(self.build_market_index)
        default_constraints = {
            arg: default
            for arg, default in zip(argspec.args[1:], argspec.defaults)
        }
        user_defined_constraints = locals().copy()
        ignored_kws = {
            "self",
            "argspec",
            "default_constraints",
            "start",
            "end",
            "date",
        }
        index_constraints = {
            kwarg: val
            for kwarg, val in user_defined_constraints.items()
            if kwarg not in ignored_kws and val != default_constraints[kwarg]
        }
        if drop_treasuries:
            index_constraints["drop_treasuries"] = True

        # Add new issue mask if required.
        if is_new_issue:
            self.df["NewIssueMask"] = new_issue_mask(self.df)

        # TODO: Modify price/amount outstading s.t. they account for currency.
        # Store category constraints.
        self._all_rules = []
        self._category_vals = {}
        category_constraints = {
            "currency": ("Currency", currency),
            "cusip": ("CUSIP", cusip),
            "isin": ("ISIN", isin),
            "issuer": ("Issuer", issuer),
            "ticker": ("Ticker", ticker),
            "sector": ("Sector", sector),
            "subsector": ("Subsector", subsector),
            "BAML_sector": ("BAMLSector", BAML_sector),
            "BAML_top_level_sector": (
                "BAMLTopLevelSector",
                BAML_top_level_sector,
            ),
            "market_of_issue": ("MarketOfIssue", market_of_issue),
            "country_of_domicile": ("CountryOfDomicile", country_of_domicile),
            "country_of_risk": ("CountryOfRisk", country_of_risk),
            "collateral_type": ("CollateralType", collateral_type),
            "coupon_type": ("CouponType", coupon_type),
        }
        for col, constraint in category_constraints.values():
            self._add_category_input(constraint, col)

        # Store flag constraints.
        self._flags = {}
        flag_constraints = {
            "in_returns_index": ("USCreditReturnsFlag", in_returns_index),
            "in_stats_index": ("USCreditStatisticsFlag", in_stats_index),
            "in_agg_returns_index": ("USAggReturnsFlag", in_agg_returns_index),
            "in_agg_stats_index": ("USAggStatisticsFlag", in_agg_stats_index),
            "in_hy_stats_index": ("USHYStatisticsFlag", in_hy_stats_index),
            "in_hy_returns_index": ("USHYReturnsFlag", in_hy_returns_index),
            "in_any_index": ("AnyIndexFlag", in_any_index),
            "in_H4UN_index": ("H4UNFlag", in_H4UN_index),
            "in_H0A0_index": ("H0A0Flag", in_H0A0_index),
            "in_HC1N_index": ("HC1NFlag", in_HC1N_index),
            "in_HUC2_index": ("HUC2Flag", in_HUC2_index),
            "in_HUC3_index": ("HUC3Flag", in_HUC3_index),
            "is_144A": ("Eligibility144AFlag", is_144A),
            "financial_flag": ("FinancialFlag", financial_flag),
            "is_new_issue": ("NewIssueMask", is_new_issue),
        }
        for col, constraint in flag_constraints.values():
            self._add_flag_input(constraint, col)

        # Store range constraints.
        range_constraints = {
            "date": ("Date", (start, end)),
            "original_maturity": ("OriginalMaturity", original_maturity),
            "maturity": ("MaturityYears", maturity),
            "issue_years": ("IssueYears", issue_years),
            "clean_price": ("CleanPrice", clean_price),
            "dirty_price": ("DirtyPrice", dirty_price),
            "coupon_rate": ("CouponRate", coupon_rate),
            "rating": ("NumericRating", rating),
            "analyst_rating": ("AnalystRating", analyst_rating),
            "amount_outstanding": ("AmountOutstanding", amount_outstanding),
            "market_value": ("MarketValue", market_value),
            "yield_to_worst": ("YieldToWorst", yield_to_worst),
            "OAD": ("OAD", OAD),
            "OAS": ("OAS", OAS),
            "OASD": ("OASD", OASD),
            "DTS": ("DTS", DTS),
            "mod_dur_to_worst": ("ModDurtoWorst", mod_dur_to_worst),
            "mod_dur_to_mat": ("ModDurtoMat", mod_dur_to_mat),
            "liquidity_score": ("LQA", liquidity_score),
        }
        self._range_vals = {}
        for col, constraint in range_constraints.values():
            self._add_range_input(constraint, col)

        # Identify columns with special rules.
        rule_cols = []
        if special_rules:
            if isinstance(special_rules, str):
                special_rules = [special_rules]  # make list
            # Add space around operators.
            repl = {op: f" {op} " for op in "()~&|"}
            for rule in special_rules:
                rule_str = replace_multiple(rule, repl)
                rule_cols.extend(rule_str.split())
            rule_cols = [rc for rc in rule_cols if rc not in "()~&|"]

        # Build evaluation replacement strings.
        # All binary masks are created individually as strings
        # and joined together using bitwise & to be applied to
        # :attr:`Database.df` simulatenously in order to
        # avoid re-writing the DataFrame into memory after
        # each individual mask.
        range_repl = {
            key: (
                f'(self.df["{key}"] >= self._range_vals["{key}"][0]) & '
                f'(self.df["{key}"] <= self._range_vals["{key}"][1])'
            )
            for key in self._range_vals.keys()
        }
        cat_repl = {
            key: f'(self.df["{key}"].isin(self._category_vals["{key}"]))'
            for key in self._category_vals.keys()
        }
        flag_repl = {
            key: f'(self.df["{key}"] == self._flags["{key}"])'
            for key in self._flags
        }
        repl_dict = {**range_repl, **cat_repl, **flag_repl, "~(": "(~"}

        # Format special rules.
        subset_mask_list = []
        if special_rules:
            if isinstance(special_rules, str):
                special_rules = [special_rules]  # make list
            for rule in special_rules:
                subset_mask_list.append(
                    f"({replace_multiple(rule, repl_dict)})"
                )
        # Add treasury and muncipal rules.
        if drop_treasuries:
            subset_mask_list.append('(self.df["Sector"]!="TREASURIES")')
        if drop_municipals:
            subset_mask_list.append('(self.df["Sector"]!="LOCAL_AUTHORITIES")')

        # Format all other rules.
        for rule in self._all_rules:
            if rule in rule_cols:
                continue  # already added to subset mask
            # subset_mask_list.append(replace_multiple(rule, repl_dict))
            subset_mask_list.append(repl_dict[rule])

        # Combine formatting rules into single mask and subset DataFrame,
        # and drop temporary columns.
        temp_cols = ["NewIssueMask"]
        if subset_mask_list:
            subset_mask = " & ".join(subset_mask_list)
            df = eval(f"self.df.loc[{subset_mask}]").drop(
                temp_cols, axis=1, errors="ignore"
            )
        else:
            df = self.df.drop(temp_cols, axis=1, errors="ignore")

        return Index(
            df=df, name=name, constraints=index_constraints, market=self._market
        )

    def _preprocess_portfolio_data(self, df):
        """
        Convert dtypes for columns from SQL Portfolio DataFrame
        and correct column names.

        Parameters
        ----------
        df: pd.DataFrame
            Raw DataFrame from SQL query.

        Returns
        -------
        pd.DataFrame
            DataFrame with correct column dtypes and names.
        """
        # Drop bonds with no CUSIP.
        # These are placeholders for potential future
        # Money market accounts
        df = df[~df["CUSIP"].isna()].copy()

        # Clean up column names.
        col_map = {
            "Maturity": "MaturityDate",
            "Seniority": "CollateralType",
            "Coupon": "CouponRate",
            "Country": "CountryOfRisk",
            "Moody": "MoodyRating",
            "S&P": "SPRating",
            "Fitch": "FitchRating",
            "PMV": "P_Weight",
            "PMV INDEX": "BM_Weight",
            "PMV VAR": "Weight_Diff",
            "OADAgg": "P_OAD",
            "BmOADAgg": "BM_OAD",
            "OADVar": "OAD_Diff",
            "AssetValue": "P_MarketValue",
            "Quantity": "P_Notional",
            "L4": "Sector",
            "Price_Dirty": "DirtyPrice",
            "Price": "CleanPrice",
            "OASDAgg": "P_OASD",
            "BMOASDAgg": "BM_OASD",
            "OASDVar": "OASD_Diff",
            "OASAgg": "P_OAS",
            "BMOASAgg": "BM_OAS",
            "OASVar": "OAS_Diff",
            "YTWAgg": "P_YTW",
            "BMYTWAgg": "BM_YTW",
            "YTWVar": "YTW_Diff",
            "DTSAgg": "P_DTS",
            "BMDTSAgg": "BM_DTS",
            "DTSVar": "DTS_Diff",
            "MKT_VAL": "StrategyMarketValue",
            "RVRecommendation": "AnalystRating",
            "YTW": "YieldToWorst",
        }
        df.columns = [col_map.get(col, col) for col in df.columns]

        # Map collateral types to standard values.
        equiv_ranks = {
            "UNSECURED": ["BONDS", "SR UNSECURED", "NOTES", "COMPANY GUARNT"],
            "SECURED": ["SR SECURED"],
            "1ST MORTGAGE": ["1ST REF MORT", "GENL REF MORT"],
        }
        for key, val in equiv_ranks.items():
            df.loc[df["CollateralType"].isin(val), "CollateralType"] = key

        # Clean other columns.
        df["Issuer"] = df["Issuer"].str.upper()
        df["Sector"] = self._clean_sectors(df["Sector"])
        df["Ticker"] = self._clean_tickers(df["Ticker"])

        # Convert date columns to datetime dtype.
        for col in ["Date", "MaturityDate"]:
            df[col] = pd.to_datetime(
                df[col], format="%Y-%m-%d", errors="coerce"
            )

        # Add commonly used columns.
        df["NumericRating"] = df["Idx_Rtg"].map(self._ratings)
        df["FinancialFlag"] = convert_sectors_to_fin_flags(df["Sector"])
        df["Subsector"] = get_bloomberg_subsector(df["CUSIP"])
        day = "timedelta64[D]"
        df["MaturityYears"] = (df["MaturityDate"] - df["Date"]).astype(
            day
        ) / 365

        # Set perpetuals to a constant maturity date of 1/1/2220.
        df.loc[df["MaturityDate"].isna(), "MaturityDate"] = pd.to_datetime(
            "1/1/2220"
        )

        # Add utility business structure.
        df = self._update_utility_business_structure(df)

        # Clean analyst ratings.
        df["AnalystRating"] = df["AnalystRating"].replace("NR", np.nan)

        # Add DTS PCT columns.
        for col in ["P_DTS", "BM_DTS", "DTS_Diff"]:
            df[col.replace("DTS", "DTS_PCT")] = df[col] / df["BM_DTS"].sum()

        return clean_dtypes(df).drop_duplicates(
            subset=["Date", "CUSIP", "Account"]
        )

    def _add_EUR_sector_cols(self, df):
        # TODO
        df = df_prev.copy()
        df["Sector"] = np.nan
        df["Subsector"] = np.nan

        # Perform direct replacements of existing sectors.
        direct_replacements_d = {
            "Sector": {
                3: {
                    "FINANCIAL_SERVICES": "FINANCIAL_SERVICES",
                    "REAL_ESTATE": "REAL_ESTATE",
                    "UTILITIES": "UTILITIES",
                    "INDUSTRIALS": "INDUSTRIALS",
                    "TELECOMMUNICATIONS": "TELECOMMUNICATIONS",
                    "BASIC_MATERIALS": "BASIC_MATERIALS",
                    "CONSUMER_GOODS": "CONSUMER_GOODS",
                    "CONSUMER_SERVICES": "CONSUMER_SERVICES",
                    "HEALTH_CARE": "HEALTH_CARE",
                    "INDUSTRIALS": "INDUSTRIALS",
                    "OIL_AND_GAS": "OIL_AND_GAS",
                    "TECHNOLOGY": "TECHNOLOGY",
                    "TELECOMMUNICATIONS": "TELECOMMUNICATIONS",
                    "UTILITIES": "UTILITIES",
                },
                5: {
                    "PHARMACEUTICALS_AND_BIOTECHNOLOGY": "PHARMA",
                },
            },
            "Subsector": {
                4: {
                    "BASIC_RESOURCES": "BASIC_RESOURCES",
                    "CHEMICALS": "CHEMICALS",
                    "BANKS": "BANKS",
                    "INSURANCE": "INSURANCE",
                    "TRAVEL_AND_LEISURE": "TRAVEL_AND_LEISURE",
                    "RETAIL": "RETAIL",
                    "MEDIA": "MEDIA",
                },
                5: {
                    "HEALTH_CARE_EQUIPMENT_AND_SERVICES": "MEDTECH",
                    "TOBACCO": "TOBACCO",
                    "AEROSPACE_AND_DEFENSE": "AEROSPACE_AND_DEFENSE",
                    "CONSTRUCTION_AND_MATERIALS": "CONSTRUCTION_AND_MATERIALS",
                    "CONSTRUCTION_AND_MATERIALS": "CONSTRUCTION_AND_MATERIALS",
                    "INDUSTRIAL_TRANSPORTATION": "INDUSTRIAL_TRANSPORTATION",
                    "ELECTRICITY": "ELECTRICITY",
                },
                6: {
                    "INDUSTRIAL_AND_OFFICE_REITS": "INDUSTRIAL_AND_OFFICE_REITS",
                    "LOGISTICS_REITS": "LOGISTICS_REITS",
                    "RESIDENTIAL_REITS": "RESIDENTIAL_REITS",
                    "RETAIL_REITS": "RETAIL_REITS",
                    "SPECIALTY_REITS": "SPECIALTY_REITS",
                    "DIVERSIFIED_REITS": "DIVERSIFIED_REITS",
                    "REAL_ESTATE_HOLDING_AND_DEVELOPMENT": "REAL_ESTATE_HOLDING_AND_DEVELOPMENT",
                    "WATER": "WATER",
                    "GAS_DISTRIBUTION": "GAS_DISTRIBUTION",
                    "MULTIUTILITIES": "MULTIUTILITIES",
                },
            },
        }
        for col, col_replacement_d in direct_replacements_d.items():
            for level, replacement_d in col_replacement_d.items():
                for prev_sector, new_sector in replacement_d.items():
                    loc = df[f"SectorLevel{level}"] == prev_sector
                    df.loc[loc, col] = new_sector

        # Use custom rules to build sectors.
        for prev_sector in ["PERSONAL_GOODS", "HOUSEHOLD_GOODS"]:
            loc = df["SectorLevel5"] == prev_sector
            df.loc[loc, "Sector"] = "PERSONAL_AND_HOUSEHOLD_GOODS"

        loc = (df["Sector"] == "INDUSTRIALS") & df["Subsector"].isna()
        df.loc[loc, "Subsector"] = "GENERAL_INDUSTRIES"

        df["CollateralType"].value_counts()
        df_raw[df_raw["Seniority Level 2"] == "Other"][
            "Seniority Level 3"
        ].value_counts()

    def _numeric_ratings_columns(self, df):
        cols = ["MoodyRating", "SPRating", "FitchRating"]
        if "Idx_Rtg" in df.columns:
            cols.append("Idx_Rtg")
        ratings_mat = np.zeros((len(df), len(cols)), dtype="object")
        for i, col in enumerate(cols):
            try:
                agency_col = df[col].cat.add_categories("NR")
            except (AttributeError, ValueError):
                agency_col = df[col]

            ratings_mat[:, i] = agency_col.fillna("NR")
        num_ratings = np.vectorize(self._ratings.__getitem__)(
            ratings_mat
        ).astype(float)
        num_ratings[num_ratings == 0] = np.nan  # json nan value is 0
        return pd.DataFrame(
            num_ratings, index=df.index, columns=cols
        ).rename_axis(None)

    def _add_rating_risk_buckets(self, df):
        rating_cols = self._numeric_ratings_columns(df)
        min_rating = rating_cols.min(axis=1)
        max_rating = rating_cols.max(axis=1)

        # fmt: off
        rating_bucket_locs = {}
        rating_bucket_locs['Any AAA/AA'] = (
            min_rating <= self._convert_single_input_rating("AA-")
        )
        rating_bucket_locs['Pure A'] = (
            (min_rating > self._convert_single_input_rating("AA-"))
            & (max_rating <= self._convert_single_input_rating("A-"))
        )
        rating_bucket_locs['Split A/BBB'] = (
            (min_rating <= self._convert_single_input_rating("A-"))
            & (max_rating >= self._convert_single_input_rating("BBB+"))
        )
        rating_bucket_locs['Pure BBB+/BBB'] = (
            (min_rating >= self._convert_single_input_rating("BBB+"))
            & (max_rating <= self._convert_single_input_rating('BBB'))
        )
        rating_bucket_locs['Any BBB-/BB'] = (
            max_rating >= self._convert_single_input_rating('BBB-')
        )
        # fmt: on

        df["RatingRiskBucket"] = np.nan
        for rating_bucket, loc in rating_bucket_locs.items():
            df.loc[loc, "RatingRiskBucket"] = rating_bucket
        df["RatingRiskBucket"] = df["RatingRiskBucket"].astype("category")
        return df

    def _add_BM_treasuries(self, df):
        bm_treasury = {}
        for i in range(25):
            if i <= 2:
                bm_treasury[i] = 2
            elif i <= 3:
                bm_treasury[i] = 3
            elif i <= 6:
                bm_treasury[i] = 5
            elif i <= 15:
                bm_treasury[i] = 10
            elif i <= 23:
                bm_treasury[i] = 20
            else:
                bm_treasury[i] = np.nan
        year = df["Date"].dt.year
        maturity_date = df["MaturityDate"].dt.year
        issue_date = df["IssueDate"].dt.year
        tenor = maturity_date - year

        loc_7yr = (tenor == 7) & (issue_date == year)
        df["BMTreasury"] = tenor.map(bm_treasury).fillna(30).astype("int8")
        df.loc[loc_7yr, "BMTreasury"] = 7
        return df

    def _add_calculated_portfolio_columns(self, df):
        df = self._add_rating_risk_buckets(df)
        if "IssueYears" in df.columns:
            df = self._add_BM_treasuries(df)
        return df

    def _get_portfolio_account_strategy(
        self,
        portfolio=None,
        account=None,
        strategy=None,
        date=None,
    ):
        if portfolio is not None:
            if portfolio in self._strategy_account_map(date):
                strategy = portfolio
            elif portfolio in self._account_strategy_map(date):
                account = portfolio
            else:
                raise ValueError(f"Unrecognized `portfolio` {portfolio}")

        else:
            if strategy is not None:
                if strategy in self._strategy_account_map(date):
                    portfolio = strategy
                else:
                    raise ValueError(f"Unrecognized `Strategy` {strategy}")
            elif account is not None:
                if account in self._account_strategy_map(date):
                    portfolio = account
                else:
                    raise ValueError(f"Unrecognized `Account` {account}")

        return portfolio, account, strategy

    def load_portfolio(
        self,
        portfolio=None,
        date=None,
        start=None,
        end=None,
        name=None,
        account=None,
        strategy=None,
        ignored_accounts=None,
        manager=None,
        universe="returns",
        drop_cash=True,
        drop_treasuries=True,
        market_cols=True,
        ret_df=False,
        get_mv=False,
        hy_duration_adjustment=0.5,
        raw=False,
        empty=False,
    ):
        """
        Load portfolio data from SQL server. If end is not specified
        data is scraped through previous day. If neither start nor
        end are given only the data from previous day is scraped.

        Parameters
        ----------
        date: datetime, optional
            Single date to scrape.
        name: str, optional
            Name for returned portfolio.
        account: str or List[str], optional
            Account(s) to include in scrape.
        strategy: str or List[str], optional
            Strategy(s) to include in scrape.
        manager: str or List[str], optional
            Manager(s) to include in scrape.
        universe: ``{'stats', 'returns'}``, default='returns'
            Benchmark universe to compare current holdings to.
        market_cols: str, List[str], or bool, default=True
            Column(s) of CUSIP level market data
            (e.g., OAS, TRet, XSRet) to include in returned result.
            If ``True``, return all available market columns.
        hy_duration_adjustment: float, default=0.5,
            Empirical adjustment to duration for HY bonds in IG
            portfolios since they do not trade with full duration.
        raw: bool, default=False
            If ``True``, do not perform any preprocessing to the data
            and simply return the pd.DataFrame of raw data.
        empty: bool, default=False,
            If ``True``, return a portfolio with empty DataFrames.
            This is useful to simply look at the stored property
            history or get the portfolio's fid.

        Returns
        -------
        :class:`Account` or :class:`Strategy`:
            Portfolio class for specified date.
        """
        # Format dates for SQL call.
        fmt = "%Y%m%d"
        date = self.date("today") if date is None else to_datetime(date)
        start = end = date.strftime(fmt)

        portfolio, account, strategy = self._get_portfolio_account_strategy(
            portfolio, account, strategy
        )

        if account is None:
            sql_account = "NULL"
        else:
            acnts = to_list(account, dtype=str)
            sql_account = f"'{','.join(acnts) if len(acnts) > 1 else acnts[0]}'"

        sql_strategy = "NULL" if strategy is None else f"'{strategy}'"
        manager = "NULL" if manager is None else f"'{manager}'"
        universe = {"stats": "statistics"}.get(universe.lower(), universe)
        universe = f"'{universe.title()}'"
        inputs = [
            start,
            sql_strategy,
            sql_account,
            manager,
            "NULL",
            universe,
            "1",
            end,
        ]

        # Build SQL calls using stored procedure.
        sql_base = f"exec LGIMADatamart.[dbo].[sp_AFI_GetPortfolioAnalytics] "
        sql_benchmark = ", ".join(inputs + ["3"])
        sql_portfolio = ", ".join(inputs + ["2"])
        sql_both = ", ".join(inputs)

        if empty:
            required_cols = [
                "CUSIP",
                "ISIN",
                "Account",
                "Sector",
                "L3",
                "MaturityYears",
            ]
            df = pd.DataFrame(columns=required_cols)
        else:
            df = self.query_datamart(sql_base + sql_both)
        if raw:
            return df

        if not empty:
            df = self._preprocess_portfolio_data(df)

        if not empty and not len(df):
            port = account or strategy
            raise ValueError(
                f"No data for {port} {universe} on {date:%m/%d/%Y}."
            )

        if empty:
            pass
        elif not market_cols:
            df.sort_values(["Date", "Account", "Ticker"], inplace=True)
        else:
            # Get market data, loading if required.
            if (
                self.loaded_dates is None
                or not self.loaded_dates
                or date < self.loaded_dates[0]
                or date > self.loaded_dates[-1]
            ):
                self.load_market_data(date=date)
            market_df = self.build_market_index(
                date=date, drop_treasuries=False
            ).df

            # Subset columns to append to portfolio data.
            if market_cols is True:
                market_cols = list(set(market_df) - set(df))
            else:
                market_cols = to_list(market_cols, dtype=str)
            idx = ["Date", "CUSIP"]
            market_cols.extend(idx)

            # Combine market data to portfolio data.
            df = (
                df.set_index(idx)
                .join(market_df[market_cols].set_index(idx), on=idx)
                .reset_index()
                .sort_values(["Date", "Account", "Ticker"])
            )

        if not empty:
            date_fid = date.strftime("%Y-%m-%d")
            lgima_sectors = load_json(f"lgima_sector_maps/{date_fid}")
            df["LGIMASector"] = df["CUSIP"].map(lgima_sectors)
            lgima_top_sectors = load_json(
                f"lgima_top_level_sector_maps/{date_fid}"
            )
            df["LGIMATopLevelSector"] = df["CUSIP"].map(lgima_top_sectors)

            if get_mv:
                missing_ix = df["AmountOutstanding"].isna()
                cusips = set(df.loc[missing_ix, "CUSIP"])
                mv = bdp(cusips, "Corp", "AMT_OUTSTANDING").squeeze() / 1e6
                cusip_mv = df.loc[missing_ix, "CUSIP"].map(mv.to_dict())
                df.loc[missing_ix, "AmountOutstanding"] = cusip_mv
                df["MarketValue"] = (
                    df["AmountOutstanding"] * df["DirtyPrice"] / 100
                )

            # Correct BB's OAD for IG accounts.
            hy_eligible_strategies = ["US Credit Plus", "US Long Credit Plus"]
            hy_eligible_accounts = list(
                chain(
                    *[
                        self._strategy_account_map(date)[strat]
                        for strat in hy_eligible_strategies
                    ]
                )
            )
            if (
                strategy in hy_eligible_strategies
                or account in hy_eligible_accounts
            ):
                df.loc[
                    (df["NumericRating"] >= 11) & (df["NumericRating"] <= 13),
                    ["OAD_Diff", "P_OAD"],
                ] *= hy_duration_adjustment

            df = self._add_calculated_portfolio_columns(df)
            # df = clean_dtypes(df)
        if ret_df:
            return df

        account_market_values = self.account_market_values.loc[date]
        if start == end:
            if sql_strategy != "NULL":
                return Strategy(
                    df,
                    name=sql_strategy.strip("'"),
                    date=start,
                    account_market_values=account_market_values,
                    ignored_accounts=ignored_accounts,
                )
            elif sql_account != "NULL":
                if len(acnts) == 1:
                    return Account(
                        df,
                        sql_account.strip("'"),
                        date=start,
                        account_market_values=account_market_values,
                    )
                else:
                    return Strategy(
                        df,
                        sql_account,
                        date=start,
                        account_market_values=account_market_values,
                        ignored_accounts=ignored_accounts,
                    )
            else:
                return df
        else:
            return df

    def _read_bbg_df(self, field):
        """
        Read cached Bloomberg data file. If not cached,
        load and cache file before returning.

        Parameters
        ----------
        field: str, ``{'OAS', 'YTW', 'TRET', 'XSRET', etc.}``
            Bloomberg field to read.

        Returns
        -------
        pd.DataFrame:
            Bloomberg data for specified field.
        """
        return pd.read_csv(
            self.local(f"bloomberg_timeseries/{field}.csv"),
            index_col=0,
            parse_dates=True,
            infer_datetime_format=True,
        )

    def _clean_baml_df(self, df):
        """pd.DataFrame: clean columns in BAML decile report DataFrames."""
        new_df = pd.DataFrame()
        for col in list(df.columns):
            if "-" in col:
                continue
            elif col.count("*") == 2:
                split = col.split("*")
                col0 = split[0]
                col1 = "_".join(col.upper() for col in split[1:]).replace(
                    " ", "_"
                )
                new_df[col0] = df[col]
                new_df[col1] = df[col]
            elif col.count("*") == 1:
                split = col.split("*")
                col0 = split[0]
                col1 = split[1].replace(" ", "_")
                new_df[col0] = df[col]
                new_df[col1] = df[col]
            else:
                new_df[col] = df[col]
        return new_df.copy()

    @lru_cache(maxsize=None)
    def _read_baml_df(self, field):
        """
        Read cached Bloomberg data file. If not cached,
        load and cache file beforedont  returning.

        Parameters
        ----------
        field: str, ``{'OAS', 'YTW', 'TRET', 'XSRET', etc.}``
            Bloomberg field to read.

        Returns
        -------
        pd.DataFrame:
            Bloomberg data for specified field.
        """
        filename = {
            "OAS": "spreads",
            "YTW": "yields",
            "PRICE": "prices",
        }[field.upper()]

        raw_df = pd.read_csv(
            self.local(f"HY/decile_report/{filename}.csv"),
            index_col=0,
            parse_dates=True,
            infer_datetime_format=True,
        )
        df = self._clean_baml_df(raw_df)
        return df

    def load_bbg_data(
        self,
        securities,
        fields,
        column_names=None,
        start=None,
        end=None,
        aggregate=False,
        nan=None,
        **kwargs,
    ):
        """
        Load Bloomberg BDH data for given securities.

        Parameters
        ----------
        securities: str or List[str].
            Security or secuirites to load data for.
        fields: str or list[str], ``{'OAS', 'Price', 'TRET', 'YTW', etc.}``
            Field(s) of data to load.
        column_names: ``{'security', 'field', 'both'}``, optional
            Column names when multiple securities and fields are included.
            By default the security name will be used.
        start: datetime, optional.
            Inclusive start date for data.
        end: datetime, optional.
            Inclusive end date for data.
        aggregate: bool, default=False
            If ``True`` aggregate total or excess returns over the
            specified period.
        nan: ``{None, 'drop', 'ffill', 'interp'}``, optional
            Method to use for filling missing values in loaded data.
            Method 'fbfill' performs a forward fill followed by a
            backwards fill, useful when aggregating.
        kwargs:
            Keyword arguments to pass to interpolating function.

            * method: ``{'linear', 'spline', etc.}``, default='linear'
                ``'spline'`` and ``'polynomial'`` require an order.
            * order: int, order for polynomial or spline.

        Returns
        -------
        df: pd.DataFrame
            Bloomberg data subset to proper date range and
            processed missing data.
        """
        # Load data for selected securities.
        fields = [field.upper() for field in to_list(fields, dtype=str)]
        securities = to_list(securities, dtype=str)

        if aggregate is True:
            if len(fields) > 1:
                raise ValueError("Cannot aggregate more than 1 field.")
            return self._aggregate_bbg_returns(
                securities, fields[0], start, end
            )

        multiple_securities = len(securities) > 1 or securities == ["all"]
        multiple_fields = len(fields) > 1

        if len(securities) == 1:
            securities = securities[0]

        if len(fields) > 1 and multiple_securities:
            if len(fields) == len(securities):
                if column_names is None or column_names == "security":
                    col_names = securities
                elif column_names == "field":
                    col_names = fields
                elif column_names == "both":
                    col_names = [f"{s}_{f}" for s, f in zip(securities, fields)]
                else:
                    raise ValueError("Improper value for `column_names`")
                df = pd.concat(
                    (
                        self._read_bbg_df(field)[security].rename(col)
                        for field, security, col in zip(
                            fields, securities, col_names
                        )
                    ),
                    axis=1,
                )
            else:
                raise ValueError(
                    "Number of securities does not match number of fields."
                )

        elif len(fields) > 1:
            df = pd.concat(
                (
                    self._read_bbg_df(field)[securities].rename(field)
                    for field in fields
                ),
                axis=1,
            )
        else:
            field = fields[0]
            if securities == "all":
                df = self._read_bbg_df(field).copy()
            else:
                df = self._read_bbg_df(field)[securities].copy()

        # Subset to proper date region
        if start is not None:
            df = df[df.index >= to_datetime(start)]
        if end is not None:
            df = df[df.index <= to_datetime(end)]

        # Process missing values.
        df.dropna(how="all", inplace=True)
        if nan is None:
            pass
        elif nan == "drop":
            df.dropna(inplace=True)
        elif nan == "ffill":
            df.fillna(method="ffill", inplace=True)
        elif nan == "interp":
            interp_kwargs = {"method": "linear"}
            interp_kwargs.update(**kwargs)
            df.interpolate(
                limit_direction="forward", axis=0, inplace=True, **interp_kwargs
            )
        elif nan == "fbfill":
            df = df.fillna(method="ffill").fillna(method="bfill")

        return df

    def _aggregate_bbg_returns(self, securities, field, start, end):
        regular_xsret_timeseries_securities = {
            "SP500",
            "EURO_STOXX_50",
            "CDX_HY",
            "CDX_IG",
            "ITRAXX_MAIN",
            "ITRAXX_XOVER",
        }
        d = {}
        for security in securities:
            if "TRET" in field:
                tret = self.load_bbg_data(
                    security, "TRET", nan="drop", start=start, end=end
                )
                d[security] = tret.iloc[-1] / tret.iloc[0] - 1
            elif field == "XSRET":
                if security in regular_xsret_timeseries_securities:
                    # Bloomberg has normal index series.
                    # Treat similar to total returns
                    xsret = self.load_bbg_data(
                        security, "XSRET", nan="drop", start=start, end=end
                    )
                    d[security] = xsret.iloc[-1] / xsret.iloc[0] - 1
                else:
                    # Not a derivative.
                    # Load entire history for security up to specified end.
                    df = self.load_bbg_data(
                        security, ["TRET", "XSRET"], nan="drop", end=end
                    )
                    if start is not None:
                        # Subset data to start at month before specifed
                        # start date. This is required because Bloomberg
                        # has MTD excess returns, so the full month of data
                        # is needed to start aggregation at any point in
                        # the month.
                        prev_month_start = self.date(
                            "LAST_MONTH_END", start, trade_dates=df.index
                        )
                        df = df[df.index >= prev_month_start]
                    d[security] = self._aggregate_bbg_excess_returns(df, start)
            else:
                raise KeyError(f"{field} is not a valid field to aggregate.")

        if len(securities) == 1:
            return d[securities[0]]
        else:
            return pd.Series(d, name=field)

    def _aggregate_bbg_excess_returns(self, df, start):
        """
        Parameters
        ----------
        df: pd.DataFrame
            DataFrame of total and excess returns.
        start: datetime, optional
            Starting date for aggregation.
        """
        # Split DataFrame into months.
        month_dfs = [mdf for _, mdf in df.groupby(pd.Grouper(freq="M"))]
        tret_ix_0 = month_dfs[0]["TRET"].iloc[-1]  # last value of prev month

        xs_col_month_list = []
        for mdf in month_dfs[1:]:  # first month is ignored
            a = np.zeros(len(mdf))
            for i, row in enumerate(mdf.itertuples()):
                tret_ix, cum_xsret = row[1], row[2] / 100
                if i == 0:
                    a[i] = cum_xsret
                    tret = (tret_ix - tret_ix_0) / tret_ix_0
                    prev_tret_ix = tret_ix
                    prev_cum_rf_ret = 1 + tret - cum_xsret
                else:
                    cum_tret = (tret_ix - tret_ix_0) / tret_ix_0
                    tret = tret_ix / prev_tret_ix - 1
                    rf_ret = (cum_tret - cum_xsret + 1) / prev_cum_rf_ret - 1
                    a[i] = tret - rf_ret
                    prev_tret_ix = tret_ix
                    prev_cum_rf_ret *= 1 + rf_ret

            tret_ix_0 = tret_ix
            xs_col_month_list.append(pd.Series(a, index=mdf.index))

        xsret_s = pd.concat(xs_col_month_list, sort=True)
        tret_s = (df["TRET"] / df["TRET"].shift(1) - 1)[1:]
        if start is not None:
            xsret_s = xsret_s[xsret_s.index >= start]
            tret_s = tret_s[tret_s.index >= start]
        rf_ret_s = tret_s - xsret_s
        total_ret = np.prod(1 + tret_s) - 1
        rf_total_ret = np.prod(1 + rf_ret_s) - 1
        return total_ret - rf_total_ret

    def load_baml_data(
        self, securities, fields, start=None, end=None, nan=None, **kwargs
    ):
        """
        Load BAML Index data for given securities.

        Parameters
        ----------
        securities: str or List[str].
            Security or secuirites to load data for.
        fields: str or list[str], ``{'OAS', 'Price', 'TRET', 'YTW', etc.}``
            Field(s) of data to load.
        start: datetime, optional.
            Inclusive start date for data.
        end: datetime, optional.
            Inclusive end date for data.
        nan: ``{None, 'drop', 'ffill', 'interp'}``, optional
            Method to use for filling missing values in loaded data.
        kwargs:
            Keyword arguments to pass to interpolating function.

            * method: ``{'linear', 'spline', etc.}``, default='linear'
                ``'spline'`` and ``'polynomial'`` require an order.
            * order: int, order for polynomial or spline.

        Returns
        -------
        df: pd.DataFrame
            BAML data subset to proper date range and
            processed missing data.
        """
        # Load data for selected securities.
        fields = to_list(fields, dtype=str)
        securities = to_list(securities, dtype=str)

        multiple_securities = len(securities) > 1 or securities == ["all"]
        if len(fields) > 1 and multiple_securities:
            raise ValueError(
                "Cannot have multiple securities and multiple fields."
            )

        if len(securities) == 1:
            securities = securities[0]

        fields = [field.upper() for field in fields]
        if len(fields) > 1:
            df = pd.concat(
                (
                    self._read_baml_df(field)[securities].rename(field)
                    for field in fields
                ),
                axis=1,
            )
        else:
            field = fields[0]
            if securities == "all":
                df = self._read_baml_df(field).copy()
            else:
                df = self._read_baml_df(field)[securities].copy()

        # Subset to proper date region
        if start is not None:
            df = df[df.index >= to_datetime(start)]
        if end is not None:
            df = df[df.index <= to_datetime(end)]

        # Process missing values.
        df.dropna(how="all", inplace=True)
        if nan is None:
            pass
        elif nan == "drop":
            df.dropna(inplace=True)
        elif nan == "ffill":
            df.fillna(method="ffill", inplace=True)
        elif nan == "interp":
            interp_kwargs = {"method": "linear"}
            interp_kwargs.update(**kwargs)
            df.interpolate(
                limit_direction="forward", axis=0, inplace=True, **interp_kwargs
            )

        return df

    @cached_property
    def _bbg_name_dict(self):
        """
        Formatted names for Bloomberg data.

        Returns
        -------
        names: Dict[str: str].
            Map from Bloomberg json code to foramtted name.
        """
        bbg_ts_codes = load_json("bloomberg_timeseries_codes")
        names = {
            security: fields["NAME"]
            for security, fields in bbg_ts_codes.items()
            if "NAME" in fields
        }
        return names

    def bbg_names(self, codes):
        code_list = to_list(codes, dtype=str)
        names = [self._bbg_name_dict[code] for code in code_list]
        if len(names) == 1:
            return names[0]
        else:
            return names

    def load_cusip_event_dates(self, start=None, end=None):
        """
        Load event dates for cusips.

        Parameters
        ----------
        start: datetime, optional.
            Inclusive start date for data.
        end: datetime, optional.
            Inclusive end date for data.

        Returns
        -------
        s: pd.Series
            Event dates indexed by respective CUSIPs.
        """
        sql = cleandoc(
            """
            select i.*
            from dbo.DimInstrument i
            inner join
            (
                select cusip, max(instrumentkey) as InstrumentKey,
                    max(dateend) as DateEnd
                from diminstrument
                group by cusip
            ) a
            on i.instrumentkey = a.instrumentkey
            order by i.cusip
            """
        )
        df = self.query_datamart(sql)
        s = pd.to_datetime(df.set_index("CUSIP")["DateEnd"])
        if start is not None:
            s = s[s >= to_datetime(start)]
        if end is not None:
            s = s[s <= to_datetime(end)]
        return s

    def strategy_fid(self, strategy):
        repl = {" ": "_", "/": "_", "%": "pct"}
        return replace_multiple(strategy, repl)

    def ticker_overweights(self, strategy, ticker, start=None, end=None):
        tickers = to_list(ticker, str)
        fid = f"{self.strategy_fid(strategy)}_tickers.parquet"
        dir_name = self.local("strategy_overweights")
        df = pd.read_parquet(dir_name / fid)
        if start:
            df = df[df.index >= to_datetime(start)]
        if end:
            df = df[df.index <= to_datetime(end)]
        return df[tickers].squeeze()

    def _convert_single_input_rating(self, rating, nan_val=None):
        """int: Convert single input rating to numeric rating."""
        if rating is None:
            return nan_val

        try:
            return self._ratings[str(rating)]
        except KeyError:
            raise KeyError(f"'{rating}' is not an allowable rating.")

    def convert_input_ratings(self, rating_range):
        """tuple[int]: Convert input ratings to a numeric ratings."""
        if isinstance(rating_range, str):
            if rating_range == "IG":
                return (1, 10)
            elif rating_range == "HY":
                return (11, 21)
            else:
                # Single rating provided.
                rating = self._convert_single_input_rating(rating_range)
                return (rating, rating)
        else:
            min_rating = self._convert_single_input_rating(rating_range[0], 0)
            max_rating = self._convert_single_input_rating(rating_range[1], 23)
            return (min_rating, max_rating)

    def _add_hy_index_flags(self, df):
        """pd.DataFrame: Add index flags for HY indexes to Dataframe."""
        ix_date_isins = df.set_index("Date")["ISIN"]
        start = ix_date_isins.index[0]
        end = ix_date_isins.index[-1]

        indexes = ["H4UN", "H0A0", "HC1N", "HUC2", "HUC3"]
        for index in indexes:
            # Load flag data and subset to correct dates.
            flag_df = self._hy_index_flags(index)
            flag_d = flag_df[
                (flag_df.index >= start) & (flag_df.index <= end)
            ].T.to_dict()
            flags = np.zeros(len(ix_date_isins))
            for i, (date, isin) in enumerate(ix_date_isins.items()):
                try:
                    flags[i] = flag_d[date].get(isin, 0)
                except KeyError:
                    continue
            df[f"{index}Flag"] = flags.astype("int8")

        return df

    def convert_state_codes(self, states):
        """List[str]: Convert state names to state codes."""
        state_codes = []
        all_states = set(self.state_codes.keys())
        for state in to_list(states, dtype=str):
            key, score = fuzzywuzzy.process.extract(state, all_states)[0]
            if score > 85:
                state_codes.append(self.state_codes[key])
            else:
                state_codes.append(np.NaN)

        if len(state_codes) == 1:
            return state_codes[0]
        else:
            return state_codes

    def query_3PDH(self, table, query):
        import awswrangler as wr

        return wr.athena.read_sql_query(
            sql=query,
            database=table,
            s3_output=self._passwords["AWS"]["loc"],
            ctas_approach=False,
            boto3_session=self._3PDH_sess,
        )

    def account_flows(self, account):
        mv = self.account_market_values[account].dropna()
        strategy = self._account_strategy_map()[account]
        bm = self._strategy_benchmark_map[strategy]
        tret_level = self.load_bbg_data(
            bm, "TRET", start=mv.index[0], end=mv.index[-1]
        )
        dates = sorted(list(set(mv.index) & set(tret_level.index)))

        mv_pct_chg = mv.loc[dates].pct_change()
        tret = tret_level.loc[dates].pct_change()
        return (mv_pct_chg - tret).dropna()


# %%
def main():
    pass
    # %%
    import time
    from collections import defaultdict

    import seaborn as sns
    from tqdm import tqdm

    from lgimapy import vis
    from lgimapy.bloomberg import bdp, bdh, bds, get_cashflows
    from lgimapy.data import IG_sectors, HY_sectors
    from lgimapy.portfolios import AttributionIndex
    from lgimapy.utils import (
        get_ordinal,
        mkdir,
        mock_df,
        Time,
        to_clipboard,
        to_sql_list,
    )

    vis.style()
    self = Database()
    self.display_all_columns()
    self.display_all_rows(100)

    db = Database()
    # %%
    db.load_market_data()

    # %%
    # port = db.load_portfolio(account="LIB150")
    # port.derivatives_df.iloc[0]
    # port.derivatives_df.iloc[1]

    # %%
    db = Database()
    port = db.load_portfolio(account="PMCHY")
    port.performance()

    sorted(port.df.columns)
    # %%
    port.df.dtypes.value_counts()
    cols = []
    for col, dtype in port.df.dtypes.items():
        if dtype == "float64":
            cols.append(col)

    sorted(cols)

    # %%
    port = db.load_portfolio("SALD")
    # %%
    db = Database()
    df = db.load_market_data(
        date="6/30/2022",
        clean=False,
        preprocess=False,
        local=False,
        ret_df=True,
    )
    tsy_df = df[df["IndustryClassification4"] == "TREASURIES"]
    len(tsy_df)

    # %%
    db._account_strategy_map()["LEGSCC"]
