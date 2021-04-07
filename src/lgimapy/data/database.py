import json
import pickle
import warnings
from bisect import bisect_left
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from functools import lru_cache, partial
from glob import glob
from inspect import cleandoc, getfullargspec
from itertools import chain
from pathlib import Path

import datetime as dt
import numpy as np
import pandas as pd
import pyodbc

from lgimapy.bloomberg import bdp, get_bloomberg_subsector
from lgimapy.data import (
    clean_dtypes,
    concat_index_dfs,
    convert_sectors_to_fin_flags,
    credit_sectors,
    HY_sectors,
    IG_sectors,
    IG_market_segments,
    Index,
    new_issue_mask,
    Account,
    Strategy,
)
from lgimapy.utils import (
    check_market,
    dump_json,
    load_json,
    nearest_date,
    replace_multiple,
    root,
    sep_str_int,
    to_datetime,
    to_int,
    to_list,
    to_set,
)

# %%


def get_basys_fids(market):
    """
    Get list of basys fids for given market.

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
    dir = Path(f"S:/FrontOffice/Bonds/BASys/CSVFiles/MarkIT/{market}/")
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
        self.market = check_market(market)
        # Load mapping from standard ratings to numeric values.
        self._bbg_dfs = {}
        n = 1 if server.upper() == "LIVE" else 2
        self._conn = pyodbc.connect(
            "Driver={SQL Server};"
            # f"SERVER=XWNUSSQL0{n}\\{server.upper()};"
            f"SERVER=l00-lgimadatamart-sql,14333;"
            "DATABASE=LGIMADatamart;"
            "Trusted_Connection=yes;"
            "UID=inv\JB11115;"
            "PWD=$Jrb1236463716;"
        )

    def make_thread_safe(self):
        """
        Remobe pyodb.Connection object so :class:`Database` can
        be pickled and used in multiprocessing/multithreading.
        """
        del self._conn

    @property
    @lru_cache(maxsize=None)
    def _ratings(self):
        """Dict[str: int]: Memoized rating map."""
        ratings_map = load_json("ratings")
        for i in range(23):
            ratings_map[i] = i
        return ratings_map

    @property
    @lru_cache(maxsize=None)
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
        fid = root(f"data/{market}/trade_dates.parquet")
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

    @property
    @lru_cache(maxsize=None)
    def _rating_changes_df(self):
        """pd.DataFrame: Rating change history."""
        fid = root("data/rating_changes.parquet")
        return pd.read_parquet(fid)

    @property
    @lru_cache(maxsize=None)
    def long_corp_sectors(self):
        """List[str]: list of sectors in Long Corp Benchmark."""
        fid = root("data/long_corp_sectors.parquet")
        sector_df = pd.read_parquet(fid)
        return sorted(sector_df["Sector"].values)

    def credit_sectors(self, *args, **kwargs):
        return credit_sectors(*args, **kwargs)

    def HY_sectors(self, *args, **kwargs):
        return HY_sectors(*args, **kwargs)

    def IG_sectors(self, *args, **kwargs):
        return IG_sectors(*args, **kwargs)

    def IG_market_segments(self, *args, **kwargs):
        return IG_market_segments(*args, **kwargs)

    def rating_changes(
        self,
        start=None,
        end=None,
        fallen_angels=False,
        rising_stars=False,
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

    def date(
        self,
        date_delta,
        reference_date=None,
        market=None,
        fcast=False,
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
        trade_dates = self.trade_dates(market=market)

        if date_delta == "PORTFOLIO_START":
            return pd.to_datetime("9/1/2018")
        if date_delta == "HY_START":
            return pd.to_datetime("6/1/2020")
        if date_delta == "MARKET_START":
            return {
                "US": pd.to_datetime("2/2/1998"),
                "EUR": pd.to_datetime("12/1/2014"),
                "GBP": pd.to_datetime("12/1/2014"),
            }[market]
        # Use today as reference date if none is provided,
        # otherwise find closes trading date to reference date.
        if reference_date is None:
            today = trade_dates[-1]
        else:
            today = self.nearest_date(reference_date, market=market, **kwargs)

        last_trade = partial(bisect_left, trade_dates)
        if date_delta == "TODAY":
            return today
        elif date_delta == "YESTERDAY" or date_delta == "DAILY":
            return self.nearest_date(
                today, market=market, inclusive=False, after=False
            )
        elif date_delta in {"WTD", "LAST_WEEK_END"}:
            return trade_dates[
                last_trade(today - timedelta(today.weekday() + 1)) - 1
            ]
        elif date_delta == "MONTH_START":
            return trade_dates[last_trade(today.replace(day=1))]
        elif date_delta == "NEXT_MONTH_START":
            # Find first trade date that is on or after the 1st
            # of the next month.
            next_month = 1 if today.month == 12 else today.month + 1
            next_year = today.year + 1 if today.month == 12 else today.year
            next_month_start = pd.to_datetime(f"{next_month}/1/{next_year}")
            return self.nearest_date(
                next_month_start, market=market, before=False
            )
        elif date_delta == "MONTH_END":
            next_month = self.date("NEXT_MONTH_START", reference_date=today)
            return self.date("MTD", reference_date=next_month)
        elif date_delta in {"MTD", "LAST_MONTH_END"}:
            return trade_dates[last_trade(today.replace(day=1)) - 1]
        elif date_delta in {"YTD", "LAST_YEAR_END"}:
            return trade_dates[last_trade(today.replace(month=1, day=1)) - 1]
        else:
            # Assume value-unit specification.
            positive = "+" in date_delta
            val, unit = sep_str_int(date_delta.strip("+"))
            reverse_kwarg_map = {
                "days": ["D", "DAY", "DAYS"],
                "weeks": ["W", "WEEK", "WEEKS"],
                "months": ["M", "MONTH", "MONTHS"],
                "years": ["Y", "YEAR", "YEARS"],
            }
            dt_kwarg_map = {
                key: kwarg
                for kwarg, keys in reverse_kwarg_map.items()
                for key in keys
            }
            dt_kwargs = {dt_kwarg_map[unit]: val}
            if positive:
                if fcast:
                    return today + relativedelta(**dt_kwargs)
                else:
                    return self.nearest_date(
                        today + relativedelta(**dt_kwargs),
                        market=market,
                        **kwargs,
                    )
            else:
                return self.nearest_date(
                    today - relativedelta(**dt_kwargs), market=market, **kwargs
                )

    def account_values(self):
        query = f"""
            SELECT AV.DateKey, AV.MarketValue, DA.BloombergID
            FROM dbo.AccountValue AV
            JOIN dbo.DimAccount DA ON AV.AccountKey = DA.AccountKey
            """
        df = pd.read_sql(query, self._conn)
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

    @property
    @lru_cache(maxsize=None)
    def _account_to_strategy(self):
        """Dict[str: str]: Account name to strategy map."""
        return load_json("account_strategy")

    @property
    @lru_cache(maxsize=None)
    def _strategy_to_accounts(self):
        """Dict[str: str]: Respective accounts for each strategy."""
        return load_json("strategy_accounts")

    @property
    @lru_cache(maxsize=None)
    def _manager_to_accounts(self):
        """Dict[str: str]: Respective accounts for each PM."""
        return load_json("manager_accounts")

    @property
    @lru_cache(maxsize=None)
    def DM_countries(self):
        """Set[str]: Developed market country codes."""
        fid = root("data/DM_countries.parquet")
        return set(pd.read_parquet(fid).squeeze())

    @lru_cache(maxsize=None)
    def _hy_index_flags(self, index):
        """pd.DaataFrame: Index flag boolean for ISIN vs date."""
        fid = root(f"data/index_members/{index}.parquet")
        return pd.read_parquet(fid)

    def load_trade_dates(self):
        """List[datetime]: Dates with credit data."""
        dates_sql = "select distinct effectivedatekey from \
            dbo.InstrumentAnalytics order by effectivedatekey"

        return list(
            pd.to_datetime(
                pd.read_sql(dates_sql, self._conn).values.ravel(),
                format="%Y%m%d",
            )
        )

    @property
    @lru_cache(maxsize=None)
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
        elif isinstance(ratings, (float, int)):
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
        }
        df.rename(columns=col_map, inplace=True)

        # Make flag columns -1 for nonexistent data to
        # mitigate memory usage by allwoing int dtype.
        df["Eligibility144AFlag"].fillna(-1, inplace=True)

        # Convert str time to datetime.
        for date_col in ["Date", "MaturityDate", "IssueDate", "NextCallDate"]:
            df[date_col] = pd.to_datetime(
                df[date_col], format="%Y-%m-%d", errors="coerce"
            )

        # Capitalize market of issue, sector, and issuer.
        for col in ["MarketOfIssue", "Sector", "Issuer"]:
            df[col] = df[col].str.upper()

        # Map sectors to standard values.
        df["Sector"] = df["Sector"].str.replace(" ", "_")
        for col in ["BAMLSector", "BAMLTopLevelSector"]:
            df[col] = self._clean_sector(df[col])

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
        return clean_dtypes(df)

    def _clean_sector(self, sector):
        """Clean sector names so they can be saved as a fid."""
        pattern = "|".join([" - ", "-", "/", " "])
        return (
            sector.str.upper().str.replace("&", "AND").str.replace(pattern, "_")
        )

    def _preprocess_basys_data(self, df):
        # Fill missing sectors and subsectors.
        sector_cols = [f"Level {i}" for i in range(7)]
        df[sector_cols] = (
            df[sector_cols]
            .replace("*", np.nan)
            .fillna(method="ffill", axis=1)
            .copy()
        )
        df.loc[df["Level 1"] == "Sovereigns", "Level 5"] = "Sovereigns"

        # Fill missing collateral types.
        collat_cols = [f"Seniority Level {i+1}" for i in range(3)]
        df[collat_cols] = (
            df[collat_cols]
            .replace("*", np.nan)
            .fillna(method="ffill", axis=1)
            .copy()
        )

        col_map = {
            "Date": "Date",
            "Ticker": "Ticker",
            "Issuer": "Issuer",
            "ISIN": "ISIN",
            "CUSIP": "CUSIP",
            "Coupon": "CouponRate",
            "Final Maturity": "MaturityDate",
            "Workout date": "WorstDate",
            "Level 5": "MLSector",
            "Level 6": "MLSubsector",
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
            "Semi-Annual Yield": "YieldToWorst",
            "Semi-Annual Yield to Maturity": "YieldToMat",
            "Semi-Annual Modified Duration": "ModDurationToWorst",
            "Semi-Annual Modified Duration to Maturity": "ModDurationToMat",
            "Month-to-date Sovereign Curve Swap Return": "MTDXSRet",
            "Month-to-date Libor Swap Return": "MTDLiborXSRet",
        }
        df = df.rename(columns=col_map)[col_map.values()].copy()

        # Convert str time to datetime.
        for date_col in ["Date", "MaturityDate", "WorstDate", "NextCallDate"]:
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

        # Define maturites yearrs.
        day = "timedelta64[D]"
        df["MaturityYears"] = (df["MaturityDate"] - df["Date"]).astype(
            day
        ) / 365

        # Make sector column NaN. This column is required to be present.
        df["Sector"] = np.nan

        # Capitalize sectors, and issuers.
        for col in ["MLSubsector", "MLSector", "Issuer"]:
            df[col] = df[col].str.upper()
        df["MLSector"] = df["MLSector"].str.replace(" ", "_")

        # Put amount outstanding and market value into $M.
        for col in ["AmountOutstanding", "MarketValue"]:
            df[col] /= 1e6

        # Get numeric ratings from composit ratings.
        df["NumericRating"] = self._get_numeric_ratings(df, ["CompositeRating"])

        # Add financial flag column.
        df["FinancialFlag"] = convert_sectors_to_fin_flags(df["MLSector"])

        # Calculate DTS. Approximate OAD ~ OASD.
        df["DTS"] = df["OAD"] * df["OAS"]
        df["DTS_Libor"] = df["OAD"] * df["ZSpread"]

        df["CUSIP"].fillna(df["ISIN"], inplace=True)
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
            "USAggReturnsFlag",
            "USAggStatisticsFlag",
        }
        index_flag_cols = [
            col
            for col in df.columns
            if "flag" in col.lower() and col not in ignored_flag_cols
        ]
        df[index_flag_cols] = df[index_flag_cols].replace(-1, 0)
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
                & (df["CountryOfRisk"].isin(self.DM_countries))
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

        # Make new fields categories and drop duplicates.
        return clean_dtypes(df).drop_duplicates(subset=["CUSIP", "Date"])

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
        fid_dir = root(f"data/{self._market}/{self._fid_extension}s")
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
        df_chunk = pd.read_sql(query, self._conn, chunksize=50_000)
        chunk_list = []
        for chunk in df_chunk:
            # Preprocess chunk.
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
        df_list = []
        for fid in trade_dates["fid"]:
            df_raw = pd.read_csv(
                Path(fid),
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

    def load_market_data(
        self,
        date=None,
        start=None,
        end=None,
        cusips=None,
        clean=True,
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
            If true, apply standard cleaning rules to loaded DataFrame.
        ret_df: bool, default=False
            If True, return loaded DataFrame.
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
            "Country": "CountryOfRisk",
            "PMV": "P_Weight",
            "PMV INDEX": "BM_Weight",
            "PMV VAR": "Weight_Diff",
            "OADAgg": "P_OAD",
            "BmOADAgg": "BM_OAD",
            "OADVar": "OAD_Diff",
            "AssetValue": "P_AssetValue",
            "BMAssetValue": "BM_AssetValue",
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

        # Fix Charter having the old Time Warner Cable ticker.
        df.loc[df["Ticker"] == "TWC", "Ticker"] = "CHTR"

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

        # Clean analyst ratings.
        df["AnalystRating"] = df["AnalystRating"].replace("NR", np.nan)
        return clean_dtypes(df).drop_duplicates(
            subset=["Date", "CUSIP", "Account"]
        )

    def load_portfolio(
        self,
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
    ):
        """
        Load portfolio data from SQL server. If end is not specified
        data is scraped through previous day. If neither start nor
        end are given only the data from previous day is scraped.

        Parameters
        ----------
        date: datetime, optional
            Single date to scrape.
        start: datetime, optional
            Starting date for scrape.
        end: datetime, optional
            Ending date for scrape.
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
        Returns
        -------
        df: pd.DataFrame
            Portfolio history over specified date range.
        """
        # Format dates for SQL call.
        fmt = "%Y%m%d"
        current_date = self.trade_dates()[-1].strftime(fmt)
        if date is not None:
            start = end = date
        start = (
            current_date if start is None else to_datetime(start).strftime(fmt)
        )
        end = current_date if end is None else to_datetime(end).strftime(fmt)

        # Convert inputs for SQL call.
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

        # # Query SQL.
        df = pd.read_sql(sql_base + sql_both, self._conn)
        df = self._preprocess_portfolio_data(df)
        if not len(df):
            raise ValueError("No data for specified date.")

        if not market_cols:
            df.sort_values(["Date", "Account", "Ticker"], inplace=True)
        else:
            # Get market data, loading if required.
            if (
                self.loaded_dates is None
                or to_datetime(start) < self.loaded_dates[0]
                or to_datetime(end) > self.loaded_dates[-1]
            ):
                self.load_market_data(start=start, end=end)
            market_df = self.build_market_index(
                start=start, end=end, drop_treasuries=False
            ).df

            # Subset columns to append to portfolio data.
            if market_cols is True:
                market_cols = list(set(market_df) - set(df))
            else:
                market_cols = to_list(market_cols, dtype=str)
            ix = ["Date", "CUSIP"]
            market_cols.extend(ix)

            # Combine market data to portfolio data.
            df = (
                df.set_index(ix)
                .join(market_df[market_cols].set_index(ix), on=ix)
                .reset_index()
                .sort_values(["Date", "Account", "Ticker"])
            )

        # Return correct class.
        if start == end:
            date = pd.to_datetime(start).strftime("%Y-%m-%d")
            lgima_sectors = load_json(f"lgima_sector_maps/{date}")
            df["LGIMASector"] = df["CUSIP"].map(lgima_sectors)
            lgima_top_sectors = load_json(f"lgima_top_level_sector_maps/{date}")
            df["LGIMATopLevelSector"] = df["CUSIP"].map(lgima_top_sectors)

        if get_mv:
            missing_ix = df["AmountOutstanding"].isna()
            cusips = set(df.loc[missing_ix, "CUSIP"])
            mv = bdp(cusips, "Corp", "AMT_OUTSTANDING").squeeze() / 1e6
            cusip_mv = df.loc[missing_ix, "CUSIP"].map(mv.to_dict())
            df.loc[missing_ix, "AmountOutstanding"] = cusip_mv
            df["MarketValue"] = df["AmountOutstanding"] * df["DirtyPrice"] / 100

        # Correct BB's OAD for IG accounts.
        hy_eligible_strategies = ["US Credit Plus", "US Long Credit Plus"]
        hy_eligible_accounts = list(
            chain(
                *[
                    self._strategy_to_accounts[strat]
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

        if ret_df:
            return df
        if start == end:
            if sql_strategy != "NULL":
                return Strategy(
                    df,
                    name=sql_strategy.strip("'"),
                    date=start,
                    ignored_accounts=ignored_accounts,
                )
            elif sql_account != "NULL":
                if len(acnts) == 1:
                    return Account(df, sql_account.strip("'"), date=start)
                else:
                    return Strategy(
                        df,
                        sql_account,
                        date=start,
                        ignored_accounts=ignored_accounts,
                    )
            else:
                return df
        else:
            return df

    def update_portfolio_account_data(self):
        # Load SQL table.
        sql = """\
            SELECT\
                BloombergId,\
                PrimaryPortfolioManager,\
                s.[Name] as PMStrategy\
            FROM [LGIMADatamart].[dbo].[DimAccount] a (NOLOCK)\
            INNER JOIN LGIMADatamart.dbo.DimStrategy s (NOLOCK)\
                ON a.StrategyKey = s.StrategyKey\
            WHERE a.DateEnd = '9999-12-31'\
                AND a.DateClose IS NULL\
            ORDER BY BloombergID\
            """
        df = pd.read_sql(sql, self._conn)
        df.columns = ["account", "manager", "strategy"]
        # Save account: strategy json.
        account_strategy = {
            row["account"]: row["strategy"] for _, row in df.iterrows()
        }
        dump_json(account_strategy, "account_strategy")

        # Save PM: accounts json.
        pm_accounts = {
            pm: list(df[df["manager"] == pm]["account"])
            for pm in df["manager"].unique()
        }
        dump_json(pm_accounts, "manager_accounts")

        # Save Strategy: accounts json.
        strategy_accounts = {
            strat: list(df[df["strategy"] == strat]["account"])
            for strat in df["strategy"].unique()
        }
        dump_json(strategy_accounts, "strategy_accounts")

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
        try:
            return self._bbg_dfs[field]
        except KeyError:
            df = pd.read_csv(
                root(f"data/bloomberg_timeseries/{field}.csv"),
                index_col=0,
                parse_dates=True,
                infer_datetime_format=True,
            )
            self._bbg_dfs[field] = df
            return df

    def load_bbg_data(
        self, securities, fields, start=None, end=None, nan=None, **kwargs
    ):
        """
        Load Bloomberg BDH data for given securities.

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
            Bloomberg data subset to proper date range and
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

        return df

    @property
    @lru_cache(maxsize=None)
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
        df = pd.read_sql(sql, self._conn)
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
        dir_name = root("data/strategy_overweights")
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


# %%
def main():
    pass
    # %%
    from collections import defaultdict
    from lgimapy import vis
    from lgimapy.utils import Time, load_json, dump_json
    from lgimapy.bloomberg import bdp, bdh
    from tqdm import tqdm
    from lgimapy.data import groupby

    from lgimapy.data import IG_sectors, HY_sectors

    vis.style()
    self = Database()

    # %%
    db = Database()
    db.load_market_data()
    ix_lc = db.build_market_index(in_H4UN_index=True)
    # %%
    isins = set()
    for sector in HY_sectors():
        ix = ix_lc.subset(**db.index_kwargs(sector, source="baml"))
        isins |= set(ix.isins)

    ix = ix_lc.subset(isin=isins, special_rules="~ISIN")
    len(ix.df)
    sorted(ix.df["BAMLSector"].unique())

    # %%
    db.load_market_data(start="1/1/2020")
    # %%
    cusips = ["03523TBV9", "21036PBD9", "60871RAH3", "423012AG8"]
    ix = db.build_market_index(cusip=cusips, start="3/31/2020")
    ix.get_value_history("OAS").to_csv("spreads_for_My.csv")
    # %%
    df = db.rating_changes(start="3/1/2021")
    df = df[
        (df["NumericRating_CHANGE"] < 0)
        | (df["SPRating_CHANGE"] < 0)
        | (df["MoodyRating_CHANGE"] < 0)
        | (df["FitchRating_CHANGE"] < 0)
    ]
    port = db.load_portfolio(account="CITLD")
    bm = port.df["BM_DTS"].sum()
    p = port.df["P_DTS"].sum()
    p / bm
    p * 0.95 / bm
    port.dts("pct")

    weeks = (db.date("today") - pd.to_datetime("1/1/2021")).days / 7
    132 / weeks

    # %%
    name = 'I00039'
    start = '4/1/2021'
    fields = ['INDEX_OAS_TSY', 'INDEX_OAD_TSY', 'INDEX_VALUE', 'INDEX_EXCESS_RETURN_MTD', 'INDEX_YIELD_TO_WORST', 'INDEX_MARKET_VALUE']
    bdh(name, yellow_key="Index", fields=fields, start=start).T
