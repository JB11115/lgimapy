import json
import pickle
import warnings
from bisect import bisect_left
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from functools import lru_cache, partial
from glob import glob
from inspect import cleandoc, getfullargspec

import datetime as dt
import numpy as np
import pandas as pd
import pyodbc

from lgimapy.bloomberg import get_bloomberg_subsector
from lgimapy.data import concat_index_dfs, Index, new_issue_mask
from lgimapy.utils import (
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
def clean_dtypes(df):
    """pd.DataFrame: Convert dtypes columns to proper dtype."""
    reverse_dtype_dict = {
        "float32": [
            "CouponRate",
            "CleanPrice",
            "OAD",
            "OAS",
            "OASD",
            "OAS_1W",
            "OAS_1M",
            "OAS_3M",
            "OAS_6M",
            "OAS_12M",
            "LiquidityCostScore",
            "AccruedInterest",
            "AmountOutstanding",
            "MarketValue",
            "LQA",
            "KRD06mo",
            "KRD02yr",
            "KRD05yr",
            "KRD10yr",
            "KRD20yr",
            "KRD30yr",
            "YieldToWorst",
            "NumericRating",
            "MaturityYears",
            "IssueYears",
            "DirtyPrice",
            "MarketValue",
            "Quantity",
            "AccountWeight",
            "BenchmarkWeight",
        ],
        "int8": [
            "FinancialFlag",
            "Eligibility144AFlag",
            "AnyIndexFlag",
            "USCreditReturnsFlag",
            "USCreditStatisticsFlag",
            "USAggReturnsFlag",
            "USAggStatisticsFlag",
            "USHYReturnsFlag",
            "USHYStatisticsFlag",
        ],
        "category": [
            "CUSIP",
            "ISIN",
            "Ticker",
            "Issuer",
            "RiskEntity",
            "Sector",
            "Subsector",
            "MoodyRating",
            "SPRating",
            "FitchRating",
            "CollateralType",
            "CouponType",
            "CallType",
            "Currency",
            "CountryOfRisk",
            "CountryOfDomicile",
            "MarketOfIssue",
            "Account",
        ],
    }
    # Build col:dtype dict and apply to input DataFrame.
    df_columns = set(df.columns)
    dtype_dict = {}
    for dtype, col_names in reverse_dtype_dict.items():
        for col in col_names:
            if col in df_columns:
                dtype_dict[col] = dtype
            else:
                continue
    return df.astype(dtype_dict)


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
    trade_dates: List[datetime].
        List of dates with bond data.
    """

    def __init__(self, server="LIVE"):
        # Load mapping from standard ratings to numeric values.
        self._ratings = load_json("ratings")
        self._loaded_dates = None
        self._bbg_dfs = {}
        n = 1 if server.upper() == "LIVE" else 2
        self._conn = pyodbc.connect(
            "Driver={SQL Server};"
            f"SERVER=XWNUSSQL0{n}\\{server.upper()};"
            "DATABASE=LGIMADatamart;"
            "Trusted_Connection=yes;"
        )

    @property
    @lru_cache(maxsize=None)
    def all_dates(self):
        """List[datetime]: Memoized list of all dates in DataBase."""
        return list(self._trade_date_df.index)

    @property
    @lru_cache(maxsize=None)
    def trade_dates(self):
        """List[datetime]: Memoized list of trade dates."""
        trade_dates = self._trade_date_df[self._trade_date_df["holiday"] == 0]
        return list(trade_dates.index)

    @property
    @lru_cache(maxsize=None)
    def holiday_dates(self):
        """List[datetime]: Memoized list of holiday dates."""
        holidays = self._trade_date_df[self._trade_date_df["holiday"] == 1]
        return list(holidays.index)

    @property
    @lru_cache(maxsize=None)
    def _trade_date_df(self):
        """pd.DataFrame: Memoized trade date boolean series for holidays."""
        return pd.read_csv(
            root("data/trade_dates.csv"),
            index_col=0,
            parse_dates=True,
            infer_datetime_format=True,
        )

    def date(self, date_delta, reference_date=None, **kwargs):
        """
        Find date relative to a specified date.

        Parameters
        ----------
        date_delta: str, ``{'today', 'WTD', 'MTD', 'YTD', '3d', '5y', etc}``
            Difference between reference date and target date.
        reference_date: datetime, optional
            Reference date to use, if None use most recent trade date.
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
        date_delta = date_delta.upper()
        # Use today as reference date if none is provided,
        # otherwise find closes trading date to reference date.
        if reference_date is None:
            today = self.trade_dates[-1]
        else:
            today = self.nearest_date(reference_date, **kwargs)

        last_trade = partial(bisect_left, self.trade_dates)
        if date_delta == "TODAY":
            return today
        elif date_delta == "YESTERDAY" or date_delta == "DAILY":
            return self.nearest_date(today, inclusive=False, after=False)
        elif date_delta == "WTD":
            return self.trade_dates[
                last_trade(today - timedelta(today.weekday() + 1)) - 1
            ]
        elif date_delta == "MTD":
            return self.trade_dates[last_trade(today.replace(day=1)) - 1]
        elif date_delta == "YTD":
            return self.trade_dates[
                last_trade(today.replace(month=1, day=1)) - 1
            ]
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
                return self.nearest_date(
                    today + relativedelta(**dt_kwargs), **kwargs
                )
            else:
                return self.nearest_date(
                    today - relativedelta(**dt_kwargs), **kwargs
                )

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
    def _manager_to_accounts(self):
        """Dict[str: str]: Respective accounts for each PM."""
        return load_json("manager_accounts")

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

    def display_all_columns(self):
        """Set DataFrames to display all columnns in IPython."""
        pd.set_option("display.max_columns", 999)

    def nearest_date(self, date, **kwargs):
        """
        Return trade date nearest to input date.

        Parameters
        ----------
        date: datetime object
            Input date.
        kwargs:
            Keyword arguments for :func:`nearest_date`.

        Returns
        -------
        datetime:
            Trade date nearest to input date.

        See Also
        --------
        :func:`nearest_date`
        """
        return nearest_date(date, self.trade_dates, **kwargs)

    def _convert_sectors_to_fin_flags(self, sectors):
        """
        Convert sectors to flag indicating if they
        are non-financial (0), financial (1), or other (2).

        Parameters
        ----------
        sectors: pd.Series
            Sectors

        Returns
        -------
        fin_flags: nd.array
            Array of values indicating if sectors are non-financial (0),
            financial (1), or other (Treasuries, Sovs, Govt owned).
        """

        financials = {
            "P&C",
            "LIFE",
            "APARTMENT_REITS",
            "BANKING",
            "BROKERAGE_ASSETMANAGERS_EXCHANGES",
            "RETAIL_REITS",
            "HEALTHCARE_REITS",
            "OTHER_REITS",
            "FINANCIAL_OTHER",
            "FINANCE_COMPANIES",
            "OFFICE_REITS",
        }
        other = {
            "TREASURIES",
            "SOVEREIGN",
            "SUPRANATIONAL",
            "INDUSTRIAL_OTHER",
            "GOVERNMENT_GUARANTEE",
            "OWNED_NO_GUARANTEE",
        }
        fin_flags = np.zeros(len(sectors))
        fin_flags[sectors.isin(financials)] = 1
        fin_flags[sectors.isin(other)] = 2
        return fin_flags

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
        dates = list(self._loaded_dates)[::-1]
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

    @staticmethod
    def _preprocess_market_data(df):
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
        col_map = {"IndustryClassification4": "Sector", "Price": "CleanPrice"}
        df.columns = [col_map.get(col, col) for col in df.columns]

        # Make flag columns -1 for nonexistent data to
        # mitigate memory usage by allwoing int dtype.
        df["Eligibility144AFlag"].fillna(-1, inplace=True)

        # Convert str time to datetime.
        for date_col in ["Date", "MaturityDate", "IssueDate", "NextCallDate"]:
            df[date_col] = pd.to_datetime(
                df[date_col], format="%Y-%m-%d", errors="coerce"
            )

        # Capitalize market of issue.
        df["MarketOfIssue"] = df["MarketOfIssue"].str.upper()

        # Capitalize all sectors and map sectors to standard values.
        df["Sector"] = df["Sector"].str.upper()
        df["Sector"] = df["Sector"].str.replace(" ", "_")
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

        # Add AAA rattings to treasury strips which have no ratings.
        cols = ["MoodyRating", "SPRating", "FitchRating"]
        strip_mask = df["Ticker"] == "SP"
        for col in cols:
            df.loc[strip_mask, col] = "AAA"

        # Fix treasury strip coupon to zero coupon.
        df.loc[strip_mask, "CouponType"] = "ZERO COUPON"

        # Fill NaNs for rating categories.
        df[cols] = df[cols].fillna("NR")
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
        ratings_mat = df[cols].fillna("NR").values

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
        coupontypes = {
            "FIXED",
            "VARIABLE",
            "STEP CPN",
            "FIXED PIK",
            "ZERO COUPON",
            "ADJUSTABLE",
            "HYBRID VARIABLE",
            "DEFAULTED",
        }
        calltypes = {"NONCALL", "MKWHOLE", "CALL/NR", "CALL/RF", "EUROCAL"}
        bad_sectors = {
            "STRANDED_UTILITY",
            "GOVERNMENT_SPONSORED",
            "CAR_LOAN",
            "NON_AGENCY_CMBS",
            "CREDIT_CARD",
            "AGENCY_CMBS",
            "MTG_NON_PFANDBRIEFE",
            "PFANDBRIEFE_TRADITIONAL_HYPOTHEKEN",
            "PFANDBRIEFE_TRADITIONAL_OEFFENLICHE",
            "PFANDBRIEFE_JUMBO_HYPOTHEKEN",
            "PFANDBRIEFE_JUMBO_OEFFENLICHE",
            "PSLOAN_NON_PFANDBRIEFE",
            "CONVENTIONAL_15_YR",
            "CONVENTIONAL_15_YR",
            "CONVENTIONAL_30_YR",
            "5/1_ARM",
            "7/1_ARM",
            "3/1_ARM",
            "GNMA_15_YR",
            "GNMA_30_YR",
            "NA",
        }
        bad_tickers = {"TVA", "FNMA", "FHLMC"}
        bad_pay_ranks = {"CERT OF DEPOSIT", "GOVT LIQUID GTD", "INSURED"}

        # Drop rows with the required columns.
        # TODO: add 'Issuer' back in
        required_cols = [
            "Date",
            "CUSIP",
            "Ticker",
            "CollateralType",
            "CouponRate",
            "MaturityDate",
            "CallType",
        ]
        df.dropna(subset=required_cols, how="any", inplace=True)

        # Get middle-or-lower numeric rating for each cusip.
        rating_cols = ["MoodyRating", "SPRating", "FitchRating"]
        df["NumericRating"] = self._get_numeric_ratings(df, rating_cols)

        # Define maturites and issues (yrs) and drop maturites above 150 yrs.
        day = "timedelta64[D]"
        df["MaturityYears"] = (df["MaturityDate"] - df["Date"]).astype(
            day
        ) / 365
        df["IssueYears"] = (df["Date"] - df["IssueDate"]).astype(day) / 365

        # Find columns to be kept.
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
        }
        keep_cols = [c for c in df.columns if c not in drop_cols]

        # Subset DataFrame by specified rules.
        df = df[
            (df["CouponType"].isin(coupontypes))
            & (df["CallType"].isin(calltypes))
            & (~df["Sector"].isin(bad_sectors))
            & (~df["Ticker"].isin(bad_tickers))
            & (~df["CollateralType"].isin(bad_pay_ranks))
            & (df["MaturityYears"] < 150)
            & (df["NumericRating"] <= 22)
            & (
                (df["CouponType"] != "VARIABLE")
                | ((df["MaturityDate"] - df["NextCallDate"]).astype(day) < 370)
            )
        ][keep_cols].copy()

        # Convert outstading from $ to $M.
        df["AmountOutstanding"] /= 1e6

        # Replace matuirities on bonds with variable coupon type and
        # maturity less call below 1 yr (e.g., 11NC10).
        # TODO: Decide if this is best way to deal with these bonds.
        df.loc[
            (df["CouponType"] == "VARIABLE")
            & ((df["MaturityDate"] - df["NextCallDate"]).astype(day) <= 370),
            "MaturityDate",
        ] = df["NextCallDate"]

        # Determine original maturity as int.
        # TODO: Think about modifying for a Long 10 which should be 10.5.
        df["OriginalMaturity"] = np.nan_to_num(
            np.round(df["MaturityYears"] + df["IssueYears"])
        ).astype(int)

        # Add financial flag column.
        df["FinancialFlag"] = self._convert_sectors_to_fin_flags(df["Sector"])

        # Add bloomberg subsector.
        df["Subsector"] = get_bloomberg_subsector(df["CUSIP"].values)

        # Calculate dirty price.
        df.eval("DirtyPrice = CleanPrice + AccruedInterest", inplace=True)

        # Use dirty price to calculate market value.
        df.eval(
            "MarketValue = AmountOutstanding * DirtyPrice / 100", inplace=True
        )

        # TEMP: Calculate HY Flags.
        hy_eliginble_countries = {
            "AU",
            "BE",
            "CA",
            "CH",
            "CY",
            "DE",
            "DK",
            "ES",
            "FI",
            "FR",
            "GB",
            "GR",
            "HK",
            "IE",
            "IT",
            "JE",
            "LU",
            "MO",
            "NL",
            "NO",
            "NZ",
            "PR",
            "SE",
            "SG",
            "US",
        }
        non_hy_sectors = [
            "OWNED_NO_GUARANTEE",
            "SOVEREIGN",
            "SUPRANATIONAL",
            "LOCAL_AUTHORITIES",
        ]
        df.loc[
            ((df["USHYStatisticsFlag"] == -1) | (df["USHYReturnsFlag"] == -1))
            & (df["NumericRating"] >= 11)
            & (df["NumericRating"] <= 21)
            & (df["Currency"] == "USD")
            & (~df["Sector"].isin(non_hy_sectors))
            & (df["AmountOutstanding"] >= 150)
            & (df["CountryOfRisk"].isin(hy_eliginble_countries))
            & (df["AnyIndexFlag"] == 1),
            ["USHYStatisticsFlag", "USHYReturnsFlag"],
        ] = 1
        df.loc[df["USHYStatisticsFlag"] == -1, "USHYStatisticsFlag"] = 0
        df.loc[df["USHYReturnsFlag"] == -1, "USHYReturnsFlag"] = 0

        # Make new fields categories and drop duplicates.
        return clean_dtypes(df).drop_duplicates(subset=["CUSIP", "Date"])

    def load_market_data(
        self,
        date=None,
        start=None,
        end=None,
        cusips=None,
        clean=True,
        ret_df=False,
        local=False,
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
        start: str, datetime, optional
            Starting date for scrape.
        end: str, datetime, optional
            Ending date for scrape.
        cusips: List[str], optional
            List of cusips to specify for the load, by default load all.
        clean: bool, default=True
            If true, apply standard cleaning rules to loaded DataFrame.
        ret_df: bool, default=False
            If True, return loaded DataFrame.
        local: bool, default=False
            Load index from local feather file.

        Returns
        -------
        df: pd.DataFrame
            DataFrame for specified date's index data if `ret_df` is true.
        """
        # Process dates.
        if date is not None:
            start = end = date

        # Store data if provided.
        if data is not None:
            self.df = data
            if ret_df:
                return self.df
            else:
                return

        # Load local feather files if specified.
        if local:
            # Find correct feather files to load.
            fmt = "%Y_%m.feather"
            s = self.trade_dates[-1] if start is None else to_datetime(start)
            start_month = pd.to_datetime(f"{s.month}/1/{s.year}")
            end = self.trade_dates[-1] if end is None else end
            feathers = pd.date_range(start_month, end, freq="MS").strftime(fmt)
            fids = [root(f"data/feathers/{f}") for f in feathers]

            # Load feather files as DataFrames and subset to correct dates.
            self.df = concat_index_dfs([pd.read_feather(f) for f in fids])
            self.df = self.df[
                (self.df["Date"] >= to_datetime(s))
                & (self.df["Date"] <= to_datetime(end))
            ]
            self._loaded_dates = list(pd.to_datetime(self.df["Date"].unique()))
            if ret_df:
                return self.df
            else:
                return

        # Use day count if start is integer.
        if isinstance(start, int):
            start = self.trade_dates[-start].strftime("%m/%d/%Y")
            end = start
        else:
            # Format input dates for SQL query.
            start = (
                None
                if start is None
                else to_datetime(start).strftime("%m/%d/%Y")
            )
            end = None if end is None else to_datetime(end).strftime("%m/%d/%Y")

        # Perform SQL query.
        yesterday = self.trade_dates[-1].strftime("%m/%d/%Y")
        sql = "exec [LGIMADatamart].[dbo].[sp_AFI_Get_SecurityAnalytics] {}"
        if start is None and end is None:
            query = sql.format(f"'{yesterday}', '{yesterday}'")
        elif start is not None and end is None:
            query = sql.format(f"'{start}', '{yesterday}'")
        else:
            query = sql.format(f"'{start}', '{end}'")
        # Add specific cusips if required.
        if cusips is not None:
            cusip_str = f", '{','.join(cusips)}'"
            query += cusip_str

        # Load data into a DataFrame form SQL in chunks to
        # reduce memory usage, cleaning them on the fly.

        df_chunk = pd.read_sql(query, self._conn, chunksize=50_000)
        chunk_list = []
        for chunk in df_chunk:
            # Preprocess chunk.
            chunk = self._preprocess_market_data(chunk)
            if clean:
                chunk = self._clean_market_data(chunk)
            chunk_list.append(chunk)
        if clean:
            try:
                sql_df = concat_index_dfs(chunk_list)
            except (IndexError, ValueError):
                raise ValueError("Empty Index, try selecting different dates.")
        else:
            try:
                sql_df = pd.concat(chunk_list, join="outer", sort=False)
            except (IndexError, ValueError):
                raise ValueError("Empty Index, try selecting different dates.")

        # Save unique loaded dates to memory.
        self._loaded_dates = list(pd.to_datetime(sql_df["Date"].unique()))

        # Standardize cusips over full period.
        if clean:
            self.df = self._standardize_cusips(sql_df)
        else:
            self.df = sql_df

        if ret_df:
            return self.df

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
        currency=None,
        cusip=None,
        isin=None,
        issuer=None,
        ticker=None,
        sector=None,
        subsector=None,
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
        liquidity_score=(None, None),
        in_stats_index=None,
        in_returns_index=None,
        in_agg_stats_index=None,
        in_agg_returns_index=None,
        in_hy_stats_index=None,
        in_hy_returns_index=None,
        in_any_index=None,
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
        currency: str, List[str], optional
            Currency or list of currencies to include.
        cusip: str, List[str]: optional,
            CUSIP or list of CUSIPs to include.
        isin: str, List[str]: optional,
            ISIN or list of ISINs to include.
        issuer: str, List[str], optional
            Issuer, or list of issuers to include.
        ticker: str, List[str], optional
            Ticker or list of tickers to include in index, default is all.
        sector: str, List[str], optional
            Sector or list of sectors to include in index.
        subsector: str, List[str], optional
            Subsector or list of subsectors to include in index.
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
        original_maturity: Tuple[float, float], default=(None, None).
            Range of original bond maturities to include.
        clean_price: Tuple[float, float]), default=(None, None).
            Clean price range of bonds to include, default is all.
        dirty_price: Tuple[float, float]), default=(None, None).
            Dirty price range of bonds to include, default is all.
        coupon_rate: Tuple[float, float]), default=(None, None).
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
        amount_outstanding: Tuple[float, float], default=(None, None).
            Range of amount outstanding to include in index (Millions).
        market_value: Tuple[float, float], default=(None, None).
            Range of market values to include in index (Millions).
        issue_years: Tuple[float, float], default=(None, None).
            Range of years since issue to include in index,
            default is all.
        collateral_type: str, List[str], optional
            Collateral type or list of types to include,
            default is all.
        yield_to_worst: Tuple[float, float], default=(None, None).
            Range of yields (to worst) to include, default is all.
        OAD: Tuple[float, float], default=(None, None).
            Range of option adjusted durations to include,
            default is all.
        OAS: Tuple[float, float], default=(None, None).
            Range of option adjusted spreads to include,
            default is all.
        OASD:  Tuple[float, float], default=(None, None).
            Range of option adjusted spread durations,
            default is all.
        liquidity_score: Tuple[float, float], default=(None, None).
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
            If True, only include bonds in any index.
            If False, only include bonds not in any index.
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
        elif isinstance(rating, str):
            if rating == "IG":
                rating = (1, 10)
            elif rating == "HY":
                rating = (11, 21)
            else:
                # Single rating value.
                rating = (self._ratings[rating], self._ratings[rating])
        else:
            rating = (self._ratings[rating[0]], self._ratings[rating[1]])

        # Convert all category constraints to lists.
        currency = to_set(currency, dtype=str)
        ticker = to_set(ticker, dtype=str)
        cusip = to_set(cusip, dtype=str)
        isin = to_set(isin, dtype=str)
        issuer = to_set(issuer, dtype=str)
        market_of_issue = to_set(market_of_issue, dtype=str)
        country_of_domicile = to_set(country_of_domicile, dtype=str)
        country_of_risk = to_set(country_of_risk, dtype=str)
        collateral_type = to_set(collateral_type, dtype=str)
        coupon_type = to_set(coupon_type, dtype=str)
        sector = to_set(sector, dtype=str)
        subsector = to_set(subsector, dtype=str)

        # Convert all flag constraints to int.
        in_returns_index = to_int(in_returns_index)
        in_stats_index = to_int(in_stats_index)
        in_agg_returns_index = to_int(in_agg_returns_index)
        in_agg_stats_index = to_int(in_agg_stats_index)
        in_hy_stats_index = to_int(in_hy_stats_index)
        in_hy_returns_index = to_int(in_hy_returns_index)
        in_any_index = to_int(in_any_index)
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
            "market_of_issue": ("MarketOfIssue", market_of_issue),
            "country_of_domicile": ("CountryOfDomicile", country_of_domicile),
            "country_of_risk": ("CountryOfRisk", country_of_risk),
            "collateral_type": ("CollateralType", collateral_type),
            "coupon_type": ("CouponType", coupon_type),
            "financial_flag": ("FinancialFlag", financial_flag),
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
            "amount_outstanding": ("AmountOutstanding", amount_outstanding),
            "market_value": ("MarketValue", market_value),
            "yield_to_worst": ("YieldToWorst", yield_to_worst),
            "OAD": ("OAD", OAD),
            "OAS": ("OAS", OAS),
            "OASD": ("OASD", OASD),
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
            subset_mask_list.append(replace_multiple(rule, repl_dict))

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

        return Index(df, name, index_constraints)

    @staticmethod
    def _preprocess_portfolio_data(df):
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
        # Fix bad column names.
        col_map = {
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
        }
        df.columns = [col_map.get(col, col) for col in df.columns]
        df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
        return clean_dtypes(df)

    def load_portfolio(
        self,
        date=None,
        start=None,
        end=None,
        accounts=None,
        strategy=None,
        manager=None,
        universe="returns",
        drop_cash=True,
        drop_treasuries=True,
        market_cols=None,
    ):
        """
        Load market data from SQL server. If end is not specified
        data is scraped through previous day. If neither start nor
        end are given only the data from previous day is scraped.
        Optionally load from local compressed format for increased
        performance or feed a DataFrame directly.
        """
        # Format dates for SQL call.
        fmt = "%Y%m%d"
        current_date = self.trade_dates[-1].strftime(fmt)
        if date is not None:
            start = end = date
        start = (
            current_date if start is None else to_datetime(start).strftime(fmt)
        )
        end = current_date if end is None else to_datetime(end).strftime(fmt)

        # Convert inputs for SQL call.
        if accounts is None:
            accounts = "NULL"
        else:
            acnts = to_list(accounts, dtype=str)
            accounts = f"'{','.join(acnts) if len(acnts) > 1 else acnts[0]}'"

        strategy = "NULL" if strategy is None else f"'{strategy}'"
        manager = "NULL" if manager is None else f"'{manager}'"
        universe = {"stats": "statistics"}.get(universe.lower(), universe)
        universe = f"'{universe.title()}'"
        inputs = [
            start,
            strategy,
            accounts,
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
        # bm_df = pd.read_sql(sql_base + sql_benchmark, self._conn)
        # p_df = pd.read_sql(sql_base + sql_portfolio, self._conn)
        df = pd.read_sql(sql_base + sql_both, self._conn)
        df = self._preprocess_portfolio_data(df)
        if not len(df):
            raise ValueError("No data for specified date.")

        # Fix dtypes.
        df["Date"] = pd.to_datetime(df["Date"])

        # Drop cash and treasuries if needed.
        bad_sectors = []
        if drop_cash:
            bad_sectors.append("CASH")
        if drop_treasuries:
            bad_sectors.append("TREASURIES")
        if bad_sectors:
            df = df[~df["Sector"].isin(set(bad_sectors))].copy()

        if market_cols is None:
            return df.sort_values(["Date", "Ticker"])

        else:
            # Get market data, loading if required.
            if (
                self._loaded_dates is None
                or to_datetime(start) < self._loaded_dates[0]
                or to_datetime(end) > self._loaded_dates[-1]
            ):
                self.load_market_data(start=start, end=end, local=True)
            market_df = self.build_market_index(start=start, end=end).df

            # Subset columns to append to portfolio data.
            if market_cols is True:
                market_cols = list(set(market_df) - set(df))
            else:
                market_cols = to_list(market_cols, dtype=str)
            ix = ["Date", "CUSIP"]
            market_cols.extend(ix)

            # Combine market data to portfolio data.
            return (
                df.set_index(ix)
                .join(market_df[market_cols].set_index(ix), on=ix)
                .reset_index()
                .sort_values(["Date", "Ticker"])
            )

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
        self, securities, field, start=None, end=None, nan=None, **kwargs
    ):
        """
        Load Bloomberg BDH data for given securities.

        Parameters
        ----------
        securities: str or List[str].
            Security or secuirites to load data for.
        field: str, ``{'OAS', 'Price', 'TRET', 'XSRET', 'YTW', etc.}``
            Field of data to load.
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
        field = field.upper()
        if securities == "all":
            df = self._read_bbg_df(field.upper).copy()
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
