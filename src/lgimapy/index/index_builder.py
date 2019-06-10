import json
import pickle
import warnings
from functools import lru_cache

import datetime as dt
import feather
import numpy as np
import pandas as pd
import pyodbc

from lgimapy.bloomberg import get_bloomberg_subsector
from lgimapy.index import Index
from lgimapy.utils import replace_multiple, root, Time


# %%
class IndexBuilder:
    """
    Class to pull and clean index data from SQL database.

    Attributes
    ----------
    df: pd.DataFrame
        Full DataFrame of all cusips over loaded period.
    trade_dates: list[datetime].
        List of dates with bond data.
    loaded_dates: list[datetime].
        List of dates currently loaded by builder.
    """

    def __init__(self):
        # Load mapping from standard ratings to numeric values.
        with open(root("data/ratings.json"), "rb") as fid:
            self._ratings = json.load(fid)

    @property
    @lru_cache(maxsize=None)
    def trade_dates(self):
        """Datetime index array of dates with credit data."""
        dates_sql = "select distinct effectivedatekey from \
            dbo.InstrumentAnalytics order by effectivedatekey"
        conn = pyodbc.connect(
            "Driver={SQL Server};"
            "SERVER=XWNUSSQL01\\LIVE;"
            "DATABASE=LGIMADatamart;"
            "Trusted_Connection=yes;"
        )
        return list(
            pd.to_datetime(
                pd.read_sql(dates_sql, conn).values.ravel(), format="%Y%m%d"
            )
        )

    def nearest_date(self, date):
        """
        Return trade date nearest to input date.

        Parameters
        ----------
        date: datetime object
            Input date.

        Returns
        -------
        t_date: datetime object
            Trade date nearest to input date.
        """
        return min(self.trade_dates, key=lambda x: abs(x - date))

    def _convert_sectors_to_fin_flags(self, sectors):
        """
        Convert list of sectors to a financial flag of 'financial',
        'non-financial', or 'other'.

        Parameters
        ----------
        sectors: List[str].
            List of sectors from index DataFrame.

        Returns
        -------
        fin_flags: List[str].
            List of {'financial', 'non-financial', 'other'} for
            each input sector.
        """

        financials = [
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
        ]

        other = [
            "TREASURIES" "SOVEREIGN",
            "SUPRANATIONAL",
            "INDUSTRIAL_OTHER",
            "GOVERNMENT_GUARANTEE",
            "OWNED_NO_GUARANTEE",
        ]

        fin_flags = []
        for sector in sectors:
            if sector in financials:
                fin_flags.append("financial")
            elif sector in other:
                fin_flags.append("other")
            else:
                fin_flags.append("non-financial")

        return fin_flags

    def _standardize_cusips(self, df):
        """
        Standardize CUSIPs, converting cusips which changed name
        to most recent cusip value for full history.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame of selected index cusips.

        Returns
        -------
        df: pd.DataFrame
            DataFrame with all CUSIPs updated to current values.
        """
        dates = list(self.loaded_dates)[::-1]
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
    def _convert_dtypes(df):
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
        df: pd.DataFrame
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
        rating_cols = ["MoodyRating", "SPRating", "FitchRating"]
        strip_mask = df["Ticker"] == "SP"
        for col in rating_cols:
            df.loc[strip_mask, col] = "AAA"

        # Fix treasury strip coupon to zero coupon.
        df.loc[strip_mask, "CouponType"] = "ZERO COUPON"

        # Fill NaNs for rating categories.
        df[rating_cols] = df[rating_cols].fillna("NR")

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
                "LQA",
                "KRD06mo",
                "KRD02yr",
                "KRD05yr",
                "KRD10yr",
                "KRD20yr",
                "KRD30yr",
                "Vega",
                "Delta",
                "Gamma",
                "Theta",
                "Rho",
                "OAC",
            ],
            "int8": [
                "Eligibility144AFlag",
                "USCreditReturnsFlag",
                "USCreditStatisticsFlag",
                "AnyIndexFlag",
                "USAggReturnsFlag",
                "USAggStatisticsFlag",
            ],
            "category": [
                "CUSIP",
                "Ticker",
                "Issuer",
                "Sector",
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
            ],
        }
        # Build col:dtype dict and apply to input DataFrame.
        dtype_dict = {}
        for dtype, col_names in reverse_dtype_dict.items():
            for col in col_names:
                dtype_dict[col] = dtype
        return df.astype(dtype_dict)

    def _clean(self, df):
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
        coupontypes = [
            "FIXED",
            "VARIABLE",
            "STEP CPN",
            "FIXED PIK" "ZERO COUPON",
            "ADJUSTABLE",
            "HYBRID VARIABLE",
            "DEFAULTED",
        ]
        calltypes = ["NONCALL", "MKWHOLE", "CALL/NR", "CALL/RF", "EUROCAL"]
        bad_sectors = [
            "STRANDED_UTILITY",
            "GOVERNMENT_SPONSORED",
            "CAR_LOAN",
            "NON_AGENCY_CMBS",
            "CREDIT_CARD",
            "AGENCY_CMBS",
            "MTG_NON_PFANDBRIEFE",
            "PFANDBRIEFE_TRADITIONAL_HYPOTHEKEN",
            "PFANDBRIEFE_TRADITIONAL_OEFFENLICHE",
            "CONVENTIONAL_15_YR",
            "CONVENTIONAL_15_YR",
            "CONVENTIONAL_30_YR",
            "5/1_ARM",
            "7/1_ARM",
            "3/1_ARM",
            "GNMA_15_YR",
            "GNMA_30_YR",
            "NA",
        ]
        bad_tickers = ["TVA", "FNMA", "FHLMC"]
        bad_pay_ranks = ["CERT OF DEPOSIT", "GOVT LIQUID GTD", "INSURED"]

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

        # Make temporary matrix of numeric ratings.
        rating_cols = ["MoodyRating", "SPRating", "FitchRating"]
        ratings_mat = df[rating_cols].fillna("NR").values
        num_ratings = np.vectorize(self._ratings.__getitem__)(
            ratings_mat
        ).astype(float)
        num_ratings[num_ratings == 0] = np.nan  # json nan value is 0

        # Vectorized implementation of middle-or-lower.
        warnings.simplefilter(action="ignore", category=RuntimeWarning)
        nans = np.sum(np.isnan(num_ratings), axis=1)
        MOL = np.median(num_ratings, axis=1)  # middle
        lower = np.nanmax(num_ratings, axis=1)  # max for lower
        lower_mask = (nans == 1) | (nans == 2)
        MOL[lower_mask] = lower[lower_mask]
        df["NumericRating"] = MOL
        warnings.simplefilter(action="default", category=RuntimeWarning)

        # Define maturites and issues (yrs) and drop maturites above 150 yrs.
        day = "timedelta64[D]"
        df["MaturityYears"] = (df["MaturityDate"] - df["Date"]).astype(
            day
        ) / 365
        df["IssueYears"] = (df["Date"] - df["IssueDate"]).astype(day) / 365

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
        ].copy()

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
        df["DirtyPrice"] = df["CleanPrice"] + df["AccruedInterest"]

        # Make new fields categories.
        df = df.astype(
            {c: "category" for c in ["FinancialFlag", "Subsector", "CUSIP"]}
        )

        # Drop duplicates.
        df.drop_duplicates(subset=["CUSIP", "Date"], inplace=True)
        return df

    def load(
        self,
        start=None,
        end=None,
        cusips=None,
        clean=True,
        dev=False,
        ret_df=False,
        local=False,
        data=None,
    ):
        """
        Load data from SQL server. If end is not specified data is scraped
        through previous day. If neither start nor end are given only the
        data from previous day is scraped. Optionally load from local
        compressed format for increased performance or feed a DataFrame
        directly.

        Parameters
        ----------
        start: str, datetime, default=None
            Starting date for scrape.
        end: str, datetime, default=None
            Ending date for scrape.
        cusips: List[str], default=None
            List of cusips to specify for the load, by default load all.
        clean: bool, default=True
            If true, apply standard cleaning rules to loaded DataFrame.
        dev: bool, default=False
            If True, use development SQL query to load all fields.
            **DO NOT USE IN PRODUCTION CODE**
        ret_df: bool, default=False
            If True, return loaded DataFrame.
        local: bool, default=False
            Load index from local binary file.

        Returns
        -------
        df: pd.DataFrame
            DataFrame for specified date's index data if `ret_df` is true.
        """
        # Store data if provided.
        if data is not None:
            self.df = data
            if ret_df:
                return self.df
            else:
                return

        # Load local feather if
        if local:
            fid = root("data/ixb_feathers/full.feather")
            self.df = feather.read_dataframe(fid)
            if ret_df:
                return self.df
            else:
                return

        if dev:
            # Development SQL call.
            # **DO NOT USE IN PRODUCTION CODE**
            sql = """\
                DECLARE @RunDateBegin DATE = {};\
                DECLARE @RunDateEnd DATE = ISNULL({}, @RunDateBegin);\
            	DECLARE @CUSIPS VARCHAR(8000) = \
                NULLIF(LTRIM(RTRIM({})), '');

                SELECT\
                      D.[Date],\
                      I.CUSIP,\
                      Ticker = COALESCE(I.BBTicker, I.FutureTicker, IA.Ticker),\
                      I.Issuer,\
                      CouponRate = ISNULL(I.CouponRate, IA.Coupon),\
                      MaturityDate = ISNULL(I.MaturityDate, IA.MaturityDate),\
                      I.IssueDate,\
                      IndustryClassification4 = \
                        dbo.GetInstrumentClassificationLadder(\
                            I.SourceKey,\
                            I.IndustryClassification4,\
                            IA.Classification4,\
                            NULL,\
                            I.ISIN\
                            ),\
                      I.MoodyRating,\
                      I.SPRating,\
                      I.FitchRating,\
                      I.CollateralType,\
                      I.AmountOutstanding,\
                      I.MarketOfIssue,\
                      I.NextCallDate,\
                      I.CouponType,\
                      CallType = ISNULL(I.CallType,P.CallType),\
                      IA.Price,\
                      IA.OAD,\
                      IA.OAS,\
                      IA.OASD,\
                      OAS_1W = IA.OAS1WChange,\
                      OAS_1M = IA.OAS1MChange,\
                      OAS_3M = IA.OAS3MChange,\
                      OAS_6M = IA.OAS6MChange,\
                      OAS_12M = IA.OAS12MChange,\
                      ISNULL(IA.LiquidityCostScore,IAP.LiquidityScore) \
                        as LiquidityCostScore,\
                      I.Eligibility144AFlag,\
                      Currency = I.ISOCurrency,\
                      I.CountryOfRisk,\
                      I.CountryOfDomicile,\
                      CASE IAP.CustomUSCreditFlag\
                            WHEN 'US CREDIT - Both' THEN 1\
                            WHEN 'US CREDIT - Returns' THEN 1\
                            WHEN 'US CREDIT - Statistics' THEN 0\
                            ELSE -1\
                      END AS USCreditReturnsFlag,\
                      CASE IAP.CustomUSCreditFlag \
                            WHEN 'US CREDIT - Both' THEN 1\
                            WHEN 'US CREDIT - Statistics' THEN 1\
                            WHEN 'US CREDIT - Returns' THEN 0\
                            ELSE -1\
                      END AS USCreditStatisticsFlag,\
                      CASE IAP.AnyIndexFlag\
                            WHEN 'YES' THEN 1\
                            WHEN 'NO' THEN 0\
                            ELSE -1\
                      END AS AnyIndexFlag,\
                      CASE IAP.IdxFlagUSAgg\
                            WHEN 'BOTH_IND' THEN 1\
                            WHEN 'BACKWARD' THEN 1\
                            WHEN 'FORWARD' THEN 0\
                            ELSE -1\
                      END AS USAggReturnsFlag,\
                      CASE IAP.IdxFlagUSAgg\
                            WHEN 'BOTH_IND' THEN 1\
                            WHEN 'FORWARD' THEN 1\
                            WHEN 'BACKWARD' THEN 0\
                            ELSE -1\
                      END AS USAggStatisticsFlag,\
                      IA.AccruedInterest,\
                      IA.LiquidityCostScore as LQA,\
                      IA.KRD06mo,\
                      IA.KRD02yr,\
                      IA.KRD05yr,\
                      IA.KRD10yr,\
                      IA.KRD20yr,\
                      IA.KRD30yr,\
                      IA.Vega,\
                      IA.Delta,\
                      IA.Gamma,\
                      IA.Theta,\
                      IA.Rho,\
                      IA.OAC\
                FROM DimDate AS D WITH (NOLOCK)\
                INNER JOIN InstrumentAnalytics AS IA WITH (NOLOCK)\
                      ON    IA.EffectiveDateKey = D.DateKey\
                      AND IA.SourceKey in (1,5)\
                INNER JOIN DimInstrument AS I WITH (NOLOCK)\
                      ON    I.InstrumentKey = IA.InstrumentKey\
                LEFT JOIN Staging.CallType_Bloomberg_PORT AS P WITH (NOLOCK)\
                      ON    I.Cusip = P.Cusip\
                LEFT JOIN Staging.InstrumentAnalytics_Port AS IAP WITH (NOLOCK)\
                      ON    I.Cusip = IAP.Cusip\
                      AND D.[Date] = CONVERT(DATE,IAP.EffectiveDate)\
                WHERE D.[Date] BETWEEN @RunDateBegin AND @RunDateEnd\
                      AND   IA.Price IS NOT NULL\
                      {}
                ORDER BY 1, 2, 3, 4, 5;\
                """
        else:
            sql = "exec [LGIMADatamart].[dbo].[sp_AFI_Get_SecurityAnalytics] {}"

        # Use day count if start is integer.
        if isinstance(start, int):
            start = self.trade_dates[-start].strftime("%m/%d/%Y")
            end = start
        else:
            # Format input dates for SQL query.
            start = (
                None
                if start is None
                else pd.to_datetime(start).strftime("%m/%d/%Y")
            )
            end = (
                None
                if end is None
                else pd.to_datetime(end).strftime("%m/%d/%Y")
            )

        # Perform SQL query.
        yesterday = self.trade_dates[-1].strftime("%m/%d/%Y")
        if dev:
            if cusips is None:
                cusip_str1 = "''"
                cusip_str2 = "AND I.CUSIP IS NOT NULL"
            else:
                cusip_str1 = f"'{','.join(cusips)}'"
                cusip_str2 = """
                    AND I.CUSIP IN \
                        (SELECT ParamItem FROM \
                        dbo.SplitDelimitedString(@CUSIPS, ','))\
                    """
            if start is None and end is None:
                query = sql.format(
                    f"'{yesterday}'", "NULL", cusip_str1, cusip_str2
                )
            elif start is not None and end is None:
                query = sql.format(
                    f"'{start}'", f"'{yesterday}'", cusip_str1, cusip_str2
                )
            else:
                query = sql.format(
                    f"'{start}'", f"'{end}'", cusip_str1, cusip_str2
                )

        else:
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
        conn = pyodbc.connect(
            "Driver={SQL Server};"
            "SERVER=XWNUSSQL01\\LIVE;"
            "DATABASE=LGIMADatamart;"
            "Trusted_Connection=yes;"
        )
        df_chunk = pd.read_sql(query, conn, chunksize=50_000)
        chunk_list = []
        for chunk in df_chunk:
            # Preprocess chunk.
            chunk = self._convert_dtypes(chunk)
            if clean:
                chunk = self._clean(chunk)
            chunk_list.append(chunk)
        sql_df = pd.concat(chunk_list, ignore_index=True)

        # Verify data was scraped.
        if len(sql_df) == 0:
            print("Warning: Scraped Index DataFrame is Empty.")

        # Save unique loaded dates to memory.
        self.loaded_dates = sorted(list(set(sql_df["Date"])))

        # Standardize cusips over full period.
        if clean:
            self.df = self._standardize_cusips(sql_df)
        else:
            self.df = sql_df

        if ret_df:
            return self.df

    def _add_str_list_input(self, input_val, col_name):
        """
        Add inputs from `build()` function with type of either str or
        list[str] to hash table to use when subsetting full DataFrame.

        Parameters
        ----------
        input_val: str, list[str].
            Input variable from `build()`.
        col_nam: str
            Column name in full DataFrame.
        """
        if input_val is not None:
            self._all_rules.append(col_name)
            if isinstance(input_val, str):
                self._str_list_vals[col_name] = [input_val]
            else:
                self._str_list_vals[col_name] = input_val

    def _add_range_input(self, input_val, col_name):
        """
        Add inputs from `build()` function with type tuple of ranged float
        values to hash table to use when subsetting full DataFrame.
        `-np.infty` and `np.infty` can be used to drop rows with NaNs.

        Parameters
        ----------
        input_val: Tuple[float, float].
            Input variable from `build()`.
        col_nam: str
            Column name in full DataFrame.
        """
        if col_name == "MaturityYears" and isinstance(input_val, int):
            low = {5: 4, 10: 6, 20: 11, 30: 25}
            high = {5: 6, 10: 11, 20: 25, 30: 31}
            self._range_vals[col_name] = (low[input_val], high[input_val])
            self._all_rules.append(col_name)
        else:
            i0, i1 = input_val[0], input_val[1]
            if i0 is not None and i1 is not None:
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
        Add inputs from `build()` function with flag type to
        hash table to use when subsetting full DataFrame.

        Parameters
        ----------
        input_val: bool.
            Input variable from `build()`.
        col_nam: str
            Column name in full DataFrame.
        """
        if input_val:
            self._all_rules.append(col_name)
            self._flags.append(col_name)

    def build(
        self,
        name="",
        start=None,
        end=None,
        rating="IG",
        currency=None,
        cusip=None,
        issuer=None,
        ticker=None,
        sector=None,
        subsector=None,
        treasuries=False,
        municipals=True,
        maturity=(None, None),
        price=(None, None),
        country_of_domicile=None,
        country_of_risk=None,
        amount_outstanding=(300, None),
        issue_years=(None, None),
        collateral_type=None,
        OAD=(None, None),
        OAS=(None, None),
        OASD=(None, None),
        liquidity_score=(None, None),
        credit_stats_only=False,
        credit_returns_only=False,
        financial_flag=None,
        special_rules=None,
    ):
        """
        Build index with customized rules from :attr:`IndexBuilder.df`.

        Parameters
        ----------
        name: str, default=''
            Optional name for returned index.
        start: datetime, default=None
            Start date for index, if None the start date from load is used.
        end: datetime, default=None
            End date for index, if None the end date from load is used.
        rating: str , Tuple[str, str], default='IG'
            Bond rating/rating range for index.

            Examples:

            * str: 'HY', 'IG', 'AAA', 'Aa1', etc.
            * Tuple[str, str]: (AAA, BB) uses all bonds in
              specified inclusive range.
        currency: str, List[str], default=None
            Currency or list of currencies to include.
        cusip: str, List[str]: default=None,
            Cusip or list of cusips to include.
        issuer: str, List[str], default=None
            Issuer, or list of issuers to include.
        ticker: str, List[str], default=None
            Ticker or list of tickers to include in index, default is all.
        sector: str, List[str], default=None
            Sector or list of sectors to include in index.
        subsector: str, List[str], default=None
            Subsector or list of subsectors to include in index.
        treasuries: bool, default=False
            If true return only treasures in index, else exclude.
        municipals: bool, default=True
            If False, remove municipal bonds from index.
        maturity: Tuple[float, float], {5, 10, 20, 30}, default=None
            Maturities to include, if int is specified the following ranges
            are used:

            * 5: 4-6
            * 10: 6-11
            * 20: 11-25
            * 30: 25 - 31
        price: Tuple[float, float]), default=(None, None).
            Price range of bonds to include, default is all.
        country_of_domicile: str, List[str], default=None
            Country or list of countries of domicile to include, default is all.
        country_of_risk: str, List[str], default=None
            Country or list of countries wherer risk is centered to include
            in index, default is all.
        amount_outstanding: Tuple[float, float], default=(300, None).
            Range of amount outstanding to include in index (Millions).
        issue_years: Tuple[float, float], default=(None, None).
            Range of years since issue to include in index, default is all.
        collateral_type: str, List[str], default=None
            Collateral type or list of types to include, default is all.
        OAD: Tuple[float, float], default=(None, None).
            Range of option adjusted durations to include, default is all.
        OAS: Tuple[float, float], default=(None, None).
            Range of option adjusted spreads to include, default is all.
        OASD:  Tuple[float, float], default=(None, None).
            Range of option adjusted spread durations, default is all.
        liquidity_score: Tuple[float, float], default=(None, None).
            Range of liquidty scores to use, default is all.
        credit_stats_only: bool, default=False
            If True, only include bonds with credit stats in index.
        credit_returns_only: bool, default=False
            If True, only include bonds with credit returns in index.
        financial_flag: {'financial', 'non-financial', 'other'}, default=None
            Financial flag setting to identify fin and non-fin credits.
        special_rules: str, List[str] default=None
            Special rule(s) for subsetting index using bitwise operators.
            If None, all specified inputs are applied independtently of
            eachother as bitwise &. All rules can be stacked using paranthesis
            to create more complex rules.

            Examples:

            * Include specified sectors or subsectors:
              'Sector | Subsector'
            * Include all but specified sectors:
              '~Sector'
            * Include either (all but specified currency or specified
              sectors) xor specified maturities:
              '(~Currnecy | Sector) ^ MaturityYears'

        Returns
        -------
        :class:`Index`:
            :class:`Index` with specified rules.
        """
        # convert start and end dates to datetime.
        start = None if start is None else pd.to_datetime(start)
        end = None if end is None else pd.to_datetime(end)

        # Convert rating to range of inclusive ratings.
        if isinstance(rating, str):
            if rating == "IG":
                ratings = (1, 10)
            elif rating == "HY":
                ratings = (11, 21)
            else:
                # Single rating value.
                ratings = (self._ratings[rating], self._ratings[rating])
        else:
            ratings = (self._ratings[rating[0]], self._ratings[rating[1]])

        # TODO: Modify price/amount outstading s.t. they account for currency.
        # Make dict of values for all str inputs.
        self._all_rules = []
        self._str_list_vals = {}
        self._add_str_list_input(currency, "Currency")
        self._add_str_list_input(ticker, "Ticker")
        self._add_str_list_input(cusip, "CUSIP")
        self._add_str_list_input(issuer, "Issuer")
        self._add_str_list_input(country_of_domicile, "CountryOfDomicile")
        self._add_str_list_input(country_of_risk, "CountryOfRisk")
        self._add_str_list_input(collateral_type, "CollateralType")
        self._add_str_list_input(financial_flag, "FinancialFlag")
        self._add_str_list_input(subsector, "Subsector")

        if sector in ["financial", "non-financial", "other"]:
            self._add_str_list_input(sector, "FinancialFlag")
        else:
            self._add_str_list_input(sector, "Sector")

        # Make dict of values for all tuple float range inputs.
        self._range_vals = {}
        self._add_range_input((start, end), "Date")
        self._add_range_input(ratings, "NumericRating")
        self._add_range_input(maturity, "MaturityYears")
        self._add_range_input(price, "DirtyPrice")
        self._add_range_input(amount_outstanding, "AmountOutstanding")
        self._add_range_input(issue_years, "IssueYears")
        self._add_range_input(liquidity_score, "LiquidityCostScore")
        self._add_range_input(OAD, "OAD")
        self._add_range_input(OAS, "OAS")
        self._add_range_input(OASD, "OASD")

        # Set values for including bonds based on credit flags.
        self._flags = []
        self._add_flag_input(credit_returns_only, "USCreditReturnsFlag")
        self._add_flag_input(credit_stats_only, "USCreditStatisticsFlag")

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
        # :attr:`IndexBuilder.df` simulatenously in order to
        # avoid re-writing the DataFrame into memory after
        # each individual mask.
        range_val_repl = {
            key: (
                f'(self.df["{key}"] >= self._range_vals["{key}"][0]) & '
                f'(self.df["{key}"] <= self._range_vals["{key}"][1])'
            )
            for key in self._range_vals.keys()
        }
        str_val_repl = {
            key: f'(self.df["{key}"].isin(self._str_list_vals["{key}"]))'
            for key in self._str_list_vals.keys()
        }
        flag_val_repl = {key: f'(self.df["{key}"] == 1)' for key in self._flags}

        # Format special rules.
        subset_mask_list = []
        if special_rules:
            for rule in special_rules:
                subset_mask_list.append(
                    replace_multiple(
                        rule,
                        {
                            **range_val_repl,
                            **str_val_repl,
                            **flag_val_repl,
                            "~(": "(~",
                        },
                    )
                )
        # Add treasury and muncipal rules.
        if treasuries:
            subset_mask_list.append('(self.df["Sector"]=="TREASURIES")')
        else:
            subset_mask_list.append('(self.df["Sector"]!="TREASURIES")')
        if not municipals:
            subset_mask_list.append('(self.df["Sector"]!="LOCAL_AUTHORITIES")')
        # Format all other rules.
        for rule in self._all_rules:
            if rule in rule_cols:
                continue  # already added to subset mask
            subset_mask_list.append(
                replace_multiple(
                    rule,
                    {
                        **range_val_repl,
                        **str_val_repl,
                        **flag_val_repl,
                        "~(": "(~",
                    },
                )
            )
        # Combin formatting rules into single mask and subset DataFrame.
        subset_mask = " & ".join(subset_mask_list)
        return Index(eval(f"self.df.loc[{subset_mask}]"), name)


# %%
def main():
    subset_mask_list
    maturity = (None, None)
    rating = "IG"
    sector = "AIRLINES"
    special_rules = "~Sector"
    with Time() as t:
        self = IndexBuilder()
        # start = '12/31/2003'
        start = "5/31/2019"
        end = None
        # df = ixb.load(start, end, dev=True, ret_df=True)
        self.load(start, end, dev=True)

    with Time() as t:
        ix = self.build(
            rating=("AAA", "BB"), sector="AIRLINES", special_rules="~Sector"
        )

    ix.df.head(60)
    sorted(list(set(ix.df.Sector)))


if __name__ == "__main__":
    main()
