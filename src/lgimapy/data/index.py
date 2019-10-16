import warnings
from bisect import bisect_left
from collections import defaultdict
from functools import lru_cache

import numpy as np
import pandas as pd

from lgimapy.bloomberg import get_bloomberg_ticker
from lgimapy.data import Bond, concat_index_dfs, new_issue_mask, TreasuryCurve
from lgimapy.utils import check_all_equal, load_json, replace_multiple, root

# %%


class Index:
    """
    Class for indexes built by :class:`Database`.

    Parameters
    ----------
    index_df: pd.DataForame
        Index DataFrame from :meth:`Database.build_market_index`.
    name: str, default=''
        Optional name of index.

    Attributes
    ----------
    df: pd.DataFrame
        Full Index DataFrame.
    """

    def __init__(self, index_df, name=""):
        self.df = index_df.set_index("CUSIP", drop=False)
        self.name = name
        # Initialize cache for storing daily DataFrames.
        self._day_cache = {}
        self._day_cache_key = "_".join(self.df.columns)

    def __repr__(self):
        start = self.dates[0].strftime("%m/%d/%Y")
        end = self.dates[-1].strftime("%m/%d/%Y")
        if start == end:
            return f"{self.name} Index {start}"
        else:
            return f"{self.name} Index {start} - {end}"

    def __add__(self, other):
        """Combine mutltiple instances of :class:`Index` together."""
        if isinstance(other, Index):
            return Index(concat_index_dfs([self.df, other.df]))
        else:
            raise TypeError(f"Right operand must be an {type(ix).__name__}.")

    def __radd__(self, other):
        """Combine mutltiple instances of :class:`Index` together."""
        if isinstance(other, Index):
            return Index(concat_index_dfs([self.df, other.df]))
        elif other == 0:
            return self
        else:
            raise TypeError(f"Left operand must be an {type(ix).__name__}.")

    def __iadd__(self, other):
        """Combine mutltiple instances of :class:`Index` together."""
        if isinstance(other, Index):
            return Index(concat_index_dfs([self.df, other.df]))
        else:
            raise TypeError(f"Right operand must be an {type(ix).__name__}.")

    @property
    @lru_cache(maxsize=None)
    def dates(self):
        """List[datetime]: Memoized unique sorted dates in index."""
        dates = list(pd.to_datetime(self.df["Date"].unique()))
        return [d for d in dates if d not in self._holiday_dates]

    @property
    @lru_cache(maxsize=None)
    def cusips(self):
        """List[str]: Memoized unique cusips in index."""
        return list(self.df.index.unique())

    @property
    @lru_cache(maxsize=None)
    def sectors(self):
        """List[str]: Memoized unique sorted sectors in index."""
        return sorted(list(self.df["Sector"].unique()))

    @property
    @lru_cache(maxsize=None)
    def subsectors(self):
        """List[str]: Memoized unique sorted subsectors in index."""
        return sorted(list(self.df["Subsector"].unique()))

    @property
    @lru_cache(maxsize=None)
    def issuers(self):
        """List[str]: Memoized unique sorted issuers in index."""
        return sorted(list(self.df["Issuer"].unique()))

    @property
    @lru_cache(maxsize=None)
    def tickers(self):
        """List[str]: Memoized unique sorted tickers in index."""
        return sorted(list(self.df["Ticker"].unique()))

    @property
    def bonds(self):
        """List[:class:`Bond`]: List of individual bonds in index."""
        return [Bond(bond) for _, bond in self.df.iterrows()]

    @property
    @lru_cache(maxsize=None)
    def all_trade_dates(self):
        """List[datetime]: Memoized list of trade dates."""
        return list(self._trade_date_df.index)

    @property
    @lru_cache(maxsize=None)
    def _holiday_dates(self):
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

    @property
    @lru_cache(maxsize=None)
    def _ratings(self):
        """Dict[str: int]: Ratings map from letters to numeric."""
        return load_json("ratings")

    def copy(self):
        """Create copy of current :class:`Index`."""
        return Index(self.df.copy())

    def total_value(self, synthetic=False):
        """float: Total value of index in $M."""
        dates = self.dates[1:] if synthetic else self.dates
        a = np.zeros(len(dates))
        for i, date in enumerate(dates):
            df = self.synthetic_day(date) if synthetic else self.day(date)
            a[i] = np.sum(df["MarketValue"])
        return pd.Series(a, index=dates, name="total_value")

    def day(self, date, as_index=False):
        """
        Memoized call to a dict of single day DataFrame
        with date and all computed columns as key. If new
        columns are added to :attr:`Index.df`, an updated
        single day DataFrame is loaded and saved to cache.

        Parameters
        ----------
        date: datetime object
            Date of daily DataFrame to return.
        as_index: bool, default=False
            If true, return an :class:`Index` for specified day instead
            of a DataFrame.

        Returns
        -------
        df or Index: pd.DataFrame or :class:`Index`
            DataFrame or Index for single specified date.
        """
        if not isinstance(date, pd._libs.tslibs.timestamps.Timestamp):
            date = pd.to_datetime(date)
        # Create cache key of all columns added to :attr:`Index.df`.
        current_cache_key = "_".join(self.df.columns)
        if current_cache_key == self._day_cache_key:
            # No new column changes since last accessed,
            # treat like normal cache.
            try:
                if as_index:
                    return Index(self._day_cache[date], self.name)
                else:
                    return self._day_cache[date]
            except KeyError:
                # Cache doesn't exist, create it.
                self._day_cache = {
                    date: df for date, df in self.df.groupby("Date")
                }
        else:
            # New columns added since last accessed,
            # update cache and cache key.
            self._day_cache = {date: df for date, df in self.df.groupby("Date")}
            self._day_cache_key = current_cache_key

        if as_index:
            return Index(self._day_cache[date], self.name)
        else:
            return self._day_cache[date]

    def synthetic_day(self, date, as_index=False):
        """
        Creates :class:`Index` or DataFrames containing only the intersection
        of constituents which existed in the index both the day before
        and the specified day.

        Parameters
        ----------
        date: datetime object
            Date of daily DataFrame to return.
        as_index: bool, default=False
            If true, return an Index for specified day instead
            of a DataFrame.

        Returns
        -------
        df or Index: pd.DataFrame or :class:`Index`
            DataFrame or Index for specified date.
        """
        # Get data for current and previous day.
        if not isinstance(date, pd._libs.tslibs.timestamps.Timestamp):
            current_date = pd.to_datetime(date)
        else:
            current_date = date
        prev_date = self.dates[bisect_left(self.dates, current_date) - 1]
        if prev_date == self.dates[-1]:
            msg = "First day in index history, synthetic day not possible."
            raise IndexError(msg)
        current_df = self.day(current_date)
        prev_df = self.day(prev_date)

        # Return only bond data for bonds on the current day
        # that were in the index on the previous day.
        intersection_ix = set(current_df.index).intersection(prev_df.index)
        df = current_df[current_df.index.isin(intersection_ix)]
        if as_index:
            return Index(df, self.name)
        else:
            return df

    def clear_day_cache(self):
        """Clear cache of stored days."""
        self._day_cache = {}

    def subset(
        self,
        name=None,
        date=None,
        start=None,
        end=None,
        rating=None,
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
        coupon_rate=(None, None),
        country_of_domicile=None,
        country_of_risk=None,
        amount_outstanding=(None, None),
        issue_years=(None, None),
        collateral_type=None,
        OAD=(None, None),
        OAS=(None, None),
        OASD=(None, None),
        liquidity_score=(None, None),
        in_stats_index=None,
        in_returns_index=None,
        is_144A=None,
        is_new_issue=None,
        financial_flag=None,
        special_rules=None,
    ):
        """
        Subset :class:`Index` with customized rules from
        :attr:`Index.df`.

        Parameters
        ----------
        name: str, default=''
            Optional name for returned index.
        date: datetime, default=None
            Single date to subset new index.
        start: datetime, default=None
            Start date for index.
        end: datetime, default=None
            End date for index.
        rating: str , Tuple[str, str], default=None
            Bond rating/rating range for index.

            Examples:

            * str: ``'HY'``, ``'IG'``, ``'AAA'``, ``'Aa1'``, etc.
            * Tuple[str, str]: ``('AAA', 'BB')`` uses all bonds in
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
        coupon_rate: Tuple[float, float]), default=(None, None).
            Coupon rate range of bonds to include, default is all.
        country_of_domicile: str, List[str], default=None
            Country or list of countries of domicile to
            include, default is all.
        country_of_risk: str, List[str], default=None
            Country or list of countries wherer risk is centered to
            include in index, default is all.
        amount_outstanding: Tuple[float, float], default=(None, None).
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
        in_stats_index: bool, default=None
            If True, only include bonds in stats index.
            If False, only include bonds out of stats index.
            By defualt include both.
        in_returns_index: bool, default=None
            If True, only include bonds in returns index.
            If False, only include bonds out of returns index.
            By defualt include both.
        is_144A: bool, default=None
            If True, only include 144A bonds.
            If False, only include non 144A bonds.
            By defualt include both.
        is_new_issue: bool, default=None
            If True, only include bonds in the month they were issued.
            If False, include all bonds.
            By default include all bonds.
        financial_flag: {'financial', 'non-financial', 'other'}, default=None
            Financial flag setting to identify fin and non-fin credits.
        special_rules: str, List[str] default=None
            Special rule(s) for subsetting index using bitwise operators.
            If None, all specified inputs are applied independtently
            of eachother as bitwise &. All rules can be stacked using
            paranthesis to create more complex rules.

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
        name = f"{self.name} subset" if name is None else name

        # Convert dates to datetime.
        if date is not None:
            start = end = date
        start = None if start is None else pd.to_datetime(start)
        end = None if end is None else pd.to_datetime(end)

        # Convert rating to range of inclusive ratings.
        if rating is None:
            ratings = (None, None)
        elif isinstance(rating, str):
            if rating == "IG":
                ratings = (1, 10)
            elif rating == "HY":
                ratings = (11, 21)
            else:
                # Single rating value.
                ratings = (self._ratings[rating], self._ratings[rating])
        else:
            ratings = (self._ratings[rating[0]], self._ratings[rating[1]])

        # Add new issue mask if required.
        if is_new_issue:
            self.df["NewIssueMask"] = new_issue_mask(self.df)

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
        self._add_range_input(coupon_rate, "CouponRate")
        self._add_range_input(amount_outstanding, "AmountOutstanding")
        self._add_range_input(issue_years, "IssueYears")
        self._add_range_input(liquidity_score, "LiquidityCostScore")
        self._add_range_input(OAD, "OAD")
        self._add_range_input(OAS, "OAS")
        self._add_range_input(OASD, "OASD")

        # Set values for including bonds based on credit flags.
        self._flags = {}
        self._add_flag_input(in_returns_index, "USCreditReturnsFlag")
        self._add_flag_input(in_stats_index, "USCreditStatisticsFlag")
        self._add_flag_input(is_144A, "Eligibility144AFlag")
        self._add_flag_input(is_new_issue, "NewIssueMask")

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
        str_repl = {
            key: f'(self.df["{key}"].isin(self._str_list_vals["{key}"]))'
            for key in self._str_list_vals.keys()
        }
        flag_repl = {
            key: f'(self.df["{key}"] == self._flags["{key}"])'
            for key in self._flags
        }
        repl_dict = {**range_repl, **str_repl, **flag_repl, "~(": "(~"}

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
            subset_mask_list.append(replace_multiple(rule, repl_dict))

        # Combine formatting rules into single mask and subset DataFrame,
        # and drop temporary columns.
        subset_mask = " & ".join(subset_mask_list)
        temp_cols = ["NewIssueMask"]
        df = eval(f"self.df.loc[{subset_mask}]").drop(
            temp_cols, axis=1, errors="ignore"
        )
        return Index(df, name)

    def _add_str_list_input(self, input_val, col_name):
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
            if isinstance(input_val, str):
                self._str_list_vals[col_name] = [input_val]
            else:
                self._str_list_vals[col_name] = input_val

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
        with bool type to hash table to use when subsetting
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
            self._flags[col_name] = input_val

    def clean_treasuries(self):
        """
        Clean treasury index for build model curve.

        Keep only non-callable treasuries. Additionally,
        remove bonds that have current maturities less than
        the original maturity of another tenor (e.g., remove
        30 year bonds once their matuirty is less than 10 years,
        10 and 7,7 and 5, etc.).
        """
        # Add Bloomberg ticker.
        self.df["BTicker"] = get_bloomberg_ticker(self.df["CUSIP"].values)

        # For each bond, determine if it meets all the rules
        # required to be used in the curve.
        is_treasury = self.df["BTicker"].isin({"US/T", "T"})
        is_zero_coupon = (self.df["CouponType"] == "ZERO COUPON") | (
            (self.df["CouponType"] == "FIXED") & (self.df["CouponRate"] == 0)
        )
        is_not_strip = ~self.df["BTicker"].isin({"S", "SP", "SPX", "SPY"})
        matures_on_15 = pd.Series(
            [d.day == 15 for d in self.df["MaturityDate"]], index=self.df.index
        )
        is_noncall = self.df["CallType"] == "NONCALL"
        oas_not_strongly_negative = self.df["OAS"] > -30
        oas_not_strongly_positive = self.df["OAS"] < 40
        maturity_map = {2: 0, 3: 2, 5: 3, 8: 5, 7: 5, 10: 7, 30: 10}
        has_normal_mat = self.df["OriginalMaturity"].isin(maturity_map.keys())
        matures_in_more_than_3_months = self.df["MaturityYears"] > 3 / 12
        bad_cusips = ["912820FL6"]
        is_not_bad_cusip = ~self.df["CUSIP"].isin(bad_cusips)

        # Apply all rules.
        df = self.df[
            (is_treasury | is_zero_coupon)
            & (is_not_strip | matures_on_15)
            & is_noncall
            & has_normal_mat
            & oas_not_strongly_negative
            & oas_not_strongly_positive
            & matures_in_more_than_3_months
            & is_not_bad_cusip
        ].copy()
        mask = [
            my > maturity_map[om] or bt in {"B", "S", "SP", "SPX", "SPY"}
            for my, om, bt in zip(
                df["MaturityYears"], df["OriginalMaturity"], df["BTicker"]
            )
        ]
        df = df[mask].copy()
        self.df = df.copy()

    def get_value_history(
        self,
        col,
        start=None,
        end=None,
        inclusive_end_date=True,
        synthetic=False,
    ):
        """
        Get history of any column for all cusips in Index.

        Parameters
        ----------
        col: str
            Column from :attr:`Index.df` to build history for (e.g., 'OAS').
        start: datetime, default=None
            Start date for value history.
        end: datetime, default=None
            End date for value history.
        inclusive_end_date: bool, default=True
            If True include end date in returned DataFrame.
            If False do not include end date.
        synthethic: bool, default=False
            If True, use synthethic day data.

        Returns
        -------
        pd.DataFrame
            DataFrame with datetime index, CUSIP columns, and price values.
        """
        cusips = set(self.cusips)
        dates = self.dates[1:] if synthetic else self.dates
        if start is not None:
            dates = dates[dates >= pd.to_datetime(start)]
        if end is not None:
            if inclusive_end_date:
                dates = dates[dates <= pd.to_datetime(end)]
            else:
                dates = dates[dates < pd.to_datetime(end)]

        # Build dict of historical values for each CUSIP and
        # convert to DataFrame.
        hist_d = defaultdict(list)
        for d in dates:
            day_df = self.synthetic_day(d) if synthetic else self.day(d)
            # Add prices for CUSIPs with column data.
            for c, v in zip(day_df["CUSIP"].values, day_df[col].values):
                hist_d[c].append(v)
            # Add NaNs for CUSIPs that are missing.
            missing_cusips = cusips.symmetric_difference(set(day_df["CUSIP"]))
            for c in missing_cusips:
                hist_d[c].append(np.NaN)

        return pd.DataFrame(hist_d, index=dates)

    def get_synthetic_differenced_history(self, col):
        """
        Save the difference history of a column.

        Parameters
        ----------
        col: str
            Column from :attr:`Index.df` to build synthetic
            differenced history for (e.g., 'OAS').

        Returns
        -------
        pd.Series
            Series of sythetically differenced market value
            weighting for specified column.
        """
        dates = self.dates[1:]
        a = np.zeros(len(dates))
        cols = ["MarketValue", col]
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i, date in enumerate(dates):
            df = self.synthetic_day(date).dropna(subset=cols)
            prev_df = self.day(self.dates[i]).dropna(subset=cols)
            prev_df = prev_df[prev_df.index.isin(df.index)]
            a[i] = (
                np.sum(df["MarketValue"] * df[col]) / np.sum(df["MarketValue"])
            ) - (
                np.sum(prev_df["MarketValue"] * prev_df[col])
                / np.sum(prev_df["MarketValue"])
            )
        warnings.simplefilter("default", category=RuntimeWarning)

        # Add the the cumulative synthetic differences to the current
        # value backwards in time to yield the synthetic history
        # of the specified column.
        current_val = np.sum(df["MarketValue"] * df[col]) / np.sum(
            df["MarketValue"]
        )
        offset = np.cumsum(a[::-1])
        synthetic_history = np.concatenate(
            [(current_val - offset)[::-1], [current_val]]
        )
        return pd.Series(synthetic_history, index=self.dates, name=col)

    def get_cusip_history(self, cusip):
        """
        Get full history for specified cusip.

        Parameters
        ----------
        cusip: str
            Specified cusip.
        Returns
        -------
        pd.DataFrame
            DataFrame with datetime index and :attr:`Index.df`
            columns for specified cusip.
        """
        return pd.concat(
            [
                pd.DataFrame(self.day(d).loc[cusip, :]).T.set_index("Date")
                for d in self.dates
            ]
        )

    def market_value_weight(self, col, synthetic=False):
        """
        Market value weight a specified column vs entire
        index market value.

        Parameters
        ----------
        col: str
            Column name to weight, (e.g., 'OAS').
        synthethic: bool, default=False
            If True, use synthethic day data.

        Returns
        -------
        pd.Series:
            Series of market value weighting for specified column.
        """
        dates = self.dates[1:] if synthetic else self.dates

        a = np.zeros(len(dates))
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i, date in enumerate(dates):
            df = self.synthetic_day(date) if synthetic else self.day(date)
            df = df.dropna(subset=["MarketValue", col])
            a[i] = np.sum(df["MarketValue"] * df[col]) / np.sum(
                df["MarketValue"]
            )
        warnings.simplefilter("default", category=RuntimeWarning)
        return pd.Series(a, index=dates, name=col)

    def find_rating_changes(self, rating_agency):
        """
        Find rating changes of persistent index.

        Parameters
        ----------
        rating_agency: {'SP', 'Moody', 'Fitch'}.
            Rating agency to find changes for.

        Returns
        -------
        change_df: pd.DataFrame
            DataFrame of all rating changes, with no index,
            columns are: ``['cusip', 'date', 'change', 'new_val',
            'old_val', 'sector', 'subsector']``.
        """
        # Create numeric rating column for single rating agency.
        col = f"{rating_agency}Rating"
        new_col = col.replace("R", "NumericR")
        rating_vector = self.df[col].fillna("NR").values
        ratings_json = load_json("ratings")
        num_ratings = np.vectorize(ratings_json.__getitem__)(
            rating_vector
        ).astype(float)
        num_ratings[num_ratings == 0] = np.nan
        self.df[new_col] = num_ratings

        # Get rating history for all cusips.
        rating_df = self.get_value_history(new_col)
        print(rating_df.iloc[:10, :10])
        change_cusips = []
        for cusip in rating_df.columns:
            if not check_all_equal(list(rating_df[cusip].dropna())):
                change_cusips.append(cusip)

        # Store direction and magnitude of rating changes and
        # the date of rating changes in separate dicts.
        change_dict = defaultdict(list)
        for cusip in change_cusips:
            ratings = rating_df[cusip].dropna().values
            diff = np.diff(ratings)
            nonzero_locs = np.nonzero(diff)[0]
            for loc in nonzero_locs:
                # Ensure at least 20 days with no change on either
                # side of rating change to include change.
                start = int(max(0, loc - 20))
                end = int(min(len(diff) - 1, loc + 20))
                local_diff = diff[start:end]
                changes = local_diff[local_diff != 0]
                if len(changes) != 1:
                    continue  # not true rating change
                change_val = diff[loc]
                if change_val == 0:
                    continue  # not true rating change

                rating_series = rating_df[cusip].dropna()
                date = rating_series.index[loc + 1]
                new_val = ratin

    @property
    @lru_cache(maxsize=None)
    def _treasury(self):
        """:class:`TreasuryCurve`:  Treasury curves."""
        return TreasuryCurve()

    def compute_total_returns(self):
        """
        Vectorized implementation for computing total returns
        for all cusips in index, adjusting for coupon delivery
        and reinvestment.

        Appends results as new column `TRet` in :attr:`Index.df`.
        """
        # Stop computation if already performed.
        if "TRet" in list(self.df):
            return

        # Compute total returns without adjusting for coupons.
        price_df = self.get_value_history("DirtyPrice")
        cols = list(price_df)
        price = price_df.values
        tret = price[1:] / price[:-1] - 1

        # Find dates when coupons were paid by finding dates
        # with a decrease in accrued interest.
        accrued = self.get_value_history("AccruedInterest")[cols].values
        warnings.simplefilter("ignore", category=RuntimeWarning)
        accrued_mask = np.zeros(accrued.shape)
        accrued_mask[1:] = np.where(np.diff(accrued, axis=0) < 0, 1, 0)
        warnings.simplefilter("default", category=RuntimeWarning)

        # Find coupon rate, add coupon back into prices and assume
        # coupon is reinvested at same rate.
        coupon_rate = self.get_value_history("CouponRate")[cols].values / 2
        coupon_adj_price = price + accrued_mask * coupon_rate
        reinvesting_multiplier = accrued_mask * (1 + coupon_rate / price)
        reinvesting_multiplier[reinvesting_multiplier == 0] = 1
        reinvesting_multiplier[np.isnan(reinvesting_multiplier)] = 1
        reinvesting_multiplier = np.cumprod(reinvesting_multiplier, axis=0)
        coupon_adj_tret = (
            coupon_adj_price[1:] / coupon_adj_price[:-1] - 1
        ) * reinvesting_multiplier[1:]

        # Combine both methods taking element-wise maximum to
        # account for day when coupon is paid.
        tret_df = pd.DataFrame(
            np.maximum(tret, coupon_adj_tret),
            index=price_df.index[1:],
            columns=cols,
        )

        # Append total returns to :attr:`Index.df`.
        self.df["TRet"] = np.NaN
        for date in tret_df.index:
            cusips = list(self.day(date).index)
            self.df.loc[self.df["Date"] == date, "TRet"] = tret_df.loc[
                date, cusips
            ].values

    def compute_excess_returns(self):
        """
        Compute excess returns for all cusips in index.

        Appends results as new column `XSRet` in :attr:`Index.df`.
        """
        # Stop computation if already performed.
        if "XSRet" in list(self.df):
            return

        # Compute total returns.
        self.compute_total_returns()

        # Fill missing 0.5 year KRD values with 0.
        krd_cols = [col for col in list(self.df) if "KRD" in col]
        self.df.loc[
            (self.df["KRD06mo"].isna())
            & (self.df[krd_cols].isna().sum(axis=1) == 1),
            "KRD06mo",
        ] = 0

        # Compute and append excess returns iteratively for each day.
        self.df["XSRet"] = np.NaN
        for date in self.dates[1:]:
            df = self.day(date)
            # Calculate hypothetical treasury weights for each cusip.
            weights = np.zeros([len(df), 7])
            weights[:, :-1] = df[krd_cols].values / self._treasury.KRDs(date)

            # Add cash component to make weights sum to 1.
            weights[:, -1] = 1 - np.sum(weights, axis=1)
            tsy_trets = np.sum(weights * self._treasury.trets(date), axis=1)
            ex_rets = df["TRet"].values - tsy_trets
            self.df.loc[self.df["Date"] == date, "XSRet"] = ex_rets

    def aggregate_excess_returns(self, start_date=None):
        """
        Aggregate excess returns since start date.

        Parameters
        ----------
        start_date: datetime
            Date to start aggregating returns.
        col_name: str
            Column name to store aggregate returns.

        Returns
        -------
        float or pd.Seires
            Aggregated excess returns for either full index
            or individual cusips.
        """
        self.compute_excess_returns()

        # Find excess and total returns in date range.
        ex_rets = self.market_value_weight("XSRet")
        t_rets = self.market_value_weight("TRet")
        if start_date is not None:
            ex_rets = ex_rets[ex_rets.index > pd.to_datetime(start_date)]
            t_rets = t_rets[t_rets.index > pd.to_datetime(start_date)]

        # Calculate implied treasy returns.
        tsy_t_rets = t_rets - ex_rets
        # Calculate total returns over period and use treasury
        # returns to back out excess returns.
        total_ret = np.prod(1 + t_rets) - 1
        tsy_total_ret = np.prod(1 + tsy_t_rets) - 1
        return total_ret - tsy_total_ret


# %%
def main():
    from lgimapy.data import Database
    from lgimapy.utils import Time, load_json

    db = Database()
    # db.load_market_data(start="7/1/2019", end="8/15/2019", local=True)

    db.load_market_data(start="10/6/2019")
    db.load_market_data(start="1/6/2015", end="8/1/2015", local=True)

    self = Index(db.build_market_index(in_stats_index=True).df)
    ix = Index(db.build_market_index(cusip="000361AQ8").df)
    self.compute_total_returns()

    kwargs = load_json("indexes")

    list(self.day("10/9/2019").index)