import warnings
from bisect import bisect_left
from collections import defaultdict, OrderedDict
from functools import lru_cache
from inspect import getfullargspec

import numpy as np
import pandas as pd
from oslo_concurrency import lockutils

from lgimapy.bloomberg import get_bloomberg_ticker
from lgimapy.data import Bond, concat_index_dfs, new_issue_mask, TreasuryCurve
from lgimapy.utils import (
    check_all_equal,
    dump_json,
    load_json,
    mkdir,
    replace_multiple,
    root,
    to_int,
    to_datetime,
    to_list,
)

# %%
def get_unique_fid(fid_map):
    """
    Generate a unique filename given map of keys
    to filenames.

    Parameters
    ----------
    fid_map: dict
        Mapping of keys to filename values.

    Returns
    -------
    fid: str
        Unique filename not in input filename map.
    """

    def generate_random_fid():
        """str: Generate random filename."""
        n_digits_in_fid = 18
        max_val = int(10 ** n_digits_in_fid - 1)
        fid_val = np.random.randint(0, max_val, dtype="int64")
        fid = f"{str(fid_val).zfill(n_digits_in_fid)}.csv"
        return fid

    while True:
        fid = generate_random_fid()
        if fid not in fid_map:
            return fid


class Index:
    """
    Class for indexes built by :class:`Database`.

    Parameters
    ----------
    index_df: pd.DataForame
        Index DataFrame from :meth:`Database.build_market_index`.
    name: str, default=''
        Optional name of index.
    constraints: dict, optional
        Key: value pairs of the constraints used in either
        :meth:`Database.build_market_index` or
        :meth:`Index.subset` to create current :class:`Index`.

    Attributes
    ----------
    df: pd.DataFrame
        DataFrame with each row containing a bond in
        the :class:`Index`.
    """

    def __init__(self, index_df, name="", constraints=None):
        self.df = index_df.set_index("CUSIP", drop=False)
        self.name = name
        self._constraints = constraints
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
            raise TypeError(f"Right operand must be an {type(self).__name__}.")

    def __radd__(self, other):
        """Combine mutltiple instances of :class:`Index` together."""
        if isinstance(other, Index):
            return Index(concat_index_dfs([self.df, other.df]))
        elif other == 0:
            return self
        else:
            raise TypeError(f"Left operand must be an {type(self).__name__}.")

    def __iadd__(self, other):
        """Combine mutltiple instances of :class:`Index` together."""
        if isinstance(other, Index):
            return Index(concat_index_dfs([self.df, other.df]))
        else:
            raise TypeError(f"Right operand must be an {type(self).__name__}.")

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
        return sorted(list(self.df["Sector"].unique().dropna()))

    @property
    @lru_cache(maxsize=None)
    def subsectors(self):
        """List[str]: Memoized unique sorted subsectors in index."""
        return sorted(list(self.df["Subsector"].unique().dropna()))

    @property
    @lru_cache(maxsize=None)
    def issuers(self):
        """List[str]: Memoized unique sorted issuers in index."""
        return sorted(list(self.df["Issuer"].unique().dropna()))

    @property
    @lru_cache(maxsize=None)
    def tickers(self):
        """List[str]: Memoized unique sorted tickers in index."""
        return sorted(list(self.df["Ticker"].unique().dropna()))

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

    @property
    def constraints(self):
        """Dict of constraints used to construct index."""
        return OrderedDict(
            sorted(self._constraints.items(), key=lambda k: k[0])
        )

    def copy(self):
        """Create copy of current :class:`Index`."""
        return Index(self.df.copy(), constraints=self.constraints)

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
        date = to_datetime(date)
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
        current_date = to_datetime(date)
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
        self._day_cache = ""

    def subset(
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
        Subset :class:`Index` with customized rules from
        :attr:`Index.df`.

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
        name = f"{self.name} subset" if name is None else name

        # Convert dates to datetime.
        if date is not None:
            start = end = date
        start = None if start is None else to_datetime(start)
        end = None if end is None else to_datetime(end)

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
        argspec = getfullargspec(self.subset)
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
        subset_constraints = {
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

        # Store parameters used to build current index updated
        # with parameters added/modified for subset.
        cat_error_msg = (
            "The constraint provided for `{}` is not within "
            "current index constraints."
        )
        flag_error_msg = (
            "The constraint provided for `{}` does not match the "
            "current index constraint."
        )
        not_imp_msg = (
            "The constraint provided for `{}` does not match the "
            "current index constraint, which has not been safely tested."
        )
        subset_index_constraints = self.constraints.copy()
        for constraint, subset_val in subset_constraints.items():
            if constraint not in self.constraints:
                # Add new constraint.
                subset_index_constraints[constraint] = subset_val
                continue

            # If constraint exists in both index and subset:
            # Update constraint to most stringent
            # combination of current index and subset.
            index_val = self.constraints[constraint]
            if constraint in category_constraints:
                # Take intersection of two constraints.
                intersection = to_list(
                    set(index_val) & set(subset_val), sort=True
                )
                if intersection:
                    subset_index_constraints[constraint] = intersection
                else:
                    raise ValueError(cat_error_msg.format(constraint))
            elif constraint in flag_constraints:
                # Ensure flag values are the same.
                if subset_val != index_val:
                    raise ValueError(flag_error_msg.format(constraint))
            elif constraint in range_constraints:
                # Find the max of the minimums.
                if index_val[0] is None:
                    min_con = subset_val[0]
                elif subset_val[0] is None:
                    min_con = index_val[0]
                else:
                    min_con = max(index_val[0], subset_val[0])
                # Find the min of the maximums.
                if index_val[1] is None:
                    max_con = subset_val[1]
                elif subset_val[1] is None:
                    max_con = index_val[1]
                else:
                    max_con = min(index_val[1], subset_val[1])
                subset_index_constraints[constraint] = (min_con, max_con)
            else:
                # Hopefully this never happens.
                raise NotImplementedError(not_imp_msg.format(constraint))

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

        return Index(df, name, subset_index_constraints)

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
        start: datetime, optional
            Start date for value history.
        end: datetime, optional
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
            dates = dates[dates >= to_datetime(start)]
        if end is not None:
            if inclusive_end_date:
                dates = dates[dates <= to_datetime(end)]
            else:
                dates = dates[dates < to_datetime(end)]

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

    @lockutils.synchronized(
        "synthetic_difference_history",
        external=True,
        lock_path=root("data/synthetic_difference/file_maps"),
    )
    def _synthetic_difference_saved_history(self, col):
        """
        Get saved difference history if it exists.

        Parameters
        ----------
        col: str
            Column to perform synthetic difference history on.

        Returns
        -------
        fid: str
            Filename where calculated history is to be saved.
        history: dict
            Datetime key and respective synthetic
            difference values.
        """
        key = "|".join([f"{k}: {v}" for k, v in self.constraints.items()])
        json_dir = root("data/synthetic_difference/file_maps")
        history_dir = root("data/synthetic_difference/history")
        mkdir(json_dir)
        mkdir(history_dir)
        json_fid = f"synthetic_difference/file_maps/{col}"
        fid_map = load_json(json_fid, empty_on_error=True)
        try:
            # Find fid if key exists in file.
            filename = fid_map[key]
        except KeyError:
            # Get a unique fid, update the fid mappping file,
            # and return an empty history dict.
            filename = get_unique_fid(fid_map)
            fid_map[key] = filename
            dump_json(fid_map, json_fid)
            fid = history_dir / filename
            history = {}
            return fid, history
        else:
            fid = history_dir / filename
            try:
                history = (
                    pd.read_csv(
                        fid,
                        index_col=0,
                        parse_dates=True,
                        infer_datetime_format=True,
                    )
                    .iloc[:, 0]
                    .to_dict()
                )
            except FileNotFoundError:
                history = {}
            return fid, history

    def get_synthetic_differenced_history(self, col, dropna=False):
        """
        Save the difference history of a column.

        Parameters
        ----------
        col: str
            Column from :attr:`Index.df` to build synthetic
            differenced history for (e.g., 'OAS').
        dropna: bool, default=False
            If True drop columns with missing values for either
            `MarketValue` or specified column.

        Returns
        -------
        pd.Series
            Series of sythetically differenced market value
            weighting for specified column.
        """
        fid, saved_history = self._synthetic_difference_saved_history(col)
        dates = self.dates[1:]
        a = np.zeros(len(dates))
        cols = ["MarketValue", col]
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i, date in enumerate(dates):
            # Check if date exists in saved history.
            try:
                a[i] = saved_history[date]
            except KeyError:
                pass
            else:
                if date == dates[-1]:
                    pass
                else:
                    continue

            # Date does not exist in saved history.
            # Calculate the differenced history for
            # bonds which existed for both current and
            # previous days.
            if dropna:
                df = self.synthetic_day(date).dropna(subset=cols)
                prev_df = self.day(self.dates[i]).dropna(subset=cols)
            else:
                df = self.synthetic_day(date)
                prev_df = self.day(self.dates[i])
            prev_df = prev_df[prev_df.index.isin(df.index)]
            a[i] = (
                np.sum(df["MarketValue"] * df[col]) / np.sum(df["MarketValue"])
            ) - (
                np.sum(prev_df["MarketValue"] * prev_df[col])
                / np.sum(prev_df["MarketValue"])
            )
            saved_history[date] = a[i]

        warnings.simplefilter("default", category=RuntimeWarning)

        # Save synthetic difference history to file.
        new_history = pd.DataFrame(pd.Series(saved_history))
        new_history.to_csv(fid)

        # pd.DataFrame(calculated_s).to_csv(fid)
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

    def market_value_weight(self, col, low_memory=False):
        """
        Market value weight a specified column vs entire
        index market value.

        Parameters
        ----------
        col: str
            Column name to weight, (e.g., 'OAS').
        low_memory: bool, default=False
            If True, perform all operations inplace on :attr:`Index.df`.

        Returns
        -------
        pd.Series:
            Series of market value weighting for specified column.
        """
        if low_memory:
            # Perform entire operation inplace to save memory.
            return (
                (
                    self.df[["Date", "MarketValue", col]]
                    .eval(f"mvw_col=MarketValue*{col}")
                    .groupby("Date")
                    .sum()
                )
                .eval("mvw_col/MarketValue")
                .rename(col)
            )
        else:
            df = self.df[["Date", "MarketValue", col]].copy()
            df["mvw_col"] = df["MarketValue"] * df[col]
            g = df[["Date", "MarketValue", "mvw_col"]].groupby("Date").sum()
            return (g["mvw_col"] / g["MarketValue"]).rename(col)

    def market_value_weight_vol(self, col, window_size=20, annualized=True):
        """
        Market value weight the volatilities of the specified
        column using a rolling window approach. The variance
        of each bond in the index is computed daily and the
        average variance is then transformed back to volatility.

        Parameters
        ----------
        col: str
            Column name to weight, (e.g., ``'OAS'``).
        window_size: int, default=20
            Number of sample periods to use in rolling window.
        annualized: bool, default=True
            If True annualized the resulting volatilities by
            multiply by square root of 252.

        Returns
        -------
        vol: pd.Series
            Computed volatility with datetime index.
        """
        col_df = self.get_value_history(col)
        mv_df = self.get_value_history("MarketValue")
        var_df = col_df.rolling(
            window=window_size, min_periods=window_size
        ).var()
        weights_df = mv_df.divide(np.sum(mv_df, axis=1).values, axis=0)
        vol = np.sum(
            ((var_df * weights_df ** 2) ** 0.5).dropna(how="all"), axis=1
        )
        if annualized:
            vol *= 252 ** 0.5
        return vol

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
        np.seterr(divide="ignore", invalid="ignore")
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
        # In Bloomberg this is done as two separate calculations.
        # For individual bonds, the coupon is not assumed to be
        # reinvested. However, when aggregating bonds for an
        # index, the `quantity` of bonds reflects the return
        # from the coupon and is included in excess returns.
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
            ex_rets = ex_rets[ex_rets.index > to_datetime(start_date)]
            t_rets = t_rets[t_rets.index > to_datetime(start_date)]

        # Calculate implied risk free returns.
        rf_rets = t_rets - ex_rets
        # Calculate total returns over period and use treasury
        # returns to back out excess returns.
        total_ret = np.prod(1 + t_rets) - 1
        rf_total_ret = np.prod(1 + rf_rets) - 1
        return total_ret - rf_total_ret
