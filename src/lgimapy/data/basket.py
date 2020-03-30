import warnings
from bisect import bisect_left
from collections import defaultdict, OrderedDict
from functools import lru_cache
from importlib import import_module
from inspect import getfullargspec

import numpy as np
import pandas as pd
from oslo_concurrency import lockutils
from scipy import stats

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


def mode(x):
    """Get most frequent occurance in Pandas aggregate by."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return stats.mode(x)[0][0]


def groupby(df, cols):
    """
    Group basket of bonds together by seletected features.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame basket of bonds.
    cols: str or List[str].
        Column(s) in `df` to group by.

    Returns
    -------
    pd.DataFrame
        Grouped DataFrame.
    """
    if cols == "risk entity":
        groupby_cols = ["Ticker", "Issuer"]
    else:
        groupby_cols = to_list(cols, dtype=str)

    agg_rules = {
        "Ticker": mode,
        "Issuer": mode,
        "Sector": mode,
        "Subsector": mode,
        "LGIMASector": mode,
        "OAD_Diff": np.sum,
        "MarketValue": np.sum,
        "P_AssetValue": np.sum,
        "BM_AssetValue": np.sum,
        "P_Weight": np.sum,
        "Weight_Diff": np.sum,
        "OASD_Diff": np.sum,
        "OAS_Diff": np.sum,
        "DTS_Diff": np.sum,
        "DTS_Contrib": np.sum,
    }

    agg_cols = {
        col: rule
        for col, rule in agg_rules.items()
        if col in df.columns and col not in groupby_cols
    }
    return df.groupby(groupby_cols, observed=True).aggregate(agg_cols)


class BondBasket:
    """
    Container for basket of bonds. Used as top level inheritance
    for account portfolios and indexes.

    Parameters
    ----------
    df: pd.DataFame
        DataFrame with each row representing an individual CUSIP.
    name: str, opitional
        Name for basket.
    constraints: dict, optional
        Key: value pairs of the constraints used in
        :meth:`BondBasket.subset` to create
        current :class:`BondBasket`.
    """

    def __init__(self, df, name=None, constraints=None):
        self.df = df.set_index("CUSIP", drop=False)
        self.name = "" if name is None else name
        self._constraints = dict() if constraints is None else constraints

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

    @property
    def ticker_df(self):
        return groupby(self.df, "Ticker")

    def subset(
        self,
        name=None,
        date=None,
        account=None,
        strategy=None,
        start=None,
        end=None,
        rating=(None, None),
        currency=None,
        cusip=None,
        isin=None,
        issuer=None,
        ticker=None,
        L3=None,
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
        Subset :class:`BondBasket` with customized rules from
        :attr:`BondBasket.df`.

        Parameters
        ----------
        name: str, optional
            Name for returned basket.
        date: datetime, optional
            Single date to select basket from.
        account: str or List[str], optional
            Account or accounts to include.
        strategy: str or List[str], optional
            Strategy or strategies to include.
        start: datetime, optional
            Start date for basket, if None the start date
            from load is used.
        end: datetime, optional
            End date for basket, if None the end date from
            load is used.
        rating: str , Tuple[str, str], optional
            Bond rating/rating range for basket.

            Examples:

            * str: ``'HY'``, ``'IG'``, ``'AAA'``, ``'Aa1'``, etc.
            * Tuple[str, str]: ``('AAA', 'BB')`` uses all bonds in
              specified inclusive range.
        currency: str, List[str], optional
            Currency or list of currencies to include, default is all.
        cusip: str, List[str]: optional,
            CUSIP or list of CUSIPs to include, default is all.
        isin: str, List[str]: optional,
            ISIN or list of ISINs to include, default is all.
        issuer: str, List[str], optional
            Issuer, or list of issuers to include, default is all.
        ticker: str, List[str], optional
            Ticker or list of tickers to include, default is all.
        L3: str or List[str], optional
            Level 3 Bloomberg sector or list of sectors
            to include, default is all.
        sector: str, List[str], optional
            Level 4 Bloomberg sector or list of sectors
            to include, default is all.
        subsector: str, List[str], optional
            Subsector or list of subsectors to include, default is all.
        drop_treasuries: bool, default=True
            Whether to drop treausuries.
        drop_municipals: bool, default=False
            Whether to drop municipals.
        maturity: Tuple[float, float], {5, 10, 20, 30}, optional
            Maturities to include, if int is specified the
            following ranges are used:

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
            If ``True``, only include bonds in stats index.
            If ``False``, only include bonds out of stats index.
            By defualt include both.
        in_returns_index: bool, optional
            If ``True``, only include bonds in returns index.
            If ``False``, only include bonds out of returns index.
            By defualt include both.
        in_agg_stats_index: bool, optional
            If ``True``, only include bonds in aggregate stats index.
            If ``False``, only include bonds out of aggregate stats index.
            By defualt include both.
        in_agg_returns_index: bool, optional
            If ``True``, only include bonds in aggregate returns index.
            If ``False``, only include bonds out of aggregate returns index.
            By defualt include both.
        in_hy_stats_index: bool, optional
            If ``True``, only include bonds in HY stats index.
            If ``False``, only include bonds out of HY stats index.
            By defualt include both.
        in_hy_returns_index: bool, optional
            If ``True``, only include bonds in HY returns index.
            If ``False``, only include bonds out of HY returns index.
            By defualt include both.
        in_any_index: bool, optional
            If ``True``, only include bonds in any index.
            If ``False``, only include bonds not in any index.
            By defualt include both.
        is_144A: bool, optional
            If ``True``, only include 144A bonds.
            If ``False``, only include non 144A bonds.
            By defualt include both.
        is_new_issue: bool, optional
            If ``True``, only include bonds in the month they were issued.
            If ``False``, include all bonds.
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
        name = self.name if name is None else name

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
            rating = (
                self._ratings[str(rating[0])],
                self._ratings[str(rating[1])],
            )

        # Convert all category constraints to lists.
        account = to_list(account, dtype=str, sort=True)
        strategy = to_list(strategy, dtype=str, sort=True)
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
        L3 = to_list(L3, dtype=str, sort=True)
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
            "name",
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
            "account": ("Account", account),
            "strategy": ("Strategy", strategy),
            "currency": ("Currency", currency),
            "cusip": ("CUSIP", cusip),
            "isin": ("ISIN", isin),
            "issuer": ("Issuer", issuer),
            "ticker": ("Ticker", ticker),
            "L3": ("L3", L3),
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

        # Return current BondBasket or child class of BondBasket
        # If required, import child class.
        class_name = self.__class__.__name__
        if class_name == "BondBasket":
            return BondBasket(df, name, subset_index_constraints)
        elif class_name in {"Account", "Strategy"}:
            child_class = getattr(import_module("lgimapy.data"), class_name)
            return child_class(df, name, self.date, subset_index_constraints)
        else:
            child_class = getattr(import_module("lgimapy.data"), class_name)
            return child_class(df, name, subset_index_constraints)

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
