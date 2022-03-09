import sys
import warnings
from bisect import bisect_left
from collections import defaultdict, OrderedDict
from functools import lru_cache, cached_property
from importlib import import_module
from inspect import getfullargspec
from numbers import Number

import numpy as np
import pandas as pd
from oslo_concurrency import lockutils
from scipy import stats

from lgimapy.bloomberg import get_bloomberg_ticker
from lgimapy.data import (
    Bond,
    concat_index_dfs,
    groupby,
    new_issue_mask,
    TreasuryCurve,
)
from lgimapy.stats import mean, mode, percentile, std
from lgimapy.utils import (
    check_all_equal,
    check_market,
    dump_json,
    load_json,
    mkdir,
    replace_multiple,
    root,
    to_datetime,
    to_int,
    to_list,
)


# %%


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
    market: ``{"US", "EUR", "GBP"}``, default="US"
        Market to get trade dates for.
    index: str or None, default='CUSIP'
        Column to use as index for :attr:`BondBasket.df`.
    constraints: dict, optional
        Key: value pairs of the constraints used in
        :meth:`BondBasket.subset` to create
        current :class:`BondBasket`.
    """

    def __init__(
        self, df, name=None, market="US", index="CUSIP", constraints=None,
    ):
        self.index = index
        if index is not None:
            self.df = df.set_index(index, drop=False)
        else:
            self.df = df.copy()
        self.name = "" if name is None else name
        self.market = check_market(market)
        self._constraints = dict() if constraints is None else constraints

    @cached_property
    def dates(self):
        """List[datetime]: Memoized unique sorted dates in index."""
        dates = list(pd.to_datetime(self.df["Date"].unique()))
        return [d for d in dates if d not in self._holiday_dates]

    @cached_property
    def start_date(self):
        return self.dates[0]

    @cached_property
    def end_date(self):
        return self.dates[-1]

    def _unique(self, col, df=None):
        """
        List[str]:
            Sorted list of unique values for a columns in :attr:`df`.
        """
        df = self.df if df is None else df
        return sorted(list(df[col].unique().dropna()))

    @cached_property
    def cusips(self):
        """List[str]: Memoized unique CUSIPs in index."""
        return self._unique("CUSIP")

    @cached_property
    def isins(self):
        """List[str]: Memoized unique ISINs in index."""
        return self._unique("ISIN")

    @cached_property
    def sectors(self):
        """List[str]: Memoized unique sorted sectors in index."""
        return self._unique("Sector")

    @cached_property
    def subsectors(self):
        """List[str]: Memoized unique sorted subsectors in index."""
        return self._unique("Subsector")

    @cached_property
    def issuers(self):
        """List[str]: Memoized unique sorted issuers in index."""
        return self._unique("Issuer")

    @cached_property
    def tickers(self):
        """List[str]: Memoized unique sorted tickers in index."""
        return self._unique("Ticker")

    @property
    def bonds(self):
        """List[:class:`Bond`]: List of individual bonds in index."""
        return [Bond(bond) for _, bond in self.df.iterrows()]

    @cached_property
    def all_trade_dates(self):
        """List[datetime]: Memoized list of trade dates."""
        return list(self._trade_date_df.index)

    @cached_property
    def _trade_dates(self):
        """List[datetime]: Memoized list of trade dates."""
        trade_dates = self._trade_date_df[self._trade_date_df["holiday"] == 0]
        return list(trade_dates.index)

    @cached_property
    def _holiday_dates(self):
        """List[datetime]: Memoized list of holiday dates."""
        holidays = self._trade_date_df[self._trade_date_df["holiday"] == 1]
        return list(holidays.index)

    @cached_property
    def _trade_date_df(self):
        """pd.DataFrame: Memoized trade date boolean series for holidays."""
        if self.market == "US":
            fid = root(f"data/{self.market}/trade_dates.parquet")
        else:
            fid = root(f"data/{self.market}/trade_dates_{sys.platform}.parquet")
        return pd.read_parquet(fid)

    @cached_property
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

    @property
    def issuer_df(self):
        return groupby(self.df, "Issuer")

    @cached_property
    def _ratings_changes_df(self):
        """pd.DataFrame: Rating change history of basket."""
        fid = root("data/rating_changes.parquet")
        df = pd.read_parquet(fid)
        return df[
            (df["Date_PREV"] >= self.dates[0])
            & (df["Date_NEW"] <= self.dates[-1])
            & (df["ISIN"].isin(self.isins))
        ].copy()

    def subset(
        self,
        df=None,
        name=None,
        date=None,
        account=None,
        strategy=None,
        start=None,
        end=None,
        rating=(None, None),
        rating_risk_bucket=None,
        analyst_rating=(None, None),
        currency=None,
        cusip=None,
        isin=None,
        issuer=None,
        ticker=None,
        L3=None,
        sector=None,
        subsector=None,
        LGIMA_sector=None,
        LGIMA_top_level_sector=None,
        BAML_top_level_sector=None,
        BAML_sector=None,
        benchmark_treasury=(None, None),
        drop_treasuries=False,
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
        rating_risk_bucket: str, List[str], optional
            Rating risk buckets to include in index, default is all.
        analyst_rating: Tuple[float, float], default=(None, None)
            Range of analyst ratings to include.
        currency: str, List[str], optional
            Currency or list of currencies to include, default is all.
        cusip: str, List[str]: optional
            CUSIP or list of CUSIPs to include, default is all.
        isin: str, List[str]: optional
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
        BAML_top_level_sector: str, List[str], optional
            BAML top level sector or list of sectors to include in index.
        BAML_sector: str, List[str], optional
            BAML sector or list of sectors to include in index.
        LGIMA_sector: str, List[str], optional
            LGIMA custom sector(s) to include, default is all.
        LGIMA_top_level_sector: str, List[str], optional
            LGIMA top level sector(s) to include, default is all.
        benchmark_treasury: Tuple[float, float], defalt=(None, None)
            Respective benchmark treasuries to include, default is all.
        drop_treasuries: bool, default=False
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
        OASD: Tuple[float, float], default=(None, None)
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
        :class:`BondBasket`:
            :class:`BondBasket` or current child class of
            :class:`BondBasket` subset with specified rules.
        """
        df = self.df.copy() if df is None else df.copy()
        name = self.name if name is None else name

        # Convert dates to datetime.
        if date is not None:
            start = end = date
        start = None if start is None else to_datetime(start)
        end = None if end is None else to_datetime(end)

        # Convert rating to range of inclusive ratings.
        if rating == (None, None):
            pass
        else:
            rating = self.convert_input_ratings(rating)

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
        rating_risk_bucket = to_list(rating_risk_bucket, dtype=str, sort=True)
        L3 = to_list(L3, dtype=str, sort=True)
        sector = to_list(sector, dtype=str, sort=True)
        subsector = to_list(subsector, dtype=str, sort=True)
        BAML_sector = to_list(BAML_sector, dtype=str, sort=True)
        BAML_top_level_sector = to_list(
            BAML_top_level_sector, dtype=str, sort=True
        )
        LGIMA_sector = to_list(LGIMA_sector, dtype=str, sort=True)
        LGIMA_top_level_sector = to_list(
            LGIMA_top_level_sector, dtype=str, sort=True
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
        argspec = getfullargspec(self.subset)
        default_constraints = {
            arg: default
            for arg, default in zip(argspec.args[1:], argspec.defaults)
        }
        user_defined_constraints = locals().copy()
        ignored_kws = {
            "self",
            "df",
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
            df["NewIssueMask"] = new_issue_mask(df)

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
            "rating_risk_bucket": ("RatingRiskBucket", rating_risk_bucket),
            "L3": ("L3", L3),
            "sector": ("Sector", sector),
            "subsector": ("Subsector", subsector),
            "BAML_sector": ("BAMLSector", BAML_sector),
            "BAML_top_level_sector": (
                "BAMLTopLevelSector",
                BAML_top_level_sector,
            ),
            "LGIMA_sector": ("LGIMASector", LGIMA_sector),
            "LGIMA_top_level_sector": (
                "LGIMATopLevelSector",
                LGIMA_top_level_sector,
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
            "benchmark_treasury": ("BMTreasury", benchmark_treasury),
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
            elif constraint == "special_rules":
                subset_index_constraints[
                    constraint
                ] = f"({index_val}) & ({subset_val})"
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
                f'(df["{key}"] >= self._range_vals["{key}"][0]) & '
                f'(df["{key}"] <= self._range_vals["{key}"][1])'
            )
            for key in self._range_vals.keys()
        }
        cat_repl = {
            key: f'(df["{key}"].isin(self._category_vals["{key}"]))'
            for key in self._category_vals.keys()
        }
        flag_repl = {
            key: f'(df["{key}"] == self._flags["{key}"])' for key in self._flags
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
            subset_mask_list.append('(df["Sector"]!="TREASURIES")')
        if drop_municipals:
            subset_mask_list.append('(df["Sector"]!="LOCAL_AUTHORITIES")')

        # Format all other rules.
        for rule in self._all_rules:
            if rule in rule_cols:
                continue  # already added to subset mask
            subset_mask_list.append(repl_dict[rule])

        # Combine formatting rules into single mask and subset DataFrame,
        # and drop temporary columns.
        temp_cols = ["NewIssueMask"]
        if subset_mask_list:
            subset_mask = " & ".join(subset_mask_list)
            df = eval(f"df.loc[{subset_mask}]").drop(
                temp_cols, axis=1, errors="ignore"
            )
        else:
            df = df.drop(temp_cols, axis=1, errors="ignore")

        return self._child_class(
            df=df, name=name, constraints=subset_index_constraints,
        )

    def _child_class(self, **kwargs):
        """
        Return current child class of :class:`BondBasket`.
        """
        class_name = self.__class__.__name__
        if class_name == "BondBasket":
            return BondBasket(market=self.market, index=self.index, **kwargs,)
        elif class_name in {"Account", "Strategy"}:
            child_class = getattr(import_module("lgimapy.data"), class_name)
            return child_class(
                date=self.date, market=self.market, index=self.index, **kwargs,
            )
        else:
            child_class = getattr(import_module("lgimapy.data"), class_name)
            return child_class(market=self.market, index=self.index, **kwargs,)

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

    def calc_dollar_adjusted_spreads(self):
        """
        Adjust spreads by 1 bp tighter (wider) per $ over (under)
        par for BBB's and 0.5 bp for A rateds.
        """
        multiplier = np.ones(len(self.df))
        # Use a 0.5 multiplier for A rated bonds.
        multiplier[self.df["NumericRating"] < 8] = 0.5
        self.df["PX_Adj_OAS"] = self.df["OAS"] - multiplier * (
            self.df["CleanPrice"] - 100
        )

    def cusip_to_bond_name_map(self):
        """dict[str: str]: Mapping of cusip to formatted bond name."""
        cols = ["Ticker", "CouponRate", "MaturityDate"]
        bond_name_map = {}
        for cusip, ticker, coupon, maturity in self.df[cols].itertuples():
            if cusip in bond_name_map:
                continue
            bond = f"{ticker} {coupon:.2f} `{maturity.strftime('%y')}"
            bond_name_map[cusip] = bond

        return bond_name_map

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

    def _on_the_run_issue_years(self, maturity):
        return {
            2: (None, 1),
            3: (None, 1.5),
            5: (None, 1.5),
            7: (None, 2),
            10: (None, 2),
            12: (None, 2),
            20: (None, 4),
            30: (None, 5),
        }[maturity]

    def _on_the_run_maturity_years(self, maturity):
        return {
            2: (None, 2.5),
            3: (2, 3.5),
            5: (3.5, 5.5),
            7: (5.5, 7.5),
            10: (8.25, 11.5),
            12: (10, 12.5),
            20: (17, 22),
            30: (25, 32),
        }[maturity]

    def subset_on_the_runs(self, maturites=(2, 3, 5, 7, 10, 20, 30)):
        """
        Subset current index to only include the on the run bonds.

        Parameters
        ----------
        maturities: Tuple(int), default=(2, 3, 5, 7, 10, 20, 30)
            Maturities to include in new index.

        Returns
        -------
        :class:`BondBasket`:
            :class:`BondBasket` or current child class of
            :class:`BondBasket` subset to on the run bonds only.
        """
        df_list = []
        for maturity in to_list(maturites, dtype=Number):
            maturity_ix = self.subset(
                maturity=self._on_the_run_maturity_years(maturity),
                issue_years=self._on_the_run_issue_years(maturity),
                original_maturity=(maturity, maturity),
            )
            maturity_df = maturity_ix.df.set_index("ISIN")
            on_the_run_isins = (
                maturity_df[["IssueDate", "Ticker"]]
                .groupby("Ticker", observed=True)
                .idxmax()
                .squeeze()
            )
            df_list.append(maturity_ix.subset(isin=on_the_run_isins).df)

        return self._child_class(df=pd.concat(df_list))

    def expand_tickers(self):
        self.expand_bank_tickers()
        self.expand_life_tickers()
        self.expand_utility_tickers()

    def _expand_ticker_rules(self, rules):
        """
        Expand tickers with given rules.

        Parameters
        ----------
        rules: dict[pd.Index, str]
            A dict with keys being pandas index for :attr:`df`
            for respective values of name to append to each ticker.
        """
        self.df["Ticker"] = self.df["Ticker"].astype(str)
        for name, rule in rules.items():
            self.df.loc[rule, "Ticker"] = self.df.loc[rule, "Ticker"] + name
        self.df["Ticker"] = self.df["Ticker"].astype("category")

    def expand_bank_tickers(self):
        """
        Expand bank tickers to include and differentiate by
        Senior and Sub collateral.
        """
        banks = self.df["Sector"] == "BANKING"
        rules = {
            "-Snr": banks & (self.df["CollateralType"] == "UNSECURED"),
            "-Sub": banks & (self.df["CollateralType"] == "SUBORDINATED"),
        }
        self._expand_ticker_rules(rules)

    def expand_life_tickers(self):
        """
        Expand life insurance tickers to include and differentiate
        by FABN, and Senior/Sub collateral.
        """
        life = self.df["Sector"] == "LIFE"
        subs = {"SUBORDINATED", "JR SUBORDINATED"}
        rules = {
            "-FABN": life & (self.df["CollateralType"] == "SECURED"),
            "-Snr": life & (self.df["CollateralType"] == "UNSECURED"),
            "-Sub": life & (self.df["CollateralType"].isin(subs)),
        }
        self._expand_ticker_rules(rules)

    def expand_utility_tickers(self):
        """
        Expand life insurance tickers to include and differentiate
        by FABN, and Senior/Sub collateral.
        """
        utility_sectors = {
            "ELECTRIC",
            "NATURAL_GAS",
            "UTILITY_OTHER",
            "STRANDED_UTILITY",
        }
        utes = self.df["Sector"].isin(utility_sectors)
        rules = {
            "-HoldCo": utes & (self.df["Subsector"] == "HOLDCO"),
            "-OpCo": utes & (self.df["Subsector"] == "OPCO"),
        }
        self._expand_ticker_rules(rules)

    def add_BM_treasuries(self):
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
        year = self.df["Date"].dt.year
        maturity_date = self.df["MaturityDate"].dt.year
        issue_date = self.df["IssueDate"].dt.year
        tenor = maturity_date - year

        loc_7yr = (tenor == 7) & (issue_date == year)
        self.df["BMTreasury"] = tenor.map(bm_treasury).fillna(30).astype("int8")
        self.df.loc[loc_7yr, "BMTreasury"] = 7

    def _numeric_ratings_columns(self):
        cols = ["MoodyRating", "SPRating", "FitchRating"]
        ratings_mat = np.zeros((len(self.df), len(cols)), dtype="object")
        for i, col in enumerate(cols):
            try:
                agency_col = self.df[col].cat.add_categories("NR")
            except (AttributeError, ValueError):
                agency_col = self.df[col]

            ratings_mat[:, i] = agency_col.fillna("NR")
        num_ratings = np.vectorize(self._ratings.__getitem__)(
            ratings_mat
        ).astype(float)
        num_ratings[num_ratings == 0] = np.nan  # json nan value is 0
        return pd.DataFrame(
            num_ratings, index=self.df.index, columns=cols
        ).rename_axis(None)

    @property
    def _rating_risk_buckets(self):
        return [
            "Any AAA/AA",
            "Pure A",
            "Split A/BBB",
            "Pure BBB+/BBB",
            "Any BBB-/BB",
        ]

    def add_rating_risk_buckets(self):
        rating_cols = self._numeric_ratings_columns()
        min_rating = rating_cols.min(axis=1)
        max_rating = rating_cols.max(axis=1)

        # fmt: off
        rating_bucket_locs = {}
        rating_bucket_locs['Any AAA/AA'] = (
            min_rating <= self._convert_single_input_rating("AA-")
        )
        rating_bucket_locs['Pure A'] = (
            (min_rating >= self._convert_single_input_rating("A+"))
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

        self.df["RatingRiskBucket"] = np.nan
        for rating_bucket, loc in rating_bucket_locs.items():
            self.df.loc[loc, "RatingRiskBucket"] = rating_bucket
        self.df["RatingRiskBucket"] = self.df["RatingRiskBucket"].astype(
            "category"
        )

    def market_value_weight(self, col, weight="MarketValue"):
        """
        Market value weight a specified column vs entire
        index market value.

        Parameters
        ----------
        col: str
            Column name to find weighted average for.
        weight: str, default="MarketValue"
            Column to use as weights.

        Returns
        -------
        pd.Series:
            Series of market value weighting for specified column.
        """
        df = self.df[["Date", weight, col]].copy()
        df["mvw_col"] = df[weight] * df[col]
        g = df[["Date", weight, "mvw_col"]].groupby("Date").sum()
        return (g["mvw_col"] / g[weight]).rename(col)

    def SUM(self, col):
        """
        Daily mean for specified column.

        Parameters
        ----------
        col: str
            Column to compute mean on.

        Returns
        -------
        pd.Series:
            Daily mean with datetime index.
        """
        cols = ["Date", col]

        def daily_sum(df):
            """Sum for single day."""
            return np.sum(df[col])

        return self.df[cols].groupby("Date").apply(daily_sum)

    def MEAN(self, col, weights="MarketValue"):
        """
        Daily mean for specified column.

        Parameters
        ----------
        col: str
            Column to compute mean on.
        weights: str, optional, default='MarketValue'
            Column to use as weights. If ``None`` no
            weights are used.

        Returns
        -------
        pd.Series:
            Daily mean with datetime index.
        """
        cols = ["Date", col]
        if weights is not None:
            cols.append(weights)

        def daily_mean(df):
            """Mean for single day."""
            daily_weights = weights if weights is None else df[weights]
            return mean(df[col], weights=daily_weights)

        return self.df[cols].groupby("Date").apply(daily_mean)

    def MEDIAN(self, col, q=50, weights="MarketValue"):
        """
        Daily median specified column.

        Parameters
        ----------
        col: str
            Column to compute median on.
        q: int, default=50
            qth percentile to return, default of 50 is median.
        weights: str, optional, default='MarketValue'
            Column to use as weights. If ``None`` no
            weights are used.

        Returns
        -------
        pd.Series:
            Daily median with datetime index.
        """
        cols = ["Date", col]
        if weights is not None:
            cols.append(weights)

        def daily_median(df):
            """Median for single day."""
            daily_weights = weights if weights is None else df[weights]
            return percentile(df[col], weights=daily_weights, q=q)

        return self.df[cols].groupby("Date").apply(daily_median)

    def STD(self, col, weights="MarketValue"):
        """
        Daily standard deviation for specified column.

        Parameters
        ----------
        col: str
            Column to compute standard deviation on.
        weights: str, optional, default='MarketValue'
            Column to use as weights. If ``None`` no
            weights are used.

        Returns
        -------
        pd.Series:
            Daily standard deviation with datetime index.
        """
        cols = ["Date", col]
        if weights is not None:
            cols.append(weights)

        def daily_std(df):
            """Standard deviation for single day."""
            daily_weights = weights if weights is None else df[weights]
            return std(df[col], weights=daily_weights)

        return self.df[cols].groupby("Date").apply(daily_std)

    def RSD(self, col, weights="MarketValue"):
        """
        Daily relative standard deviation for specified column.

        Parameters
        ----------
        col: str
            Column to perform RSD on.
        weights: str, optional, default='MarketValue'
            Column to use as weights. If ``None`` no
            weights are used.

        Returns
        -------
        pd.Series:
            Daily RSD with datetime index.
        """
        cols = ["Date", col]
        if weights is not None:
            cols.append(weights)

        def daily_rsd(df):
            """RSD for single day."""
            daily_weights = weights if weights is None else df[weights]
            return mean(df[col], weights=daily_weights) / std(
                df[col], weights=daily_weights
            )

        return self.df[cols].groupby("Date").apply(daily_rsd)

    def IQR(self, col, qrange=[25, 75], weights="MarketValue"):
        """
        Daily interquartile range for specified column.

        Parameters
        ----------
        col: str
            Column to compute IQR on.
        qrange: list[int], default=[25, 75]
            Quartile range to use.
        weights: str, optional, default='MarketValue'
            Column to use as weights. If ``None`` no
            weights are used.

        Returns
        -------
        pd.Series:
            Daily IQR with datetime index.
        """
        cols = ["Date", col]
        if weights is not None:
            cols.append(weights)

        def daily_iqr(df):
            """QCD for single day."""
            daily_weights = weights if weights is None else df[weights]
            Q1, Q3 = percentile(df[col], weights=daily_weights, q=qrange)
            return Q3 - Q1

        return self.df[cols].groupby("Date").apply(daily_iqr)

    def QCD(self, col, qrange=[25, 75], weights="MarketValue"):
        """
        Daily quartile coefficient of dispersion for specified column.

        Parameters
        ----------
        col: str
            Column to compute QCD on.
        qrange: list[int], default=[25, 75]
            Quartile range to use.
        weights: str, optional, default='MarketValue'
            Column to use as weights. If ``None`` no
            weights are used.

        Returns
        -------
        pd.Series:
            Daily QCD with datetime index.
        """
        cols = ["Date", col]
        if weights is not None:
            cols.append(weights)

        def daily_qcd(df):
            """QCD for single day."""
            daily_weights = weights if weights is None else df[weights]
            Q1, Q3 = percentile(df[col], weights=daily_weights, q=qrange)
            return (Q3 - Q1) / (Q3 + Q1)

        return self.df[cols].groupby("Date").apply(daily_qcd)

    def OAS(self):
        """
        pd.Series:
            Daily market value weighted OAS with datetime index.
        """
        return self.market_value_weight("OAS")

    def market_value_median(self, col):
        """
        Daily market value weighted median specified column.

        Parameters
        ----------
        col: str
            Column to perform QCD on.

        Returns
        -------
        pd.Series:
            Daily median with datetime index.
        """
        cols = ["Date", "MarketValue", col]

        def daily_median(df):
            """Median for single day."""
            return percentile(df[col], weights=df["MarketValue"], q=50)

        return self.df[cols].groupby("Date").apply(daily_median)

    def is_empty(self):
        return not len(self.df)

    def add_change(self, col, date, db):
        """
        Add column change since specified date to :attr:`df`.

        Parameters
        ----------
        col: str
            Column to find change on
        date: str or Datetime
            Date from which to compute change.1
        db: :class:`Database`
            Database to load data.

        Returns
        -------
        List[str]:
            List of newly created column names.
        """
        if isinstance(date, pd.Timestamp):
            from_date = date
            date_str = date.strftime("%m-%d-%Y")
        else:
            from_date = db.date(date)
            date_str = date

        db.load_market_data(date=from_date)
        prev_ix = db.build_market_index(isin=self.isins)
        prev_oas_d = dict(zip(prev_ix.df["ISIN"], prev_ix.df[col]))
        self.df[f"{col}_{date_str}"] = self.df["ISIN"].map(prev_oas_d)
        self.df[f"{col}_abs_Change_{date_str}"] = (
            self.df[f"{col}"] - self.df[f"{col}_{date_str}"]
        )
        self.df[f"{col}_pct_Change_{date_str}"] = (
            self.df[f"{col}"] / self.df[f"{col}_{date_str}"]
        )

        return [
            f"{col}_{date_str}",
            f"{col}_abs_Change_{date_str}",
            f"{col}_pct_Change_{date_str}",
        ]

    def add_bond_description(self):
        self.df["BondDescription"] = (
            self.df["Ticker"].astype(str)
            + " "
            + self.df["CouponRate"].apply(lambda x: f"{x:.2f}")
            + " "
            + self.df["MaturityDate"].apply(lambda x: f"`{x:%y}").astype(str)
        )
