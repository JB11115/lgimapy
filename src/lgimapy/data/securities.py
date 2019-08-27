import datetime as dt
import warnings
from collections import defaultdict
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.optimize import fsolve

from lgimapy.bloomberg import (
    get_accrual_date,
    get_bloomberg_ticker,
    get_cashflows,
    get_issue_price,
    get_settlement_date,
)
from lgimapy.data import concat_index_dfs, new_issue_mask
from lgimapy.utils import (
    check_all_equal,
    load_json,
    mkdir,
    replace_multiple,
    root,
)

# %%
class Bond:
    """
    Class for bond math and manipulation given current
    state of the bond.

    Parameters
    ----------
    s: pd.Series
        Single bond row from :attr:`Index.df`.
    """

    def __init__(self, s):
        self.s = s
        self.__dict__.update({k: v for k, v in zip(s.index, s.values)})

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.cusip} - "
            f"{self.Date.strftime('%m/%d/%Y')})"
        )

    @property
    def cusip(self):
        """str: Cusip of :class:`Bond`."""
        return self.CUSIP

    @property
    @lru_cache(maxsize=None)
    def cash_flows(self):
        """
        pd.Series:
            Load/scrape cash flows with datetime index for
            the bond from current bond date.
        """
        cash_flows = get_cashflows(self.cusip)
        return cash_flows[cash_flows.index > self.Date]

    @property
    @lru_cache(maxsize=None)
    def coupon_dates(self):
        """
        List[datetime]:
            Memoized list of timestamps for all coupons.
        """
        # TODO: use scraped dates when apporiate.
        # coupon_dates = get_coupon_dates(self.CUSIP)
        # return coupon_dates[coupon_dates > self.Date]

        return self._theoretical_coupon_dates()
        # return self.cash_flows.index

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

    def _theoretical_coupon_dates(self):
        """
        Estimate timestamps for all coupons.

        Returns
        -------
        dates: List[datetime].
            Theoretical timestamps for all coupons.

        Notes
        -----
        All coupons are assumed to be on either the 15th or
        last day of the month, business days and holidays are ignored.
        """
        i_date = self.IssueDate
        dates = []
        if 14 <= i_date.day <= 19:
            # Mid month issue and coupons.
            # Make first first coupon 6 months out.
            temp = i_date.replace(day=15) + relativedelta(months=6)
            while temp <= self.MaturityDate + dt.timedelta(5):
                if temp > self.Date:
                    dates.append(temp)
                temp += relativedelta(months=6)
        else:
            # End of the month issue and coupons.
            # Make first coupon the first of the next month 6 months out.
            temp = (i_date + relativedelta(months=6, days=5)).replace(day=1)
            while temp <= self.MaturityDate + dt.timedelta(5):
                if temp > self.Date - dt.timedelta(1):
                    # Use last day of prev month.
                    dates.append(temp - dt.timedelta(1))
                temp += relativedelta(months=6)

        # Use actual maturity date as final date.
        try:
            dates[-1] = self.MaturityDate
        except IndexError:
            # No coupon dates, only maturity date remaining.
            dates = [self.MaturityDate]

        return dates

    @property
    @lru_cache(maxsize=None)
    def coupon_years(self):
        """
        ndarray:
            Memoized time in years from :attr:`Bond.Date`
            for all coupons.
        """
        return np.array(
            [(cd - self.Date).days / 365 for cd in self.cash_flows.index]
        )

    @property
    def next_coupon_date(self):
        """datetime: Date of next coupon."""
        return self.cash_flows.index[0]

    # @property
    # @lru_cache(maxsize=None)
    # def cash_flows(self):
    #     """
    #     ndarray:
    #         Memoized cash flows to be acquired on
    #         :attr:`Bond.coupon_dates`.
    #     """
    #     cash_flows = self.CouponRate / 2 + np.zeros(len(self.coupon_dates))
    #     cash_flows[-1] += 100
    #     return cash_flows

    def _ytm_func(self, y, price=None):
        """Yield to maturity error function for solver, see `ytm`."""
        error = (
            sum(
                cf * np.exp(-y * t)
                for cf, t in zip(self.cash_flows, self.coupon_years)
            )
            - price
        )
        return error

    @property
    @lru_cache(maxsize=None)
    def ytm(self):
        """
        float:
            Memoized yield to maturity of the bond at market
            dirty price using true coupon values and dates.
        """
        x0 = 0.02  # initial guess
        return fsolve(self._ytm_func, x0, args=(self.DirtyPrice))[0]

    def theoretical_ytm(self, price):
        """
        Calculate yield to maturity of the bond using true coupon
        values and dates and specified price.

        Parameters
        ----------
        price: float:
            Price of the bond to use when calculating yield to maturity.

        Returns
        -------
        float:
            Yield to maturity of the bond at specified price.
        """
        x0 = 0.02  # initial guess
        return fsolve(self._ytm_func, x0, args=(price))[0]


class TBond(Bond):
    """
    Class for treasury bond math and manipulation given current
    state of the bond.

    Parameters
    ----------
    s: pd.Series
        Single bond row from :attr:`Index.df`.
    """

    def __init__(self, s):
        super().__init__(s)

    @property
    @lru_cache(maxsize=None)
    def AccrualDate(self):
        return get_accrual_date(self.cusip)

    @property
    def DirtyPrice(self):
        """float: Dirty price ($)."""
        return self.CleanPrice + self.AccruedInterest

    def calculate_price(self, rfr):
        """
        Calculate theoretical price of the bond with given
        risk free rate.

        Parameters
        ----------
        rfr: float
            Continuously compouned risk free rate used to discount
            cash flows.

        Returns
        -------
        float:
            Theoretical price ($) of :class:`TBond`.
        """
        return sum(
            cf * np.exp(-rfr * t)
            for cf, t in zip(self.cash_flows, self.coupon_years)
        )

    def calculate_price_with_curve(self, curve):
        """
        Calculate thoretical price of bond with given
        zero curve.

        Parameters
        ----------
        curve: pd.Series
            Yield curve values with maturity index.

        Returns
        -------
        float:
            Theoretical price ($) of :class:`TBond`.
        """
        yields = np.interp(self.coupon_years, curve.index, curve.values)
        return sum(
            cf * np.exp(-y * t)
            for cf, t, y in zip(self.cash_flows, self.coupon_years, yields)
        )

    @property
    @lru_cache(maxsize=None)
    def settlement_date(self):
        """datetime: Next settlment date from current date."""
        return get_settlement_date(self.Date)

    @property
    @lru_cache(maxsize=None)
    def all_cash_flows(self):
        """pd.Series: Load/scrape all cash flows with datetime index."""
        return get_cashflows(self.cusip)

    @property
    @lru_cache(maxsize=None)
    def cash_flows(self):
        """
        pd.Series:
            Get future cashflows from current date. Check settlement
            date to verify the next coupon is correct.
        """
        # Get temporary next coupon date.
        cash_flows = self.all_cash_flows
        self.cusip
        next_coupons = cash_flows[cash_flows.index > self.Date]
        next_coupon_date = next_coupons.index[0]

        if self.settlement_date == get_settlement_date(next_coupon_date):
            self._day_count_dt = self.settlement_date
            self._day_offset = 0
        else:
            self._day_count_dt = self.Date
            self._day_offset = 1

        return cash_flows[cash_flows.index > self._day_count_dt]

    @property
    @lru_cache(maxsize=None)
    def coupon_dates(self):
        """
        List[datetime]:
            Memoized list of timestamps for all coupons.
        """
        return list(self.cash_flows.index)

    @property
    @lru_cache(maxsize=None)
    def AccruedInterest(self):
        """
        Calculate accrued interest with methodology that
        matches Port/Point analytics for treasuries.

        Returns
        -------
        float:
            Accrued interest for $100 face value.
        """
        # Get previous and next coupons.
        next_coupon_date = self.coupon_dates[0]
        next_coupons = self.cash_flows.copy()
        next_coupon = next_coupons[0]
        next_coupon = next_coupon if next_coupon < 100 else next_coupon - 100
        prev_coupon_dates = [
            date
            for date in self.all_cash_flows.index
            if date not in self.coupon_dates
        ]
        try:
            prev_coupon_date = prev_coupon_dates[-1]
        except IndexError:
            # No previous coupons, so use issue date.
            prev_coupon_date = self.IssueDate

        # Calculate accrued interest using the bonds
        return (
            next_coupon
            * ((self._day_count_dt - prev_coupon_date).days + self._day_offset)
            / (next_coupon_date - prev_coupon_date).days
        )


class Index:
    """
    Class for indexes built by :class:`Database`.

    Parameters
    ----------
    index_df: pd.DataFrame
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
        self._day_cache = {}
        self._day_key_cache = defaultdict(str)

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
        return list(pd.to_datetime(self.df["Date"].unique()))

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
            a[i] = np.sum(df["AmountOutstanding"] * df["DirtyPrice"])
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
        date = pd.to_datetime(date)
        # Create cache key of all columns added to :attr:`Index.df`.
        cache_key = "_".join(list(self.df))
        if cache_key == self._day_key_cache[date]:
            # No new column changes since last accessed,
            # treat like normal cache.
            try:
                df = self._day_cache[date]
            except KeyError:
                df = self.df[self.df["Date"] == date].drop_duplicates("CUSIP")
                self._day_cache[date] = df
        else:
            # New columns added since last accessed,
            # update cache and cache key.
            df = self.df[self.df["Date"] == date].drop_duplicates("CUSIP")
            self._day_cache[date] = df
            self._day_key_cache[date] = cache_key

        if as_index:
            return Index(df, self.name)
        else:
            return df

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
        current_date = pd.to_datetime(date)
        prev_date = self.dates[list(self.dates).index(current_date) - 1]
        if prev_date == self.dates[-1]:
            msg = "First day in index history, synthetic day not possible."
            raise IndexError(msg)
        current_df = self.day(current_date)
        prev_df = self.day(prev_date)
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
            for rule in special_rules:
                subset_mask_list.append(replace_multiple(rule, repl_dict))
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
        maturity_map = {2: 0, 3: 2, 5: 3, 8: 5, 7: 5, 10: 7, 30: 10}

        is_treasury = self.df["BTicker"].isin({"US/T", "T"})
        is_zero_coupon = (self.df["CouponType"] == "ZERO COUPON") | (
            (self.df["CouponType"] == "FIXED") & (self.df["CouponRate"] == 0)
        )
        is_noncall = self.df["CallType"] == "NONCALL"
        has_normal_mat = self.df["OriginalMaturity"].isin(maturity_map.keys())
        matures_in_more_than_a_week = self.df["MaturityYears"] > 1 / 52
        bad_cusips = ["912820FL6"]
        is_not_bad_cusip = ~self.df["CUSIP"].isin(bad_cusips)
        df = self.df[
            (is_treasury | is_zero_coupon)
            & is_noncall
            & has_normal_mat
            & matures_in_more_than_a_week
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
        ignore_holidays=True,
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
        ignore_holidays: bool, default=True
            If True, ignore holidays.

        Returns
        -------
        hist_df: pd.DataFrame
            DataFrame with datetime index, CUSIP columns, and price values.
        """
        cusips = set(self.cusips)
        dates = self.dates[1:] if synthetic else self.dates
        if ignore_holidays:
            dates = [d for d in dates if d not in self._holiday_dates]
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

        hist_df = pd.DataFrame(hist_d, index=dates)
        return hist_df

    def get_cusip_history(self, cusip, ignore_holidays=True):
        """
        Get full history for specified cusip.

        Parameters
        ----------
        cusip: str
            Specified cusip.
        ignore_holidays: bool, default=True
            If True, ignore holidays.

        Returns
        -------
        hist_df: pd.DataFrame
            DataFrame with datetime index and :attr:`Index.df`
            columns for specified cusip.
        """
        if ignore_holidays:
            dates = [d for d in self.dates if d not in self._holiday_dates]
        else:
            dates = self.dates

        return pd.concat(
            [
                pd.DataFrame(self.day(d).loc[cusip, :]).T.set_index("Date")
                for d in dates
            ]
        )

    def market_value_weight(self, col, synthetic=False, ignore_holidays=True):
        """
        Market value weight a specified column vs entire
        index market value.

        Parameters
        ----------
        col: str
            Column name to weight, (e.g., 'OAS').
        synthethic: bool, default=False
            If True, use synthethic day data.
        ignore_holidays: bool, default=True
            If True, ignore holidays.

        Returns
        -------
        pd.Series:
            Series of market value weighting for specified column.
        """
        dates = self.dates[1:] if synthetic else self.dates
        if ignore_holidays:
            dates = [d for d in dates if d not in self._holiday_dates]

        a = np.zeros(len(dates))
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i, date in enumerate(dates):
            df = self.synthetic_day(date) if synthetic else self.day(date)
            df = df.dropna(subset=["AmountOutstanding", "DirtyPrice", col])
            a[i] = np.sum(df["AmountOutstanding"] * df["DirtyPrice"] * df[col])
            a[i] /= np.sum(df["AmountOutstanding"] * df["DirtyPrice"])
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
                new_val = rating_series.values[loc + 1]
                old_val = rating_series.values[loc]
                last_date = rating_df[cusip].dropna().index[loc]
                change_dict["cusip"].append(cusip)
                change_dict["date"].append(date)
                change_dict["change"].append(change_val)
                change_dict["new_val"].append(new_val)
                change_dict["old_val"].append(old_val)
                change_dict["sector"].append(self.df.loc[date, "Sector"])
                change_dict["subsector"].append(self.df.loc[date, "Subsector"])

        change_df = pd.DataFrame(change_dict)
        return change_df

    def subset_value_by_rating(self, col, save=False, fid=None, path=None):
        """
        Find the market vaue weighted value of specified
        column by rating.

        Parameters
        ----------
        col: str
            Column from Index.df to market value weight.
        save: bool
            If true, store resulting DataFrame to fid.
        fid: str, default=None
            File name to store file, by default is input `col`.
        path: Path, default=None
            Path to store file, by default is `./data/subset_by_rating`.

        Returns
        -------
        subset_df: pd.DataFrame
            if save is false, return DataFrame of results
            with datetime index and numeric rating columns.
        """
        path = root("data/subset_by_rating") if path is None else path
        fid = col if fid is None else fid

        ratings = sorted(list(set(self.df["NumericRating"])))
        subset_dict = defaultdict(list)
        for date in self.dates:
            day_df = self.day(date)
            for rating in ratings:
                df = day_df[day_df["NumericRating"] == rating].copy()
                mvw = self.market_value_weight(
                    col, df=df, ignore_holidays=False
                )
                subset_dict[rating].append(mvw)

        cols = [int(rating) for rating in ratings]
        subset_df = pd.DataFrame(subset_dict, index=self.dates, columns=cols)
        if save:
            self._save_subset_by_rating(subset_df, fid, path)
        else:
            return subset_df

    def _save_subset_by_rating(self, subset_df, fid, path):
        """
        Save subset_df to `../data/subset_by_rating/{col}.csv`
        creating the file if it doesn't exist, and overwriting
        data which already exists.

        Parameters
        ----------
        subset_df: pd.DataFrame
            subset_df from `Index.subset_value_by_rating` with
            datetime index and numeric rating columns.
        fid: str
            File name to store file.
        path: Path
            Path to store file.
        """
        fid = path.joinpath(f"{fid}.csv")
        mkdir(path)
        try:
            df = pd.read_csv(
                fid, index_col=0, parse_dates=True, infer_datetime_format=True
            )
        except FileNotFoundError:
            # File does not exist, create a new one.;
            subset_df.to_csv(fid)
        else:
            # File exists, append to it and sort.
            old_df = df[~df.index.isin(subset_df.index)]
            old_df.columns = [int(col) for col in old_df.columns]
            new_df = old_df.append(subset_df)
            new_df.sort_index(inplace=True)
            new_df.to_csv(fid)

    @property
    @lru_cache(maxsize=None)
    def _tsy_df(self):
        """pd.DataFrame: historical treasury curve DataFrame."""
        tsy_fid = root("data/treasury_curve_params.csv")
        tsy_df = pd.read_csv(
            tsy_fid, index_col=0, parse_dates=True, infer_datetime_format=True
        )
        tsy_df = tsy_df[
            (tsy_df.index >= self.dates[0]) & (tsy_df.index <= self.dates[-1])
        ].copy()
        return tsy_df

    def cusip_excess_returns(self, cusips):
        """
        Calculate excess returns for specified cusips.

        Parameters
        ----------
        cusips: str, List[str], default=None
            Specified cusip(s) to compute excess returns for.

        Returns
        -------
        pd.DataFrame:
            DatFrame with datetime index, cusips columns,
            and excess return values.
        """
        if isinstance(cusips, str):
            cusips = [cusips]  # ensure cusips is list

        # Load treasury DataFrame and get KRDs and total returns.
        tsy_df = self._tsy_df
        tsy_krd_cols = [col for col in list(tsy_df) if "KRD" in col]
        tsy_tret_cols = [col for col in list(tsy_df) if "tret" in col]
        self._tsy_krds = tsy_df[tsy_krd_cols].values
        self._tsy_krd_trets = tsy_df[tsy_tret_cols].values

        # Store list of KRD columns for :attr:`Index.df`.
        self._krd_cols = [col for col in list(self.df) if "KRD" in col]

        ex_ret_dates = self.dates[1:]
        ex_rets = np.zeros([len(ex_ret_dates), len(cusips)])
        for i, cusip in enumerate(cusips):
            ex_rets[:, i] = self._single_cusip_excess_returns(cusip)

        return pd.DataFrame(ex_rets, index=ex_ret_dates, columns=cusips)

    def _single_cusip_excess_returns(self, cusip):
        """
        Calculate excess returns for a single cusip.

        Parameters
        ----------
        cusip: str
            Specified cusip to compute excess returns for.

        Returns
        -------
        """
        # Get KRDs and total returns from cusip.
        df = self.df[self.df.index == cusip]
        cusip_krds = df[self._krd_cols].values
        df[[*self._krd_cols, "Date"]]
        cusip_trets = (df["DirtyPrice"][1:] / df["DirtyPrice"][:-1]).values - 1
        if len(cusip_trets) < len(self.dates) - 1:
            # NaNs are present, find correct starting ix for trets
            # and KRDs and pad arrays with NaNs.
            start_ix = self.dates.index(df["Date"][0])
            temp_trets = np.nan * np.empty(len(self.dates) - 1)
            temp_trets[start_ix : start_ix + len(df) - 1] = cusip_trets
            cusip_trets = temp_trets
            temp_krds = np.nan * np.empty([len(self.dates), 6])
            temp_krds[start_ix : start_ix + len(df), :] = cusip_krds
            cusip_krds = temp_krds

        # Match hypothetical portfolios by KRDs, then add
        # cash component to make weight sum to 1.
        weights = np.zeros(self._tsy_krd_trets.shape)
        weights[:, :-1] = cusip_krds / self._tsy_krds
        weights[:, -1] = 1 - np.sum(weights, axis=1)
        tsy_trets = np.sum(weights * self._tsy_krd_trets, axis=1)[1:]
        return cusip_trets - tsy_trets

    def compute_total_returns(self):
        """
        Vectorized implementation for computing total returns
        for all cusips in index, adjusting for coupon delivery
        and reinvestment.

        Appends results as new column in :attr:`Index.df`.
        """
        # Stop computating if already performed.
        if "total_return" in list(self.df):
            return

        # Compute total returns without adjusting for coupons.
        price_df = self.get_value_history("DirtyPrice")
        cols = list(price_df)
        price = price_df.values
        tret = price[1:] / price[:-1] - 1

        # Find dates when coupons were paid. For first day in
        # dataset find points where accrued interest is 0. For all
        # others use 0 value or any decrease in accrued interest.
        accrued = self.get_value_history("AccruedInterest")[cols].values
        accrued_mask = np.where(accrued == 0, 1, 0)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        accrued_diff = np.where(np.diff(accrued, axis=0) < 0, 1, 0)
        warnings.simplefilter("default", category=RuntimeWarning)
        accrued_mask[1:] = np.maximum(accrued_mask[1:], accrued_diff)

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
        self.df["total_return"] = np.NaN
        for date in tret_df.index:
            cusips = list(self.day(date).index)
            self.df.loc[self.df["Date"] == date, "total_return"] = tret_df.loc[
                date, cusips
            ].values

    def compute_excess_returns(self):
        """
        Compute excess returns for all cusips in index,
        appending result to :attr:`Index.df`.
        """
        # Stop computating if already performed.
        if "excess_return" in list(self.df):
            return

        # Compute total returns.
        self.compute_total_returns()

        # Load treasury DataFrame and save required column names.
        tsy_df = self._tsy_df
        tsy_krd_cols = [col for col in list(tsy_df) if "KRD" in col]
        tsy_tret_cols = [col for col in list(tsy_df) if "tret" in col]
        krd_cols = [col for col in list(self.df) if "KRD" in col]

        # Compute and append excess returns iteratively for each day.
        self.df["excess_return"] = np.NaN
        for date in self.dates[1:]:
            df = self.day(date)
            # Calculate hypothetical treasury weights for each cusip.
            weights = np.zeros([len(df), 7])
            weights[:, :-1] = (
                df[krd_cols].values
                / tsy_df.loc[date, tsy_krd_cols].values[None, :]
            )
            # Add cash component to make weights sum to 1.
            weights[:, -1] = 1 - np.sum(weights, axis=1)
            tsy_trets = np.sum(
                weights * tsy_df.loc[date, tsy_tret_cols].values, axis=1
            )
            ex_rets = df["total_return"].values - tsy_trets
            self.df.loc[self.df["Date"] == date, "excess_return"] = ex_rets

    def aggregate_excess_returns(self, start_date):
        """
        Aggregate excess returns since start date.

        Parameters
        ----------
        start_date: datetime
            Date to start aggregating returns.

        Returns
        -------
        float or pd.Seires
            Aggregated excess returns for either full index
            or individual cusips.
        """
        # Find excess and total returns in date range.
        self.compute_excess_returns()
        ex_rets = self.market_value_weight(
            "excess_return", ignore_holidays=False
        )
        t_rets = self.market_value_weight("total_return", ignore_holidays=False)
        ex_rets = ex_rets[ex_rets.index > pd.to_datetime(start_date)]
        t_rets = t_rets[t_rets.index > pd.to_datetime(start_date)]
        # Calculate implied treasy returns.
        tsy_t_rets = t_rets - ex_rets
        # Calculate total returns over period.
        total_ret = np.prod(1 + t_rets) - 1
        tsy_total_ret = np.prod(1 + tsy_t_rets) - 1
        return total_ret - tsy_total_ret

    def add_issue_prices(self):
        """
        Add the issue price on the pricing date for
        all bonds priced within the index timeframe.
        """

        # Find the first day with data for each cusip that is a new issue.
        df_first_day = self.df.drop_duplicates(subset="CUSIP", keep="first")
        df_first_day = df_first_day[df_first_day["IssueYears"] <= 0.1]

        def prev_date(date, date_list):
            """datetime: returns previous trade date."""
            ix = date_list.index(date)
            if ix > 0:
                return date_list[ix - 1]
            else:
                return date_list[0] - timedelta(1)

        # Convert each date to the previous trading date
        first_dates = list(pd.to_datetime(df_first_day["Date"].unique()))
        pricing_date = {d: prev_date(d, first_dates) for d in first_dates}
        df_first_day["Date"] = [pricing_date[d] for d in df_first_day["Date"]]

        # Convert the prices to the issue price. Try using both
        # cusip and ISIN due to both giving NaNs occasionally.
        # Fill any remaining nans with $100 issue price.
        issue_prices = get_issue_price(df_first_day.index)
        for col in ["CleanPrice", "DirtyPrice"]:
            df_first_day[col] = issue_prices
            df_first_day[col].fillna(100, inplace=True)

        # Combine the issue price data with the full DataFrame
        self.df = self.df.append(df_first_day)
        self.df.sort_values("Date", inplace=True)
        self.clear_day_cache()


def main():
    from lgimapy.data import Database

    db = Database()
    dates = db.all_dates

    [d for d in dates if d > pd.to_datetime("1/26/2005")]
    db.load_market_data(date="1/27/2005")
    ix = db.build_market_index(treasuries=True)
    ix = Index(ix.df)
    ix.clean_treasuries()

    bonds = [TBond(i) for _, i in ix.df.iterrows()]
    self = TBond(ix.df.iloc[85, :])
    self.cusip
    for i, b in enumerate(bonds):
        print(i, b.cusip)
        a = b.DirtyPrice
