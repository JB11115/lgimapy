import datetime as dt
import warnings
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from functools import lru_cache

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from scipy.optimize import fsolve

from lgimapy.bloomberg import (
    get_accrual_date,
    get_cashflows,
    get_settlement_date,
)
from lgimapy.data import clean_dtypes, concat_index_dfs
from lgimapy.utils import load_json, root


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
        self.s = s.copy()
        self.__dict__.update(s.to_dict())

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"({self.Ticker} {self.CouponRate:.2f} `{self.MaturityDate:%y}: "
            f"{self.cusip}) - {self.Date:%m/%d/%Y}"
        )

    @property
    @lru_cache(maxsize=None)
    def df(self):
        return clean_dtypes(self.s.to_frame().T)

    @property
    def cusip(self):
        """str: Cusip of :class:`Bond`."""
        return self.CUSIP

    @property
    def isin(self):
        """str: Cusip of :class:`Bond`."""
        return self.ISIN

    @property
    @lru_cache(maxsize=None)
    def all_cash_flows(self):
        """pd.Series: Load/scrape all cash flows with datetime index."""
        return get_cashflows(self.cusip, maturity_date=self.MaturityDate)

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
        return cash_flows[cash_flows.index > self.Date]

    @property
    @lru_cache(maxsize=None)
    def _cf(self):
        """
        [n x 1] np.array:
            Future cash flow values.
        """
        # Get temporary next coupon date.
        return self.cash_flows.values

    @property
    def _t(self):
        """
        [n x 1] ndarray:
            Memoized time in years from :attr:`Bond.Date`
            for all coupons.
        """
        return self.coupon_years

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
        fid = root("data/US/trade_dates.parquet")
        return pd.read_parquet(fid)

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

    def _ytm_func(self, y, price=None):
        """float: Yield to maturity error function for solver."""
        return (self._cf @ np.exp(-y * self._t)) - price

    @property
    @lru_cache(maxsize=None)
    def ytm(self):
        """
        float:
            Memoized yield to maturity of the bond at market
            dirty price using true coupon values and dates.
        """
        x0 = 0.02  # initial guess
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=RuntimeWarning)
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
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=RuntimeWarning)
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
            Continuously compounded risk free rate used to discount
            cash flows.

        Returns
        -------
        float:
            Theoretical price ($) of :class:`TBond`.
        """
        return self._cf @ np.exp(-rfr * self._t)

    def calculate_price_with_curve(self, curve):
        """
        Calculate theoretical price of bond with given
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
        yields = np.interp(self._t, curve.index, curve.values)
        return self._cf @ np.exp(-yields * self._t)

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


class SyntheticTBill:
    """
    Synthetic treasury bond with yield equal to
    the fed funds midpoint. Useful for fitting
    treasury curve front end.
    """

    def __init__(self, date, ytm, days_to_maturity):
        self.Date = pd.to_datetime(date)
        self.ytm = ytm
        self.MaturityYears = days_to_maturity / 365
        self.OAD = self.MaturityYears
        self.coupon_years = np.array([self.MaturityYears])
        self.coupon_dates = [self.Date + timedelta(days_to_maturity)]
        self.cash_flows = pd.Series([100], index=self.coupon_dates)

        self.BTicker = self.Ticker = "B"
        self.cusip = f"{days_to_maturity}D"

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"({self.cusip}) - {self.Date:%m/%d/%Y}"
        )

    @property
    @lru_cache(maxsize=None)
    def DirtyPrice(self):
        """
        Calculate theoretical price of the bond
        using the fed funds rate ytm.

        Returnswe
        -------
        float:
            Theoretical price ($) of :class:`TBond`.
        """
        return self._cf @ np.exp(-self.ytm * self._t)

    def _ytm_func(self, y, price=None):
        """float: Yield to maturity error function for solver."""
        return (self._cf @ np.exp(-y * self._t)) - price

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
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=RuntimeWarning)
            return fsolve(self._ytm_func, x0, args=(price))[0]

    def calculate_price_with_curve(self, curve):
        """
        Calculate theoretical price of bond with given
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
        yields = np.interp(self._t, curve.index, curve.values)
        return self._cf @ np.exp(-yields * self._t)

    @property
    @lru_cache(maxsize=None)
    def _cf(self):
        """
        [n x 1] np.array:
            Future cash flow values.
        """
        # Get temporary next coupon date.
        return self.cash_flows.values

    @property
    def _t(self):
        """
        [n x 1] ndarray:
            Memoized time in years from :attr:`Bond.Date`
            for all coupons.
        """
        return self.coupon_years


class TreasuryCurve:
    """
    Class for loading treasury yield curve and calculating yields
    for specified dates and maturities.

    Parameters
    ----------
    date: datetime, default=None
        Default date to use for curves and KRD parameters.
    """

    def __init__(self, date=None):
        if date is not None:
            self.set_date(date)
        t_mats = [0.5, 2, 5, 10, 20, 30]
        self._t_mats = np.array(t_mats)
        self._KRD_cols = [f"KRD_{t}" for t in t_mats]
        self._coupon_cols = [f"c_{t}" for t in t_mats]
        self._tret_cols = [f"tret_{t}" for t in t_mats] + ["tret_cash"]

    @property
    @lru_cache(maxsize=None)
    def trade_dates(self):
        """List[datetime]: List of traded dates."""
        return self._curves_df.index

    @property
    @lru_cache(maxsize=None)
    def _curves_df(self):
        return pd.read_parquet(root("data/treasury_curves.parquet"))

    @property
    @lru_cache(maxsize=None)
    def _params_df(self):
        return pd.read_parquet(root("data/treasury_curve_krd_params.parquet"))

    def set_date(self, date):
        """Set default date."""
        self._date = pd.to_datetime(date)

    def yields(self, t=None, date=None):
        """
        Get yields for given date and maturity values.

        Parameters
        ----------
        t: float or ndarray[float], default=None
            Maturites to interpolate points on the curve for.
            If none, the entire curve is returned.
        date: datetime, default=None
            Date to return curve for. If None, the default
            date is used.

        Returns
        -------
        pd.Series:
            Yield curve values with maturity (yrs) index.
        """
        date = pd.to_datetime(date) if date is not None else self._date
        if date is None:
            raise ValueError("A date must be specified.")
        try:
            curve = self._curves_df.loc[date, :]
        except KeyError:
            raise ValueError("The specified date is not a traded date.")

        if t is None:
            return curve
        else:
            if type(t) in [int, float]:
                t = np.array([t])
            return pd.Series(np.interp(t, curve.index, curve.values), index=t)

    def KRDs(self, date=None):
        """
        Get yields for given date and maturity values.

        Parameters
        ----------
        date: datetime, default=None
            Date to return curve for. If None, the default
            date is used.

        Returns
        -------
        ndarray:
            Key rate durations for specified date.
        """
        date = pd.to_datetime(date) if date is not None else self._date
        if date is None:
            raise ValueError("A date must be specified.")
        try:
            return self._params_df.loc[date, self._KRD_cols].values
        except KeyError:
            raise ValueError("The specified date is not a traded date.")

    def trets(self, date=None):
        """
        Get yields for given date and maturity values.

        Parameters
        ----------
        date: datetime, default=None
            Date to return curve for. If None, the default
            date is used.

        Returns
        -------
        ndarray:
            Total returns for treasury portfolio at key
            rate duration maturities for specified date.
        """
        date = pd.to_datetime(date) if date is not None else self._date
        if date is None:
            raise ValueError("A date must be specified.")
        try:
            return self._params_df.loc[date, self._tret_cols].values
        except KeyError:
            raise ValueError("The specified date is not a traded date.")

    def coupons(self, date=None):
        """
        Get yields for given date and maturity values.

        Parameters
        ----------
        date: datetime, default=None
            Date to return curve for. If None, the default
            date is used.

        Returns
        -------
        ndarray:
            Coupons for each treasury in treasury portfolio at
            key rate duration maturities for specified date.
        """
        date = pd.to_datetime(date) if date is not None else self._date
        if date is None:
            raise ValueError("A date must be specified.")
        try:
            return self._params_df.loc[date, self._coupon_cols].values
        except KeyError:
            raise ValueError("The specified date is not a traded date.")

    def plot(
        self, date=None, trange=(0, 30), ax=None, figsize=(8, 6), **kwargs
    ):
        """
        Plot yield curve.

        Parameters
        ----------
        date: datetime
            Date at which to plot yield curve.
        trange: Tuple(float, float), default=(0.1, 30).
            Range (min, max) in years to show.
        ax: matplotlib axis, default=None
            Matplotlib axis to plot figure, if None one is created.
        figsize: list or tuple, default=(6, 6).
            Figure size.
        **kwargs: dict
            Kwargs for matplotlib plotting.
        """
        t = np.linspace(trange[0], trange[1], 200)
        curve = self.yields(t, date=date)
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Plot yield curve.
        plot_kwargs = {"color": "k", "alpha": 0.8, "lw": 2}
        plot_kwargs.update(**kwargs)
        ax.plot(curve, **plot_kwargs)
        tick = mtick.StrMethodFormatter("{x:.2%}")
        ax.yaxis.set_major_formatter(tick)
        ax.set_xlabel("Time (yrs)")
        ax.set_ylabel("Yield")
        ax.set_xlim(trange)


class SyntheticBond:
    def __init__(self, maturity, coupon, price=100):
        self.MaturityYears = maturity
        self.CouponRate = coupon
        self.DirtyPrice = price
        self.CleanPrice = price

        self.coupon_years = np.arange(0.5, maturity + 0.5, 0.5)
        self.cash_flows = coupon / 2 * np.ones(int(maturity * 2))
        self.cash_flows[-1] += 100
        self.OASD = 1  # placeholder

        self._cf = self.cash_flows
        self._t = self.coupon_years

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(coupon={self.CouponRate:.2f} "
            f"maturity={self.MaturityYears:.1f})"
        )

    def _ytm_func(self, y, price):
        """float: Yield to maturity error function for solver."""
        return (self._cf @ np.exp(-y * self._t)) - price

    @property
    def ytm(self):
        """
        float:
            Yield to maturity of the bond at $100 price and
            semiannual coupons.
        """
        x0 = 0.02  # initial guess
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=RuntimeWarning)
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
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=RuntimeWarning)
            return fsolve(self._ytm_func, x0, args=(price))[0]

    @property
    def duration(self):
        i = np.arange(1, int(2 * self.MaturityYears) + 1)
        return (self._cf / (1 + self.ytm / 2) ** i) @ self._t / 100


# %%
def main():
    pass
    # %%
    from lgimapy.data import Database
    from lgimapy.utils import Time

    # %%
    db = Database()
    db.load_market_data()
    ix = db.build_market_index(rating="AAA")
    bond = ix.bonds[0]
    bond.df
    self = bond
    y = 0.02

    # %%
    self = SyntheticBond(30, 2.5)
    self.duration
