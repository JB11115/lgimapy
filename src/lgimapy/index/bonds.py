import datetime as dt
from dateutil.relativedelta import relativedelta
from functools import lru_cache

import numpy as np
from scipy.optimize import fsolve

from lgimapy.bloomberg import get_coupon_dates

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
    def coupon_dates(self):
        """
        List[datetime]:
            Memoized list of timestamps for all coupons.
        """
        # TODO: use scraped dates when apporiate.
        # coupon_dates = get_coupon_dates(self.CUSIP)
        # return coupon_dates[coupon_dates > self.Date]
        return self._theoretical_coupon_dates()

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
            [(cd - self.Date).days / 365 for cd in self.coupon_dates]
        )

    @property
    def next_coupon_date(self):
        """datetime: Date of next coupon."""
        return self.coupon_dates[0]

    @property
    @lru_cache(maxsize=None)
    def cash_flows(self):
        """
        ndarray:
            Memoized cash flows to be acquired on
            :attr:`Bond.coupon_dates`.
        """
        cash_flows = self.CouponRate / 2 + np.zeros(len(self.coupon_dates))
        cash_flows[-1] += 100
        return cash_flows

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


# %%
