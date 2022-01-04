import warnings

import numpy as np
from scipy.optimize import fsolve

# %

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
