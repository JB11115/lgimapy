import re
import time
from bisect import bisect_left, bisect_right
from functools import lru_cache

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize

# %%

# Problem 1
def get_clean_treasuries(df):
    """

    Parameters
    ----------
    df: pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    # Compute years to maturity.
    df["Tenor"] = (df["MaturityDate"] - df["CurrentDate"]).astype(
        "timedelta64[D]"
    ) / 365

    # Make maturity map for finding on-the-run bonds.
    otr_maturity_map = {
        30: 20,
        20: 30,
        10: 7,
        8: 5,
        7: 5,
        5: 3,
        3: 2,
        2: 3 / 12,
    }
    next_og_tenor = df["OriginalTenor"].map(otr_maturity_map)
    is_on_the_run = (df["Tenor"] - next_og_tenor) > 0

    # For each bond, determine if it meets all the rules
    is_a_treasury = df["Sector"] == "TREASURIES"
    has_zero_coupon = (df["CouponType"] == "ZERO COUPON") | (
        (df["CouponType"] == "FIXED") & (df["CouponRate"] == 0)
    )
    is_not_treasury_strip = ~df["Ticker"].isin({"S", "SP", "SPX", "SPY"})
    matures_on_the_15th = df["MaturityDate"].dt.day == 15
    is_not_callable = df["CallType"] == "NONCALL"
    oas_isnt_too_high = df["OAS"] < 40
    oas_isnt_too_low = df["OAS"] > -30
    has_normal_tenor = df["OriginalTenor"].isin(otr_maturity_map.keys())

    # Apply all rules.
    cleaned_df = df[
        oas_isnt_too_high
        & oas_isnt_too_low
        & (is_a_treasury | has_zero_coupon)
        & (is_not_treasury_strip | matures_on_the_15th)
        & is_not_callable
        & has_normal_tenor
        & is_on_the_run
    ].copy()

    # Add DTS column.
    cleaned_df["DTS"] = cleaned_df["OASD"] * cleaned_df["OAS"]
    return cleaned_df


# Problem 2
def nearest_date(reference_date, date_list, before=True, after=True):
    """
    Parameters
    ----------
    reference_date: datetime
    date_list: List[datetime] or DatetimeIndex
    before: bool, default=True
    after: bool, default=True

    Returns
    -------
    datetime
    """
    if before and after:
        closest_dates = [
            nearest_date(reference_date, date_list, before=False),
            nearest_date(reference_date, date_list, after=False),
        ]
        closest_dates = [cd for cd in closest_dates if cd is not None]
        return min(closest_dates, key=lambda x: abs(x - reference_date))
    elif before:
        date = date_list[bisect_right(date_list, reference_date) - 1]
        return date if date <= reference_date else None
    elif after:
        try:
            date = date_list[bisect_left(date_list, reference_date)]
        except IndexError:
            return None
        else:
            return date
    else:
        return


def has_strictly_increasing_digits(num):
    if not num:
        # There are no digits in the number.
        return False

    last_digit = -1
    for num_char in num:
        digit = int(num_char)
        if digit <= last_digit:
            # Current digit is not strictly increasing from prior digit.
            return False
        else:
            # Update the last digit for comparison.
            last_digit = digit

    return True


# Problem 3
def count_number_of_increasing_IDs(fid):
    """
    Parameters
    ----------
    fid: str

    Returns
    -------
    int
    """
    with open(fid, "r") as f:
        IDs = f.read().splitlines()

    pattern = re.compile("\d+")
    increasing_IDs = []
    for ID in IDs:
        digits_only = "".join(pattern.findall(ID))
        if has_strictly_increasing_digits(digits_only):
            increasing_IDs.append(ID)

    return len(increasing_IDs)


# Problem 4
class Bond:
    """
    Parameters
    ----------
    price: float
    cashflows: pd.Series[datetime: float]
    """

    def __init__(self, price, cashflows):
        self._current_date = pd.to_datetime("1/1/2022")
        self.price = price
        self.cashflows = cashflows

    @property
    @lru_cache(maxsize=None)
    def _t(self):
        """
        ndarray:
            Memoized time in years from :attr:`Bond._current_date`
            for all coupons.
        """
        return np.array(
            [
                (date - self._current_date).days / 365
                for date in self.cashflows.index
            ]
        )

    def _ytm_func(self, y):
        """float: Yield to maturity error function for solver."""
        return (self.cashflows @ np.exp(-y * self._t)) - self.price

    @property
    def ytm(self):
        """
        Returns
        -------
        float
        """
        x0 = 0.02  # initial guess
        return optimize.fsolve(self._ytm_func, x0)[0]


# Problem 5
class OLS:
    """
    Parameters
    ----------
    x: List[float] or [1 x n] np.array
    y: List[float] or [1 x n] np.array
    """

    def __init__(self, x, y):
        self.x = np.asarray(x)
        self.y = np.asarray(y)

    @lru_cache(maxsize=None)
    def fit(self):
        time.sleep(1)
        return np.polyfit(self.x, self.y, 1)

    @property
    def params(self):
        return self.fit()

    @property
    @lru_cache(maxsize=None)
    def resid(self):
        """
        Returns
        -------
        [1 x n] np.array
        """
        return self.y - self.predict(self.x)

    def predict(self, x_pred):
        """
        Parameters
        ----------
        x_pred: float or [1 x n] np.array

        Returns
        -------
        float or [1 x n] np.array
        """
        beta, alpha = self.params
        return beta * x_pred + alpha

    @property
    @lru_cache(maxsize=None)
    def rsquared(self):
        ss_res = np.sum(self.resid ** 2)
        ss_tot = np.sum((self.y - np.mean(self.y)) ** 2)
        return 1 - ss_res / ss_tot

    def plot(self):
        plt.style.use("fivethirtyeight")
        fig, ax = plt.subplots()

        # Plot raw data.
        ax.plot(
            self.x, self.y, "o", color="navy", alpha=0.6, label="_nolegend_"
        )

        # Plot best fit line.
        x_fit = np.array([min(self.x), max(self.x)])
        y_fit = self.predict(x_fit)
        ax.plot(x_fit, y_fit, color="firebrick", lw=2)

        # Make title with equation and fit metric.
        alpha, beta = self.params
        title = f"y = {beta:.2f}x + {alpha:.2f}\n$R^2 = {self.rsquared:.2f}$"
        ax.set_title(title, fontweight="bold", fontsize=16)

        # Save figure.
        plt.savefig("OLS_plot.png")
        plt.close(plt.gcf())
