"""
Python programming assessment for LGIM America Associate Strategist.
--------------------------------------------------------------------

The purpose of this assessment is twofold: we of course aim to measure
your programming competence, but we also hope to give you a flavor
for the type of problems you will encounter working at LGIM America.

This assessment is likely very different than those you've encountered
in the past. Our goal is to give you an assessment which closely
resembles the environment you would have here at LGIM America. Therefore
you are encouraged to use the IDE/editor of your choice, use any modules
from the python standard library, and take as much time as desired (though
the test is written to be completed in 2-4 hours). You are however
limited to the following third party modules: `numpy`, `matplotlib`,
`pandas`, and `scipy` which we use frequently. You are encouraged
to consult any documentation (or even Stack Overflow) if you are
unsure how to do something, just as you would on the job (but
please don't copy and paste!). Finally, when we say you are allowed
to use the standard library, we really mean use the standard library.

For example, suppose we asked you to write an algorithm to count the
number of occurences (case-insensitive) of each characther in a given
alphabetical string, and print results in alphabetical order. A
perfect solution (sans documentation) would look something like this:

    '''
    from collections import Counter

    def print_number_of_char_occurences(s):
        n_occurrences_dict = Counter(s.upper())
        for char, n_occurrences in sorted(n_occurrences_dict.items()):
            print(f"{char}: {n_occurrences}")

    '''

Note that we used the built in counting algorithm as well as the
built in sorting function. If you are asked to do something
that you think requires 40+ lines of code, please check to see
if it has already been implemented in the standard library as
that is not the point of these problems. We do not require
any assumption validations. Also note how in the above example we did
not check to ensure that the input was a string consisting only of
alphabetical characters. Assume these checks take place outside
of your functions, and every piece of information we provide
you regarding the inputs to the functions will be true.

The assessment contains several problems for you to solve by
completing the included unfinished functions/classes. You are
free to write additional functions/classes/class methods as you
see fit in your solutions, but please do not modify the arguments
or class/function names that are already provided for you.

Your solutions will be judged on readability, accuracy, efficiency,
simplicity, and conciseness. Please write your solutions as if they
were to be included in a code repository shared and maintained by
multiple team members (i.e., include complete documentation and use
descriptive variable names). Note that we strictly adhere to PEP 8.
We do not use use `mypy` or any static type checker, and do not
expect you to do so in your solutions either.
"""

import re
import time
from bisect import bisect_left, bisect_right
from functools import lru_cache

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize

# %%

"""
Problem 1
---------

Write a function to subset a DataFrame of bonds, keeping the rows
which meet the following criteria (Note that relevant DataFrame
column names are provided in double quotes).

You will need to create a column "Tenor" (of dtype float) by
subtracting the "CurrentDate" column from the "MaturityDate" column.
A bond's tenor is the number of years remaining from the current
date until the bond matures.

The critera are are provided below (don't worry if you are not
familiar with the bond lingo, just follow the given logic).
Each returned row, should be a bond that:
    * OAS isnt't too high: "OAS" is less than 40
    * OAS isn't too low: "OAS" is greater than -30
    * Either:
        Is not a treasury strip:
        The "Ticker" is not one of  ('S', 'SP', 'SPX', or 'SPY')
      OR:
        It matures on the 15th of the month:
        "MaturityDate" has day value equal to 15
    * Either:
        is a treasury: "Sector" is 'TREASURIES'
      OR:
        has zero coupon,
            Either:
                "CouponType" is 'ZERO COUPON'
            OR:
                ("CouponType" is 'FIXED') AND ("CouponRate" is equal to 0)
    * is not callable: "CallType" is 'NONCALL'
    * has a normal tenor:
        "OriginalTenor" is equal to one of (2, 3, 5, 7, 10, 20, or 30)
    * is on-the-run: This means that the bond's tenor is between
        its original tenor and the descending tenor from the
        above normal tenor list.

        For example, a bond with an "OriginalTenor" of 30 years is
        on-the-run while its tenor is between 20 and 30 years, and
        should be excluded once its "Tenor" <= 20 years.
        Similarly, a bond with an "OriginalTenor" of 20 years
        on-the-run while its tenor is between 10 and 20 years, and
        should be excluded once its "Tenor" <= 10 years.
        Similarly for 10 and 7, 7 and 5, and so on.
        For bonds with "OriginalTenor" of 2 years, exclude them
        once their "Tenor" <= 3 months.

Once you have subset the DataFrame appropriately, add an additional
column named "DTS" which is equal to "OASD" * "OAS"  before returing
the cleaned DataFrame.
"""


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


# %%

"""
Problem 2
---------
Given a sorted list of non-continuous dates (for eample, days in which
the bond market traded), find the nearest date (inclusive) in the
list to a specified reference date provided to the function. Include the
ability to specify only dates before or after the reference date. If
there are two equidistant nearest dates to the reference date, return
the later date. If no dates in `date_list` meet the specifications,
return ``None``. For example:

    >>> date_list = pd.to_datetime(['2000', '2001', '2002'])
    >>> reference_date = pd.to_datetime('12/1/2001')
    >>> nearest_date(reference_date, date_list)
    Timestamp('2002-01-01')
    >>> nearest_date(reference_date, date_list, after=False)
    Timestamp('2001-01-01')

Hint: Imagine this is a function that will be called thousands of times
    per day, so efficiency is vital
"""


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


# %%

"""
Problem 3
---------
Given the filename for a .txt file containing unique identifiers
consisting alphanumeric characters, count how many of the of the
unique IDs have individual numeric digits which are strictly increasing.
For example, a text file containing the following IDs (those satisfying
the above critera are identified by arrows):

    a1b2c3 <--
    a3b2c1
    a1b1c1
    k2sd9h <--
    9ajk10
    123456 <--
    asdk8d <--
    qwerty
    001234
    123450

Should return 4.
"""


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


# %%

"""
Problem 4
---------
Complete the following class for a simple bond. The date to use when
computing yield is provided for you as `Bond._current_date`.
Please complete the Bond.ytm attribute which calcualtes the continuously
compounding yield of the bond. If you are not familiar with bond math,
refer to equation 2.8 (slide 12) from:
http://fixedincomerisk.com/Web/files/book1/slides_irr_ch2.pdf

Some notes on equation 2.8:
    * `P`: the price of the bond, provided with `price` input argument
    * `y`: the yield to maturity (`Bond.ytm`) which you are solving for
    * `t_i`: time to maturity for the i'th cashflow (in years)
        Use the time difference (in years) between the dates provided
        in the `cashflows` input argument and `Bond._current_date`
        for these values
    * `C_i`: payment for the i'th cashflow, CF_i, provided in the
        `cashflows` input argument
    * `F`: Face value of the bond. You can ignore this as the input
        `cashflows` pd.Series combines the final coupon with face
        value, (i.e., CF_N = C_N + F).

By using cashflows instead of coupons and face value,
equation 2.8 simplifies to:

    P = \summation_{i=1}^{N} { CF_i * e^(y * t_i) }


Hint 1: Use the `scipy.optimize` library
Hint 2: The calculated yields should be around 2%, or 0.02

Bonus Points: Vectorize your solution (No for loops!)
"""


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


# %%

"""
Problem 5
---------

Complete the following class for performing a simple linear regression.
We don't want you to waste time implementing OLS, so please use the
`np.linalg` library or another `numpy` function to solve for alpha and
beta in the regression.

The class should run properly even if `OLS.fit()` is not called
directly by the user. For example:

    >>> ols = OLS(x, y)
    >>> ols.plot()

should run without raising an error and produce the expected plot.
However, if the user does directly call `OLS.fit()`, the results
should be stored such that the model is not being run twice. For example:

    >>> ols = OLS(x, y)
    >>> ols.fit()
    >>> ols.plot()

should not invoke `OLS.fit()` twice. This example is a simple linear
regression, but pretend that it is a complicated model which takes a
long time to fit (which we simulate here with an arbitrary 1 second delay
for fitting but could actually be multiple hours). Leave this delay
in your code.

Do not run `OLS.fit()` during initialization of the class, rather it
should only be invoked after the user calls one of the methods in the
class.

The `OLS.resid` attribute should return the residuals of the regression:

    >>> ols = OLS(x, y)
    >>> ols.resid
    np.array([...])

The `OLS.predict()` method should return a prediction for the y value(s)
associated with the provided `x_pred`.

For the `OLS.plot()` method, we would like you to create an aesthetically
pleasing (imagine it would be shown to Senior Portfolio Managers or
External Clients) plot which may or may not include a single predicted value.
This method should save the figure using "OLS_plot.png" as the figure's
filename.

The plot should at a minimum include:
* The raw data used for fitting
* A best fit line
* The R^2 value of the regression
* Gridlines

"""


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
