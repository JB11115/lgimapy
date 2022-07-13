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
number of occurences (case-insensitive) of each character in a given
alphabetical string, and print results in alphabetical order. A
perfect solution (sans documentation) would look something like this:

    '''
    from collections import Counter

    def print_number_of_char_occurences(s):
        n_occurrences_dict = Counter(s.upper())
        for char, n_occurrences in sorted(n_occurrences_dict.items()):
            print(f"{char}: {n_occurrences}")

    '''

Note that we used the built-in counting algorithm as well as the
built-in sorting function. If you are asked to do something
that you think requires 40+ lines of code, please check to see
if it has already been implemented in the standard library as
that is not the point of these problems. We also do not require
any assumption validations. Note how in the above example we did
not check to ensure that the input was a string consisting only of
alphabetical characters. Assume these checks take place outside
of your functions, and every piece of information we provide
you regarding the inputs to the functions will be true.

The assessment contains several problems for you to solve by
completing the included unfinished functions/classes. You are
encouraged to write additional functions/classes/class methods as you
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

import time
import re

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize


# %%

"""
Problem 1
---------

Write a function to subset a DataFrame of bonds, keeping the rows
which meet the following criteria (note that relevant DataFrame
column names are provided in double quotes).

You will need to create a column "Tenor" (of dtype float) by
subtracting the "CurrentDate" column from the "MaturityDate" column.
A bond's tenor is the number of years remaining from the current
date until the bond matures. For example, a bond with with a
"CurrentDate" of 1/1/2022 and a "MaturityDate" of 7/1/2024 would
have a "Tenor" of 2.5 years.

The critera are provided below (don't worry if you are not
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
        Similarly, a bond with an "OriginalTenor" of 20 years is
        on-the-run while its tenor is between 10 and 20 years, and
        should be excluded once its "Tenor" <= 10 years.
        Similarly for 10 and 7, 7 and 5, and so on.
        For bonds with "OriginalTenor" of 2 years, exclude them
        once their "Tenor" <= 3 months.

Once you have subset the DataFrame appropriately, add an additional
column named "DTS" which is equal to "OASD" * "OAS"  before returning
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

    # Select bonds which meet bellow criterias
    df = df.loc[(df["OAS"] < 40) & (df["OAS"] > -30)]
    df = df.loc[df["CallType"] == "NONCALL"]
    df = df.loc[df["OriginalTenor"].isin([2, 3, 5, 7, 10, 20, 30])]
    df = df.loc[
        (~df["Ticker"].isin(["S", "SP", "SPX", "SPY"]))
        | (df["MaturityDate"].apply(lambda x: x.day) == 15)
    ]
    df = df.loc[
        (
            (df["Sector"] == "TREASURIES")
            | (
                (df["CouponType"] == "ZERO COUPON")
                | ((df["CouponType"] == "FIXED") & (df["CouponRate"] == 0))
            )
        )
    ]

    # Define a new column "Tenor" (of dtype float) in years on Actual/365
    df["Tenor"] = (df["MaturityDate"] - df["CurrentDate"]).astype(
        "timedelta64[D]"
    ) / 365.0

    # Dictionary for checking on-the-run bonds
    on_the_run = {
        30: 20,
        20: 10,
        10: 7,
        7: 5,
        5: 3,
        3: 2,
        2: 0.25,
    }
    df = df.loc[
        (df["Tenor"] <= df["OriginalTenor"])
        & (df["Tenor"] > df["OriginalTenor"].apply(lambda x: on_the_run[x]))
    ]
    df["DTS"] = df["OASD"] * df["OAS"]
    return df


# %%

"""
Problem 2
---------
Given a sorted list of non-continuous dates (for eample, days in which
the bond market traded, which would exclude weekends and holidays),
find the nearest date in the list (inclusive) to a specified reference
date provided to the function. Include the ability to specify only dates
before or after the reference date. If there are two equidistant nearest
dates to the reference date, return the later date. If no dates in
`date_list` meet the specifications, return ``None``. For example:

    >>> date_list = pd.to_datetime(['2000', '2001', '2002'])
    >>> reference_date = pd.to_datetime('12/1/2001')
    >>> nearest_date(reference_date, date_list)
    Timestamp('2002-01-01')
    >>> nearest_date(reference_date, date_list, after=False)
    Timestamp('2001-01-01')

Hint: Imagine this is a function that will be called thousands of times
    per day, so efficiency is vital
"""


def nearest_date(date, date_list, before=True, after=True):
    """
    Parameters
    ----------
    date: datetime
    date_list: List[datetime] or DatetimeIndex
    before: bool, default=True
    after: bool, default=True

    Returns
    -------
    datetime
    """

    # Check the absolute days difference between date and date_list
    abs_days_list = abs((date_list - date).days)
    # If there are two nearest, get the date after the target date
    date_right = max(date_list[abs_days_list == min(abs_days_list)])
    # If two dates are nearest, get the date before the target date
    date_left = min(date_list[abs_days_list == min(abs_days_list)])

    if before & after:
        if min(abs_days_list) == 0:
            return date
        else:
            return date_right
    elif before & (not after):
        if date_left < date:
            return date_left
        else:
            return None
    elif after & (not before):
        if date_right > date:
            return date_right
        else:
            return None
    else:
        if min(abs_days_list) == 0:
            return date
        else:
            return None


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


def count_number_of_increasing_IDs(fid):
    """
    Parameters
    ----------
    fid: str

    Returns
    -------
    int
    """

    count = 0
    with open(fid, "r") as f:
        while True:
            line = f.readline()
            # Regular expression to find all numeric digits
            temp = re.findall("\d", line)
            if len(temp) > 0:
                temp = [int(i) for i in temp]
                # Check if all numeric digits are strictly increasing
                res = all(i < j for i, j in zip(temp, temp[1:]))
                if res:
                    count += 1
            if not line:
                break
    return count


# %%

"""
Problem 4
---------
Complete the following class for a simple bond. The date to use when
computing yield is provided for you as `Bond._current_date`.
Please complete the `Bond.ytm` attribute which returns the continuously
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
    def ytm(self):
        """
        Returns
        -------
        float
        """
        # Calculate t based on Actual/365
        t = ((self.cashflows.index - self._current_date).days) / 365.0
        # Set initial YTM guess for Newton method
        guess = 0.02
        ytm_func = lambda y: sum(self.cashflows / np.exp(y * t)) - self.price
        return optimize.newton(ytm_func, guess)


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

    beta = 1.0
    alpha = 0.0
    r_square = 0.0

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def fit(self):
        time.sleep(1)
        A = np.vstack([self.x, np.ones(len(self.x))]).T
        results = np.linalg.lstsq(A, self.y, rcond=None)
        OLS.beta, OLS.alpha = results[0]
        resid_square = results[1]
        OLS.r_square = (1 - resid_square / (self.y.size * self.y.var()))[0]

    @property
    def resid(self):
        """
        Returns
        -------
        [1 x n] np.array
        """
        return self.y - (OLS.beta * self.x + OLS.alpha)

    def predict(self, x_pred):
        """
        Parameters
        ----------
        x_pred: float or [1 x n] np.array

        Returns
        -------
        float or [1 x n] np.array
        """
        return OLS.beta * x_pred + OLS.alpha

    def plot(self, x_pred=None):
        """
        Parameters
        ----------
        x_pred: float, optional
        """
        plt.figure(figsize=(10, 8), dpi=300)
        plt.plot(self.x, self.y, "o", label="Raw data")
        plt.plot(
            self.x,
            OLS.beta * self.x + OLS.alpha,
            "r",
            label="Fitted line by OLS",
        )
        text1 = f"$Y = {OLS.beta:.2f}$X + {OLS.alpha:.2f}"
        text2 = f"R^2 = {OLS.r_square:.2%}"
        text = text1 + "\n" + text2
        plt.plot([], label=text)
        plt.legend(loc="best", bbox_to_anchor=(1, 0.5), fontsize=10)
        plt.title("OLS Plot")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid()
        plt.savefig("OLS_YH_plot.png")
        plt.show()
        plt.close(plt.gcf())
