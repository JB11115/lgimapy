import numpy as np
import pandas as pd

from index_functions import IndexBuilder
from treasury_curve import TreasuryCurve

# %%


ixb = IndexBuilder()
ixb.load(dev=True, start="5/1/2019", end="5/1/2019")
ix = ixb.build()

tc = TreasuryCurve()


class Bond:
    """
    Class for bond math manipulation given
    current state of the bond.

    Parameters
    ----------
    series: pd.Series
        Single bond row from `index_builder` DataFrame.


    Attributes
    ----------
    coupon_dates: All coupon dates for given treasury.
    coupon_years: Time to all coupons in years.

    Methods
    -------
    calculate_price(rfr): Calculate price of bond with given risk free rate.
    """

    def __init__(self, s):
        self.s = s
        self.__dict__.update({k: v for k, v in zip(s.index, s.values)})
        self.KRDs = s[
            "KRD06mo KRD02yr KRD05yr KRD10yr KRD20yr KRD30yr".split()
        ].values

    def excess_return(self, treasury_curve):
        trsy_krds = treasury_curve.get_KRDs(self.Date)
        trsy_krds
        weights = self.KRDs / trsy_krds
        cash_weight = 1 - np.sum(weights)
        cash_weight
        dates = treasury_curve.trade_dates
        yesterday = dates[dates.index(self.Date) - 1]
        # yield_change =


self = Bond(ix.df.iloc[0, :])
self.KRDs

treasury_curve = tc
