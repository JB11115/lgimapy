from functools import lru_cache

import numpy as np
import pandas as pd

from lgimapy.data import Database
from lgimapy.utils import root

# %%


class Technicals:
    def __init__(self):
        self.db = Database()

    def kwargs(self, index):
        return None

    @property
    @lru_cache(maxsize=None)
    def supply_forecasts(self):
        fid = root("data/supply_forecasts.csv")
        return pd.read_csv(fid)

    def load_data(self, month, year):
        """
        Load data for given month and year, including last date
        prior to month start.
        """
        month_ends = pd.Series(self.db.date("MONTH_ENDS"))
        current = (month_ends.dt.year == year) & (month_ends.dt.month == month)
        current_month_end_s = month_ends[current]
        current_month_end = current_month_end_s.iloc[0]
        prev_month_end = month_ends[current_month_end_s.index - 1].iloc[0]
        self.db.load_market_data(start=prev_month_end, end=current_month_end)

    def new_issues(self, ix):
        dates = ix.dates
        prev_isins = ix.subset(date=dates[0]).isins
        return ix.subset(
            start=dates[1],
            issue_years=(None, 40 / 365),
            isin=prev_isins,
            special_rules="~ISIN",
        )


self = Technicals()
self.load_data(5, 2021)

# %%
