import os
import random
import string
import shutil
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import optimize

from lgimapy.data import Database
from lgimapy.bloomberg import get_bloomberg_ticker
from lgimapy.utils import mkdir, root



data_dir = root('src/lgimapy/notebooks/interviews/test_data')
mkdir(data_dir)

# %%
# Problem 1
pd.to_datetime(["2000", "2001", "2002"])
db.load_market_data()
ix = db.build_market_index(drop_treasuries=False)
df = ix.df.copy()

df["CurrentDate"] = df["Date"]
df["OriginalTenor"] = df["OriginalMaturity"]
tres = df["Sector"] == "TREASURIES"
df.loc[tres, "Ticker"] = get_bloomberg_ticker(df[tres]["CUSIP"].values)
cols = [
    "ISIN",
    "CurrentDate",
    "MaturityDate",
    "OriginalTenor",
    "Ticker",
    "Issuer",
    "Sector",
    "CouponType",
    "CouponRate",
    "CallType",
    "OAS",
    "OASD",
    "OAD",
]
df_clean = df[cols].reset_index(drop=True).copy()

df_clean.to_csv(data_dir / "Problem_1_data.csv")

# %%
# Problem 3
chars = list(set(string.ascii_lowercase) | set(map(str, range(10))))

ids = []
for i in range(100000):
    id = ""
    for j in range(8):
        id = f"{id}{random.choice(chars)}"
    ids.append(id)

with open(data_dir / "Problem_3_data.txt", "w") as f:
    f.write("\n".join(id for id in ids))


# %%
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

p4_data_dir = data_dir / 'Problem_4_data'
try:
    shutil.rmtree(p4_data_dir)
except FileNotFoundError:
    pass

mkdir(p4_data_dir)
start_date = pd.to_datetime('1/1/2022')
fids = Database.local('cashflows').glob('*')
i = 0
for fid in fids:
    if i == 10:
        break
    df = pd.read_parquet(fid)
    df = df[df.index>start_date]
    if len(df):
        cashflows = df['cash_flows']
        bond = Bond(price=100, cashflows=cashflows)
        if 0.018 <= bond.ytm <= 0.025:
            new_fid = p4_data_dir / f'Cashflow_test_{i}.csv'
            df.to_csv(new_fid)
            i += 1
