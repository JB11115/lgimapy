import datetime as dt

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from index_functions import IndexBuilder

pd.plotting.register_matplotlib_converters()
plt.style.use("fivethirtyeight")

# %matplotlib qt

# %%


def main():
    fdir = "../fig"
    save = False


def deviation_before_grade_change(fdir, save):
    """
    Plot deviation from rating before a rating change takes place.
    """
