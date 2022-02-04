import pandas as pd
from scipy import stats

import lgimapy.vis as vis
from lgimapy.data import Database


db = Database()
start = db.date("market_start")
# start = '3/1/2020'
db.load_market_data(start=start, local=True)
ix = db.build_market_index(in_stats_index=True, rating=("AAA", "AA-"))

qcd = ix.QCD("OAS")
rsd = ix.RSD("OAS")

rsd.drop(pd.to_datetime("1/12/2011"), inplace=True)
rsd.drop(pd.to_datetime("11/30/2006"), inplace=True)
rsd.drop(pd.to_datetime("8/12/2003"), inplace=True)
rsd.drop(pd.to_datetime("2/4/2000"), inplace=True)
rsd.drop(pd.to_datetime("2/1/2000"), inplace=True)
rsd.drop(pd.to_datetime("2/3/2000"), inplace=True)
rsd.drop(pd.to_datetime("2/2/2000"), inplace=True)
rsd.drop(pd.to_datetime("2/11/2000"), inplace=True)
rsd.drop(pd.to_datetime("1/31/2000"), inplace=True)
rsd.drop(pd.to_datetime("1/30/2000"), inplace=True)
rsd.drop(pd.to_datetime("2/7/2000"), inplace=True)
rsd.drop(pd.to_datetime("2/8/2000"), inplace=True)
rsd.drop(pd.to_datetime("2/9/2000"), inplace=True)
rsd.drop(pd.to_datetime("2/10/2000"), inplace=True)
rsd.drop(pd.to_datetime("4/28/2000"), inplace=True)
rsd.drop(pd.to_datetime("1/11/2011"), inplace=True)
#%%
# %matplotlib qt

vis.style()
vis.plot_double_y_axis_timeseries(
    rsd,
    qcd,
    figsize=(14, 10),
    ylabel_left="RSD",
    ylabel_right="QCD",
    xtickfmt="auto",
)
# vis.savefig('AA_dispersion')
vis.show()

# %%
cols = ["Date", 'OAS']

def daily_kurt(df):
    """RSD for single day."""
    return stats.kurtosis(df['OAS'])
def daily_skew(df):
    """RSD for single day."""
    return stats.skew(df['OAS'])

# %%
kurt = ix.df[cols].groupby("Date").apply(daily_kurt)
skew = ix.df[cols].groupby("Date").apply(daily_skew)
kurt = kurt[kurt <20]
skew = skew[skew > 0]
skew = skew[skew < 5]

kurt_diff = kurt.diff()
bad_dates = kurt_diff[kurt_diff > 5]
for date in bad_dates.index:
    try:
        kurt.drop(date, inplace=True)
    except KeyError:
        continue

# %%
vis.plot_double_y_axis_timeseries(
    skew,
    kurt,
    figsize=(14, 10),
    ylabel_left="Skewness",
    ylabel_right="Kurtosis",
    xtickfmt="auto",
)
# vis.savefig('AA_skew_kurtosis')
vis.show()
