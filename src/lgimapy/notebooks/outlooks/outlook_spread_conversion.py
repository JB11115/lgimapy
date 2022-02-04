import numpy as np
import pandas as pd
import statsmodels.api as sms

from lgimapy import vis
from lgimapy.data import Database
from lgimapy.utils import root, nearest_date

vis.style()
# %%


db = Database()
start = db.date("6m")
bbg_df = db.load_bbg_data(["US_IG_10+", "US_IG", "US_CORP"], "OAS", start=start)
baml_df = (
    pd.read_csv(
        root("data/outlook_indexes/BAML.csv"),
        index_col=0,
        skiprows=1,
        parse_dates=True,
        infer_datetime_format=True,
    )
    .iloc[:, 0]
    .dropna()
    .rename("BAML")
)
juli_df = (
    pd.read_csv(
        root("data/outlook_indexes/JULI.csv"),
        index_col=0,
        parse_dates=True,
        infer_datetime_format=True,
    )
    .iloc[:, 0]
    .dropna()
    .rename("JULI")
)
iboxx_df = (
    pd.read_csv(
        root("data/outlook_indexes/IBOXX.csv"),
        index_col=0,
        parse_dates=True,
        infer_datetime_format=True,
    )
    .iloc[:, 0]
    .dropna()
    .rename("IBOXX")
)
df = pd.concat((bbg_df, baml_df, juli_df, iboxx_df), axis=1)
# %%

target = 160
x_col = "IBOXX"
y_col = "US_IG_10+"

xy_df = df[[x_col, y_col]].dropna()
xy_df = xy_df[xy_df.index >= start]
x = sms.add_constant(xy_df[x_col])
y = xy_df[y_col]
res = sms.OLS(y, x).fit()


def pred(x):
    x_pred = pd.DataFrame({"const": [1], "x": [x]})
    return res.predict(x_pred).iloc[0]


x_plot = x.iloc[:, 1]
x_min, x_max = np.min(x_plot), np.max(x_plot)
ye_target = pred(target)

fig, ax = vis.subplots()
ax.plot(x_plot, y, "o", color="navy", ms=5, alpha=0.7)
ax.plot(
    (x_min, x_max),
    (pred(x_min), pred(x_max)),
    lw=2,
    color="firebrick",
    alpha=0.8,
)
ax.plot(
    [target],
    [ye_target],
    "o",
    color="darkgreen",
    ms=8,
    label=f"Long Credit Target: {ye_target:.0f} bp",
)
vis.legend(ax)
print(f"Long Credit Target: {ye_target:.0f} bp")
vis.show()
