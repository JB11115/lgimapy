import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from lgimapy.models import TreasuryCurve, svensson
from lgimapy.utils import root

pd.plotting.register_matplotlib_converters()
plt.style.use("fivethirtyeight")
# %matplotlib qt

# %%
# Load curves.
tc = TreasuryCurve()

baml_curves = (
    pd.read_csv(
        root("data/treasury_curve_validation.csv"),
        index_col=0,
        parse_dates=True,
        infer_datetime_format=True,
    ).sort_index()
    / 100
)


# Subset BAML curves to where we have data
baml_curves = baml_curves[baml_curves.index.isin(tc.trade_dates())]
baml_curves.columns = [float(c.strip("Y")) for c in baml_curves.columns]

# Make matching LGIMA curves DataFrame.
a_lgima = np.zeros(baml_curves.shape)
for i, date in enumerate(baml_curves.index):
    a_lgima[i, :] = tc.yields(baml_curves.columns, date).values
lgima_curves = pd.DataFrame(
    a_lgima, index=baml_curves.index, columns=baml_curves.columns
)

# Compute difference in curves.
diff = lgima_curves - baml_curves
mae = 1e4 * np.mean(np.abs(diff), axis=1)


print(f"{len(list(baml_curves.index))} Curves to compare")
last_date = baml_curves.index[-1]
# %%
# Plot MAE between two curves over time.
fig, ax = plt.subplots(1, 1)
ax.plot(mae, "--o", ms=2, lw=0.4, c="steelblue", alpha=0.8)
ax.set_xlabel("Date")
ax.set_ylabel("MAE (bp)")
fig.autofmt_xdate()
plt.show()

# %%
# Plot MAE of each maturity.
fig, ax = plt.subplots(1, 1, figsize=[14, 6])
sns.boxplot(
    data=np.abs(1e4 * diff), palette="Spectral", linewidth=1.5, fliersize=1
)
ax.set_ylim((None, 17))
ax.set_xlabel("Maturity")
ax.set_xticklabels(np.arange(1, 31))
ax.set_ylabel("MAE (bp)")
plt.show()

# %%
# Plot MAE of each maturity.
fig, ax = plt.subplots(1, 1, figsize=[14, 6])
ax.axhline(0, lw=1.8, color="k", alpha=0.4)
sns.boxplot(data=1e4 * diff, palette="Spectral", linewidth=1.5, fliersize=1)
ax.set_ylim((-25, 10))
ax.set_xlabel("Maturity")
ax.set_xticklabels(np.arange(1, 31))
ax.set_ylabel("Absolute Difference\nLGIMA - BAML (bp)")
plt.show()


# %%
def plot_curves(date, ax=None, figsize=(8, 6)):
    date = pd.to_datetime(date)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot BAML curve.
    x = np.arange(1, 31)
    y = baml_curves.loc[date, :]
    ax.plot(x, y, "o", c="firebrick", ms=4, alpha=0.6, label="BAML")
    # Plot lgima curve.
    tc.plot(date=date, ax=ax, label="LGIMA", lw=2, alpha=0.6, trange=(0, 100))
    ax.set_title(f'{date.strftime("%m/%d/%Y")}:   MAE={mae[date]:.1f} bp')
    ax.set_xlim((0, 30))
    ax.legend()


# %%
plot_curves("11/28/2008")
plt.show()

# %%
plot_curves("12/30/2009")
plt.show()

# %%
plot_curves(last_date)
plt.show()

# %%
# Plot minimum and maximum yield for LGIMA curves.
all_lgima_curves = TreasuryCurve()._curves_df

all_lgima_curves["min"] = np.min(all_lgima_curves, axis=1)
all_lgima_curves["max"] = np.max(all_lgima_curves, axis=1)

fed_funds_df = (
    pd.read_csv(
        root("data/fed_funds.csv"),
        index_col=0,
        parse_dates=True,
        infer_datetime_format=True,
    )
    / 100
)
fed_funds_df = fed_funds_df[
    (fed_funds_df.index <= all_lgima_curves.index[-1])
    & (fed_funds_df.index >= all_lgima_curves.index[0])
]


fig, ax = plt.subplots(1, 1, figsize=[12, 8])
kwargs = {"alpha": 0.7, "lw": 2}
ax.plot(fed_funds_df["PX_MID"], "--", c="k", alpha=0.7, lw=1, label="Fed Funds")
ax.plot(all_lgima_curves["max"], c="firebrick", label="Max Yield", **kwargs)
ax.plot(all_lgima_curves["min"], c="steelblue", label="Min Yield", **kwargs)
tick = mtick.StrMethodFormatter("{x:.2%}")
ax.yaxis.set_major_formatter(tick)
ax.set_xlabel("Date")
ax.set_ylabel("Yield")
ax.legend()
fig.autofmt_xdate()
plt.show()
