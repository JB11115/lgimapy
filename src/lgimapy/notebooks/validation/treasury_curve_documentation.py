import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

from lgimapy.data import Database, TreasuryCurve
from lgimapy.models import svensson, TreasuryCurveBuilder
from lgimapy.utils import root, savefig, smooth_weight, mkdir

pd.plotting.register_matplotlib_converters()
plt.style.use("fivethirtyeight")
# %matplotlib qt
# %%
fdir = root("latex/doc/treasury_curve_modeling")
mkdir(fdir)
tc = TreasuryCurve()

# %%
dates = [
    "2/1/2019",
    "7/4/2019",
    "4/21/2005",
    "2/14/2005",
    "1/5/2005",
    "5/16/2005",
    "11/14/2018",
]

# %%
# Plot smoothing curves
x = np.linspace(0, 1, 100)
sns.set_palette("cubehelix")
fig, ax = plt.subplots(1, 1, figsize=[8, 6])

for beta in [1, 1.5, 2, 3, 5]:
    ax.plot(x, smooth_weight(x, beta), label=f"$\\beta$ = {beta}", lw=3)
ax.set_xlabel("X")
ax.set_ylabel("Weight")
ax.legend()
savefig(fdir / "smoothing_fnc")
plt.show()

# %%
# Plot raw curves to be combined.
db = Database()
# db.load_market_data("2/1/2019")
db.load_market_data("9/1/2008")
tcb = TreasuryCurveBuilder(
    db.build_market_index(drop_treasuries=False, sector="TREASURIES")
)
# tcb.fit(verbose=1, threshold=10)
tcb.fit(verbose=1, threshold=12)
# tcb.save()
# %%
fig, axes = plt.subplots(2, 1, figsize=[14, 8], sharex=True)
mat_ranges = [(0, 3), (0, 10), (0, 15), (0, 30)]
t_ranges = [(0, 3), (2, 6), (5, 10), (8, 30)]
colors = "limegreen magenta darkorange navy".split()
for trange, mats, c in zip(t_ranges, mat_ranges, colors):
    mat_lbl = "{}-{}".format(*mats)
    params = tcb._partial_curve_params[mat_lbl]
    t = np.linspace(*trange, 100)
    y = svensson(t, params)
    axes[0].plot(t, y, c=c, alpha=0.6, lw=2, label=mat_lbl.replace("-", " - "))


axes[0].legend()
tcb.plot(ax=axes[1])
axes[1].set_xlim((-0.5, 31))
# savefig(fdir / "combining_curves")
plt.show()

# %%
# Plot close up of complex front end.
fig, ax = plt.subplots(1, 1, figsize=[8, 5])
tcb.plot(ax=ax)
ax.set_ylim((0.0235, 0.026))
ax.set_xlim((0, 8))
# savefig(fdir / "complex_front_end")
plt.show()

# %%
# Plot curve progression over a month.
month = 3
year = 2019

tc = TreasuryCurve()
next_month = 1 if month == 12 else month + 1
next_year = year + 1 if month == 12 else year
start = pd.to_datetime(f"{month}/1/{year}")
end = pd.to_datetime(f"{next_month}/1/{next_year}")
dates = tc.trade_dates()
dates = [d for d in dates if start <= d < end]

t = np.linspace(0, 30, 400)
sns.set_palette("viridis_r", len(dates))
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
for date in dates:
    curve = tc.yields(t, date=date)
    ax.plot(t, curve.values, lw=1.5, alpha=0.5, label=date.strftime("%m/%d/%Y"))

tick = mtick.StrMethodFormatter("{x:.2%}")
ax.yaxis.set_major_formatter(tick)
ax.set_xlabel("Time (yrs)")
ax.set_ylabel("Yield")
ax.legend(bbox_to_anchor=(1, 1), loc=2)

d = None

day = 30
# d = f"{month}/{day}/{year}"
if d is not None:
    curve = tc.yields(t, date=d)
    ax.plot(t, curve.values, lw=3, c="k")
    fig.suptitle(d)
plt.tight_layout()
savefig(f"monthly_{month}_{year}")
plt.show()

# %%
# Make gif of curves.
import moviepy.editor as mpy

image_dir = root("data/treasury_curves")
image_list = list(image_dir.glob("*png"))
image_list = [str(fid) for fid in image_list]
fps = 20

clips = mpy.ImageSequenceClip(image_list, fps=fps)
clips.write_videofile(str(image_dir / "curves.mp4"), fps=fps, codec="mpeg4")

# %%
# Compare LGIMA curves to BAML curves.
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
fig, ax = plt.subplots(1, 1, figsize=[12, 6])
ax.plot(mae, "o", ms=3, color="steelblue", alpha=0.8)
ax.set_xlabel("Date")
ax.set_ylabel("MAE (bp)")
ax.set_ylim((None, 10))
fig.autofmt_xdate()
# savefig('BAML_timeseries')
plt.show()


# %%
# Plot MAE of each maturity.
fig, ax = plt.subplots(1, 1, figsize=[14, 6])
sns.boxplot(
    data=np.abs(1e4 * diff), palette="Spectral", linewidth=1.5, fliersize=1
)
ax.set_ylim((None, 14))
ax.set_xlabel("Maturity")
ax.set_xticklabels(np.arange(1, 31))
ax.set_ylabel("MAE (bp)")
# savefig('BAML_maturity')
plt.show()

# %%
# Plot absolute difference of each maturity.
fig, ax = plt.subplots(1, 1, figsize=[14, 6])
ax.axhline(0, lw=1.8, color="k", alpha=0.4)
sns.boxplot(data=1e4 * diff, palette="Spectral", linewidth=1.5, fliersize=1)
ax.set_ylim((-20, 10))
ax.set_xlabel("Maturity")
ax.set_xticklabels(np.arange(1, 31))
ax.set_ylabel("Absolute Difference\nLGIMA - BAML (bp)")
# savefig('BAML_abs_maturity')
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
# savefig('BAML_11_28_2008')
plt.show()

# %%
plot_curves("12/30/2009")
# savefig('BAML_12_30_2009')
plt.show()

# %%
plot_curves("2/24/2014")
# savefig('BAML_2_24_2014')
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
savefig("max_min")
plt.show()
