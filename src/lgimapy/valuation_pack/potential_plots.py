from collections import defaultdict
from datetime import datetime as dt
from inspect import cleandoc

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from lgimapy import vis
from lgimapy.data import Database
from lgimapy.latex import Document
from lgimapy.utils import root


currencies = ["\N{dollar sign}", "\N{euro sign}", "\N{pound sign}"]

# %%
# %matplotlib qt
vis.style()
db = Database()
# $, euro, and sterling spreads.
fig, (ax1, ax2) = vis.subplots(1, 2, figsize=(14, 6), sharey=True)
securities = ["US_IG", "EU_IG", "GBP_IG"]
df = db.load_bbg_data(securities, "OAS", start=db.date("20y"), nan="ffill")
df = df[df.index != pd.to_datetime("2/15/2010")]
df = df[df.index != pd.to_datetime("2/12/2010")]
df.columns = currencies
colors = ["#04060F", "#0294A5", "#C1403D"]
vis.plot_multiple_timeseries(
    df,
    ylabel="OAS",
    c_list=colors,
    ax=ax1,
    title="Market Credit",
    xtickfmt="auto",
)
for col, color in zip(df.columns, colors):
    ax1.axhline(df[col][-1], color=color, ls="--", lw=1, alpha=0.7)
    ax1.plot(df.index[-1], df[col][-1], "o", color=color, ms=5)
securities = ["US_IG_10+", "EU_IG_10+", "GBP_IG_15+"]
df = db.load_bbg_data(securities, "OAS", start=db.date("20y"), nan="ffill")
df = df[df.index != pd.to_datetime("4/2/2008")]
df.columns = currencies
vis.plot_multiple_timeseries(
    df, ax=ax2, xtickfmt="auto", c_list=colors, title="Long Credit"
)
for col, color in zip(df.columns, colors):
    ax2.axhline(df[col][-1], color=color, ls="--", alpha=0.7, lw=1)
    ax2.plot(df.index[-1], df[col][-1], "o", color=color, ms=5)
# doc.add_figure("global_IG_OAS", savefig=True)
# vis.show()
vis.savefig("global_spreads")


# %%
# $, euro, and sterling yields.
fig, (ax1, ax2) = vis.subplots(1, 2, figsize=(14, 6), sharey=True)
securities = ["US_IG", "EU_IG", "GBP_IG"]
df = db.load_bbg_data(securities, "YTW", start=db.date("3y"), nan="ffill")
df.columns = currencies
yfmt = "{x:.1%}"
vis.plot_multiple_timeseries(
    df / 100,
    ylabel="Yield",
    xlabel="Market Credit",
    ytickfmt=yfmt,
    legend=False,
    ax=ax1,
)
securities = ["US_IG_10+", "EU_IG_10+", "GBP_IG_15+"]
df = db.load_bbg_data(securities, "YTW", start=db.date("3y"), nan="ffill")
df.columns = currencies
vis.plot_multiple_timeseries(
    df / 100, ytickfmt=yfmt, xlabel="Long Credit", ax=ax2
)
fig.suptitle("IG Yields", y=1.02)
plt.tight_layout()
doc.add_figure("global_IG_yields", savefig=True)

# %%
# $, euro, and sterling IG / HY ratio.
securities = ["US_IG", "EU_IG", "US_HY", "EU_HY"]
df = db.load_bbg_data(securities, "OAS", start=db.date("3y"), nan="ffill")
df["US"] = df["US_HY"] / df["US_IG"]
df["EU"] = df["EU_HY"] / df["EU_IG"]
df = df[["US", "EU"]]
df.columns = currencies[:2]
vis.plot_multiple_timeseries(df, title="HY/IG Ratio", figsize=(14, 8))
doc.add_figure("HY_IG_ratio", savefig=True)

# %%
# $, euro, and sterling BBB/A and BB/B ratios.
fig, (ax1, ax2) = vis.subplots(1, 2, figsize=(14, 6), sharey=True)
securities = ["US_BBB", "EU_BBB", "GBP_BBB", "US_A", "EU_A", "GBP_A"]
df = db.load_bbg_data(securities, "OAS", start=db.date("3y"), nan="ffill")
df["US"] = df["US_BBB"] / df["US_A"]
df["EU"] = df["EU_BBB"] / df["EU_A"]
df["GBP"] = df["GBP_BBB"] / df["GBP_A"]
df = df[["US", "EU", "GBP"]]
df.columns = currencies
vis.plot_multiple_timeseries(df, xlabel="BBB/A", ylabel="Ratio", ax=ax1)
securities = ["US_B", "EU_B", "US_BB", "EU_BB"]
df = db.load_bbg_data(securities, "OAS", start=db.date("3y"), nan="ffill")
df["US"] = df["US_B"] / df["US_BB"]
df["EU"] = df["EU_B"] / df["EU_BB"]
df = df[["US", "EU"]]
df.columns = currencies[:2]
vis.plot_multiple_timeseries(df, xlabel="B/BB", ax=ax2)
fig.suptitle("Compression Ratios", y=1.02)
plt.tight_layout()
doc.add_figure("global_compression_ratios", savefig=True)


# %%
# US IG Spreads
s_list = [
    ix_d[key]
    .subset(start=db.date("1y"))
    .market_value_weight("OAS")
    .rename(ix_names[key])
    for key in ["mc", "lc", "10y", "30y"]
]
vis.plot_multiple_timeseries(
    s_list,
    ylabel="OAS",
    xtickfmt="auto",
    title="US IG Spreads",
    figsize=(14, 8),
)
doc.add_figure("US_IG_spreads", savefig=True)


# %%
# US IG Spread Dispersion.
def spread_disp(ix):
    """Find spread dispersion history."""
    oas = ix.market_value_weight("OAS")
    disp = np.zeros(len(ix.dates))
    for i, date in enumerate(ix.dates):
        disp[i] = np.std(ix.day(date)["OAS"])
    return pd.Series(disp / oas, index=ix.dates)


vis.plot_multiple_timeseries(
    [
        spread_disp(ix_d["10y"]).rename("10 Yr"),
        spread_disp(ix_d["30y"]).rename("30 Yr"),
    ],
    ylabel="Spread Dispersion",
    title="US IG Spread Dispersion Normalized by OAS",
    xtickfmt="auto",
)
doc.add_figure("US_IG_spread_dispersion", savefig=True)

# %%
# US IG market value history.
yfmt = "${x:.1f}T"
vis.plot_double_y_axis_timeseries(
    ix_d["mc"].total_value().rename("Market Credit") / 1e6,
    ix_d["lc"].total_value().rename("Long Credit") / 1e6,
    ytickfmt_left=yfmt,
    ytickfmt_right=yfmt,
    ylabel_left="Market Credit",
    ylabel_right="Long Credit",
    title="Index Market Values",
    alpha=0.8,
    figsize=(8, 4),
)
doc.add_figure("US_IG_market_values", savefig=True)

# %%
# US rates vol vs equity vol.
df = db.load_bbg_data(
    ["MOVE", "VIX"], "PRICE", start=db.date("3y"), nan="ffill"
)
vis.plot_double_y_axis_timeseries(
    df["MOVE"],
    df["VIX"],
    ylabel_left="MOVE Index Price",
    ylabel_right="VIX Price",
    right_plot_kwargs={"color": "darkgreen"},
    left_plot_kwargs={"color": "goldenrod"},
    title="Rates Vol vs Equity Vol",
    alpha=0.8,
    lw=2,
    figsize=(14, 8),
)
doc.add_figure("US_rates_vol_vs_equity_vol", savefig=True)

# %%
# US IG vol vs CDX.
df = db.load_bbg_data(
    ["US_IG", "CDX_IG"], "OAS", start=db.date("3y"), nan="ffill"
)
vis.plot_double_y_axis_timeseries(
    df["US_IG"],
    df["CDX_IG"],
    ylabel_left="Market Credit OAS",
    ylabel_right="CDX IG Spread",
    right_plot_kwargs={"color": "darkorchid"},
    title="US IG Cash vs CDX",
    alpha=0.8,
    lw=2,
    figsize=(14, 8),
)
doc.add_figure("US_IG_cash_vs_CDX", savefig=True)

# %%
# CDX IG regressed on S&P 500.
df = pd.concat(
    [
        db.load_bbg_data("SP500", "PRICE", start=db.date("3m")),
        db.load_bbg_data("CDX_IG", "OAS", start=db.date("3m")),
    ],
    axis=1,
    join="inner",
    sort=True,
)
temp_dates = {
    "Current": df.index[-1],
    "1D": df.index[-2],
    "1W": nearest(df.index, db.date("1w")),
    "1M": nearest(df.index, db.date("1m")),
    "3M": df.index[0],
}
# Perform OLS.
y = df["SP500"].values
x = df["CDX_IG"].values
ols = sms.OLS(y, sms.add_constant(x)).fit()
alpha, beta = ols.params
ix = [np.argmin(x), np.argmax(x)]

fig, ax = vis.subplots()
ax.plot(
    df["CDX_IG"],
    df["SP500"],
    "o",
    ms=4,
    markeredgecolor=None,
    c="k",
    alpha=0.4,
    label="_nolegend_",
)
ax.plot(x[ix], x[ix] * beta + alpha, lw=2, c="k", alpha=0.5, label="_nolegend_")
colors = [vis.colors(c) for c in "rygb"] + ["darkorchid"]
for c, (lbl, date) in zip(colors, temp_dates.items()):
    ax.plot(
        df.loc[date, "CDX_IG"],
        df.loc[date, "SP500"],
        "o",
        markeredgecolor="k",
        c=c,
        ms=9,
        label=lbl,
    )
ax.legend()
ax.set_title("CDX IG vs Equities")
ax.set_xlabel("CDX IG Spread (bp)")
ax.set_ylabel("S&P 500 Price")
vis.format_yaxis(ax, "${x:,.0f}")
plt.tight_layout()
doc.add_figure("CDX_vs_equities", savefig=True)


# %%
ix_bb = db.build_market_index(
    rating=("BBB+", "BBB-"), maturity=(25, 32), in_stats_index=True
)
ix_a = ix_d["mc"].subset(rating=("AA+", "A-"), maturity=(5, 10))

df = pd.concat(
    [
        ix_bb.market_value_weight("OAS").rename("BBB"),
        ix_a.market_value_weight("OAS").rename("A"),
    ],
    axis=1,
    sort=True,
    join="inner",
)
df["BBB/A"] = df["BBB"] / df["A"]
vis.plot_timeseries(
    df["BBB/A"],
    xtickfmt="auto",
    title="Long BBB / Intermediate A Ratio",
    ylabel="Ratio",
)
vis.show()
# %matplotlib qt

# %%
# Fed Funds 1 year
vis.style()
db = Database()
df = db.load_bbg_data(
    ["FF_1M", "FF_12M"], "PRICE", nan="drop", start=db.date("1y")
)
diff = 100 * (df["FF_1M"] - df["FF_12M"])
vis.plot_timeseries(
    diff,
    title="Rate Hikes (Cuts) Priced in 12 Months Forward (bp)",
    xtickfmt="auto",
    color="goldenrod",
    figsize=(10, 8),
)
vis.show()
# vis.savefig("fed_funds")
# vis.close()
