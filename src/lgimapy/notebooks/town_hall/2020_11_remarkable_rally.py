import numpy
import pandas as pd
import seaborn as sns

from lgimapy import vis
from lgimapy.data import Database

vis.style()
# %%

db = Database()
sp = "S&P 500 Total Returns"
lc = "Long Credit Excess Returns"
daily_d = {}
daily_d[sp] = db.load_bbg_data("SP500", "TRET", start="12/20/1989")
daily_d[lc] = db.load_bbg_data("US_IG_10+", "XSRET", start="12/20/1989")

monthly_ret = []
monthly_level = daily_d[sp].resample("M").last()
monthly_ret.append((monthly_level / monthly_level.shift() - 1)[1:-1].rename(sp))
monthly_ret.append(daily_d[lc].resample("M").last()[1:-1].rename(lc) / 100)

df = pd.concat(monthly_ret, axis=1)
# df = df[(df[lc] > -30) & (df[lc] < 50)]
# %%
g = sns.jointplot(data=df, x=sp, y=lc, color="k", alpha=0.4, height=6)
# Plot lines for current values.
g.ax_joint.axhline(0, color="k", lw=0.5, ls="--", label="_nolegend_")
g.ax_joint.axvline(0, color="k", lw=0.5, ls="--", label="_nolegend_")
g.ax_joint.plot(
    df[sp].iloc[-1],
    df[lc].iloc[-1],
    "o",
    ms=8,
    color="firebrick",
    label="Nov 2020",
)

vis.format_xaxis(g.ax_joint, xtickfmt="{x:.0%}")
vis.format_yaxis(g.ax_joint, ytickfmt="{x:.0%}")
vis.savefig("remarkable_rally")

# %%
fig, axes = vis.subplots(1, 2, figsize=(8, 6))
sns.boxplot(data=df[sp], ax=axes[0])
axes[0].set_ylabel(sp)
sns.boxplot(data=df[lc], ax=axes[1])
axes[1].set_ylabel(lc)
vis.format_xaxis(axes[0], xtickfmt="{x:.0%}")
vis.savefig("remarkable_rally")

# %%
fig, ax = vis.subplots(1, 1, figsize=(12, 6))
last = df[sp].iloc[-1]
df_temp = df[sp].sort_values(ascending=False).reset_index(drop=True)
loc = df_temp[df_temp == last]
ax.bar(df_temp.index, df_temp.values, width=0.9, color="grey", alpha=0.5)
ax.bar(loc.index, loc.values, width=1, color="firebrick")
vis.format_yaxis(ax, ytickfmt="{x:.0%}")

ax.annotate(
    "Nov 2020",
    xy=(loc.index[0], loc.values[0]),
    xytext=(60, 20),
    textcoords="offset points",
    ha="center",
    va="bottom",
    bbox=dict(boxstyle="round,pad=0.2", fc="w", alpha=0.3),
    arrowprops=dict(
        arrowstyle="->", connectionstyle="arc3,rad=0.5", color="firebrick"
    ),
)
vis.savefig("sp_rally")
# vis.show()
#
# %%
fig, ax = vis.subplots(1, 1, figsize=(8, 6))
last = df[sp].iloc[-1]
df_temp = df[sp].sort_values(ascending=False).reset_index(drop=True)
loc = df_temp[df_temp == last]
ax.bar(df_temp.index, df_temp.values, width=0.9, color="grey", alpha=0.5)
ax.bar(loc.index, loc.values, width=1, color="firebrick")
vis.format_yaxis(ax, ytickfmt="{x:.0%}")
ax.annotate(
    "S&P 500 Total Returns",
    xy=(0.5, 1.08),
    xycoords="axes fraction",
    fontweight="bold",
    ha="center",
    fontsize=20,
)
ax.annotate(
    "Sorted Monthly Values since 1990",
    xy=(0.5, 1.03),
    xycoords="axes fraction",
    ha="center",
    fontsize=12,
)
ax.annotate(
    "Nov 2020",
    xy=(loc.index[0], loc.values[0]),
    xytext=(60, 20),
    textcoords="offset points",
    ha="center",
    va="bottom",
    bbox=dict(boxstyle="round,pad=0.2", fc="w", alpha=0.3),
    arrowprops=dict(
        arrowstyle="->", connectionstyle="arc3,rad=0.5", color="firebrick"
    ),
)
vis.savefig("sp_rally")


# %%
fig, ax = vis.subplots(1, 1, figsize=(8, 6))
last = df[lc].iloc[-1]
df_temp = df[lc].sort_values(ascending=False).reset_index(drop=True)
loc = df_temp[df_temp == last]
ax.bar(df_temp.index, df_temp.values, width=0.9, color="grey", alpha=0.5)
ax.bar(loc.index, loc.values, width=1, color="firebrick")
vis.format_yaxis(ax, ytickfmt="{x:.0%}")
ax.annotate(
    "Long Credit Excess Returns",
    xy=(0.5, 1.08),
    xycoords="axes fraction",
    fontweight="bold",
    ha="center",
    fontsize=20,
)
ax.annotate(
    "Sorted Monthly Values since 1990",
    xy=(0.5, 1.03),
    xycoords="axes fraction",
    ha="center",
    fontsize=12,
)
ax.annotate(
    "Nov 2020",
    xy=(loc.index[0], loc.values[0]),
    xytext=(60, 20),
    textcoords="offset points",
    ha="center",
    va="bottom",
    bbox=dict(boxstyle="round,pad=0.2", fc="w", alpha=0.3),
    arrowprops=dict(
        arrowstyle="->", connectionstyle="arc3,rad=0.5", color="firebrick"
    ),
)
vis.savefig("lc_rally")
