from collections import defaultdict

import pandas as pd

from lgimapy import vis
from lgimapy.data import Database

vis.style()

# %%

db = Database()
account = "CLECLD"
dates = db.trade_dates(start=db.date("PORTFOLIO_START"))
d = defaultdict(list)
for date in tqdm(dates):
    acnt = db.load_portfolio(account=account, date=date)
    d["DTS"].append(acnt.dts())
    d["tsy_OAD"].append(acnt.tsy_oad())
    d["BM_OAD"].append(acnt.df["BM_OAD"].sum())

df = pd.DataFrame(d, index=dates)
df = df[df.index >= pd.to_datetime("9/17/2018")]
# %%
df["tsy_OAD_pct"] = df["tsy_OAD"] / df["BM_OAD"]
bad_dates = pd.to_datetime(
    ["9/29/2020", "9/4/2018", "9/5/2018", "9/6/2018"]
)
df_clean = df[~df.index.isin(bad_dates)]

fig, ax = vis.subplots(figsize=(10, 6))
vis.plot_double_y_axis_timeseries(
    df_clean["DTS"],
    df_clean["tsy_OAD_pct"],
    ylabel_left="Portfolio DTS\n(As % of Benchmark DTS)",
    ylabel_right="Treasury OAD Contribution to Portfolio",
    ytickfmt_left="{x:.0%}",
    ytickfmt_right="{x:.1%}",
    plot_kws_left={"color": "navy"},
    plot_kws_right={"color": "darkorchid"},
    lw=3,
    ax=ax,
)
vis.show()
df_clean.to_csv(f"{account}_barbell_data.csv")
# vis.savefig("CLECLD_barbell")
# %%


# %%
df_clean = df[df.index != pd.to_datetime("9/29/2020")]
df_plot = df_clean.resample("M").mean().reset_index()


def make_patch_spines_invisible(ax):
"""
Make matplotlib Axes edges invisible.

Parameters
----------
ax: matplotlib Axes, optional
    Axes in which to draw plot, otherwise activate Axes.
"""
ax.set_frame_on(True)
ax.patch.set_visible(False)
for sp in ax.spines.values():
    sp.set_visible(False)


vis.style()
fig, ax_left = vis.subplots(figsize=(10, 6))
ax_right = ax_left.twinx()
ax_left.set_zorder(ax_right.get_zorder() + 1)
# Make spine visible for right axis.
make_patch_spines_invisible(ax_left)
make_patch_spines_invisible(ax_right)
ax_right.grid(False)
ax_right.spines["right"].set_position(("axes", 1.05))
ax_right.spines["right"].set_visible(True)
ax_right.spines["right"].set_linewidth(1.5)
ax_right.spines["right"].set_color("lightgrey")
ax_right.tick_params(right="on", length=5)

c_left = "navy"
c_right = "goldenrod"

ax_right.bar(
df_plot.index, df_plot["tsy_OAD"], width=0.6, alpha=0.7, color="goldenrod"
)

ax_left.set_ylabel("Portfolio DTS\n(As % of Benchmark DTS)", color=c_left)
ax_right.set_ylabel("Treasury OAD Contribution (yrs)", color=c_right)
ax_left.set_xticks(df_plot.index)
ax_right.set_xticks(df_plot.index)
ax_left.set_xticklabels(list("NDJFMAMJJASON"))
ax_left.grid(False, axis="x")
ax_right.grid(False)
ax_left.plot(
df_plot.index, df_plot["DTS"], color=c_left, zorder=100,
)
ax_left.tick_params(axis="y", colors=c_left)
ax_right.tick_params(axis="y", colors=c_right)
ax_right.set_ylim(0, 0.75)
vis.format_yaxis(ax_left, "{x:.0%}")
vis.savefig("CLECLD_barbell")

# %%
df_clean_pct = df_clean.rank(pct=True)
df_clean_pct["combined"] = df_clean_pct.sum(axis=1)

vis.plot_timeseries(
df_clean_pct["combined"].rolling(7).mean(),
ylabel="Summed 1 Year Percentile of\nPortfolio DTS and Treasury OAD Contribution",
)
# vis.savefig('barbell_measure_monthly')
vis.savefig("barbell_measure")

df_clean.to_csv(f"{account}_barbell_data.csv")
df_clean.resample("M").mean().to_csv(f"{account}_barbell_monthly_data.csv")
db.date("PORTFOLIO_START")
