import pandas as pd
import numpy as np

import lgimapy.vis as vis
from lgimapy.data import Database
from lgimapy.latex import Document

# %%
vis.style()
db = Database()
doc = Document(
    "HY_ex_energy_dispersion", path="HY/202004_ex_energy", fig_dir=True
)


db.load_market_data(start="1/1/2020", local=True)
total_ix = db.build_market_index(in_hy_stats_index=True)

energy_sectors = [
    "INDEPENDENT",
    "REFINING",
    "OIL_FIELD_SERVICES",
    "INTEGRATED",
    "MIDSTREAM",
]
ix_ex_energy = total_ix.subset(sector=energy_sectors, special_rules="~Sector")

# %%
rating_buckets = {
    "BB": ("BB+", "BB-"),
    "B": ("B+", "B-"),
    # "CCC": ("CCC+", "CCC-"),
}

oas_list = []
for ix, name in zip([total_ix, ix_ex_energy], ["", " ex Energy"]):
    oas_list.append(ix.market_value_weight("OAS").rename(f"HY{name}"))
    for rating, kwargs in rating_buckets.items():
        ix_sub = ix.subset(rating=kwargs)
        oas_list.append(
            ix_sub.market_value_weight("OAS").rename(f"{rating}{name}")
        )


oas_df = pd.concat(oas_list, axis=1, sort=True)
# oas_df.to_csv("HY_spreads.csv")


# %%
colors = ["k", "navy", "darkorchid"] * 3
linestyles = ["-"] * 3 + ["--"] * 3
vis.plot_multiple_timeseries(
    oas_df,
    figsize=(10, 6),
    c_list=colors,
    ls_list=linestyles,
    xtickfmt="auto",
    ylabel="OAS (bp)",
    lw=3,
    alpha=0.8,
)
doc.add_figure("HY_spreads", savefig=True)

# %%

bb_ex_energy = ix_ex_energy.subset(rating=rating_buckets["BB"])
b_ex_energy = ix_ex_energy.subset(rating=rating_buckets["B"])


def spread_disp(ix, col):
    """Find spread dispersion history."""
    oas = ix.market_value_weight(col)
    disp = np.zeros(len(oas.index))
    for i, date in enumerate(oas.index):
        disp[i] = np.std(ix.day(date)[col])
    return pd.Series(disp / oas, index=ix.dates)


def robust_spread_disp(ix, col):
    """Find spread dispersion history."""
    disp = np.zeros(len(ix.dates))
    for i, date in enumerate(ix.dates):
        vals = ix.day(date)[col]
        Q1 = np.percentile(vals, 25)
        Q3 = np.percentile(vals, 75)
        disp[i] = (Q3 - Q1) / (Q3 + Q1)
    return pd.Series(disp, index=ix.dates)


spread_disp_df = pd.concat(
    [
        spread_disp(bb_ex_energy, "OAS").rename("BB ex Energy"),
        spread_disp(b_ex_energy, "OAS").rename("B ex Energy"),
    ],
    axis=1,
    sort=True,
)
robust_spread_disp_df = pd.concat(
    [
        robust_spread_disp(bb_ex_energy, "OAS").rename("BB ex Energy"),
        robust_spread_disp(b_ex_energy, "OAS").rename("B ex Energy"),
    ],
    axis=1,
    sort=True,
)
fig, axes = vis.subplots(2, 1, figsize=(8, 8), sharex=True)

vis.plot_multiple_timeseries(
    spread_disp_df,
    c_list=["navy", "darkorchid"],
    ls="-",
    lw=3,
    ylabel="RSD",
    ax=axes[0],
)
vis.plot_multiple_timeseries(
    robust_spread_disp_df,
    c_list=["navy", "darkorchid"],
    ls="--",
    lw=3,
    ylabel="Robust Spread DispersionQCD",
    ax=axes[1],
    xtickfmt="auto",
)
doc.add_figure("spread_disperseion_timeseries", savefig=True)
# %%


price_disp_df = pd.concat(
    [
        spread_disp(bb_ex_energy, "DirtyPrice").rename("BB ex Energy"),
        spread_disp(b_ex_energy, "DirtyPrice").rename("B ex Energy"),
    ],
    axis=1,
    sort=True,
)
robust_price_disp_df = pd.concat(
    [
        robust_spread_disp(bb_ex_energy, "DirtyPrice").rename("BB ex Energy"),
        robust_spread_disp(b_ex_energy, "DirtyPrice").rename("B ex Energy"),
    ],
    axis=1,
    sort=True,
)
fig, axes = vis.subplots(2, 1, figsize=(8, 8), sharex=True)

vis.plot_multiple_timeseries(
    price_disp_df,
    c_list=["navy", "darkorchid"],
    ls="-",
    lw=3,
    ylabel="RSD",
    ax=axes[0],
)
vis.plot_multiple_timeseries(
    robust_price_disp_df,
    c_list=["navy", "darkorchid"],
    ls="--",
    lw=3,
    ylabel="QCD",
    ax=axes[1],
    xtickfmt="auto",
)
doc.add_figure("price_disperseion_timeseries", savefig=True)

# %%
bb_ex_energy = ix_ex_energy.subset(date="4/7/2020", rating=rating_buckets["BB"])
b_ex_energy = ix_ex_energy.subset(date="4/7/2020", rating=rating_buckets["B"])

fig, ax = vis.subplots(figsize=(10, 6))

bins = 80
vis.plot_hist(
    b_ex_energy.df["OAS"],
    weights=b_ex_energy.df["MarketValue"],
    bins=bins,
    color="darkorchid",
    ax=ax,
    label="B ex Energy",
)
vis.plot_hist(
    bb_ex_energy.df["OAS"],
    weights=bb_ex_energy.df["MarketValue"],
    bins=bins,
    color="navy",
    ax=ax,
    label="BB ex Energy",
)
ax.legend()
ax.set_title("Histogram of Spreads (weighted by market value)")
ax.set_xlim((None, 2500))
doc.add_figure("spread_disp_hist", savefig=True)

# %%
fig, ax = vis.subplots(figsize=(10, 6))

bins = 30
vis.plot_hist(
    bb_ex_energy.df["DirtyPrice"],
    weights=bb_ex_energy.df["MarketValue"],
    bins=bins,
    color="navy",
    ax=ax,
    label="BB ex Energy",
)
vis.plot_hist(
    b_ex_energy.df["DirtyPrice"],
    weights=b_ex_energy.df["MarketValue"],
    bins=bins,
    color="darkorchid",
    ax=ax,
    label="B ex Energy",
)

ax.legend()
ax.set_title("Histogram of Prices (weighted by market value)")
doc.add_figure("price_disp_hist", savefig=True)
# doc.save(save_tex=True)
