from collections import defaultdict

import numpy as np
import pandas as pd

import lgimapy.vis as vis
from lgimapy.data import Database, Index
from lgimapy.latex import Document, latex_table
from lgimapy.utils import Time
from functools import lru_cache

vis.style()
# %%

db = Database()
db.load_market_data(start="1/1/2020", local=True)
doc = Document("HY_Spread_Cascades", path="latex/HY/2020_spreads", fig_dir=True)


ix = db.build_market_index(
    in_stats_index=True,
    in_hy_stats_index=True,
    special_rules="USCreditStatisticsFlag | USHYStatisticsFlag",
    OAS=(10, 3000),
)
# energy_sectors = [
#     "INDEPENDENT",
#     "REFINING",
#     "OIL_FIELD_SERVICES",
#     "INTEGRATED",
#     "MIDSTREAM",
# ]
# ix = db.build_market_index(
#     in_stats_index=True,
#     in_hy_stats_index=True,
#     sector=energy_sectors,
#     special_rules="(USCreditStatisticsFlag | USHYStatisticsFlag) & (~Sector)",
#     OAS=(10, 3000),
# )


rating = ("BB+", "BB-")
higher_rating = "BBB-"
cascade_percent = 10
rating = ("B+", "B-")
rating = "BBB-"
# %%
@lru_cache(maxsize=None)
def cascade(ix, rating, higher_rating, cascade_percent):
    # Get top percent of index with specified rating.
    rated_ix = ix.subset(rating=rating)
    if cascade_percent is None:
        return rated_ix

    # Sort by OAS within each date.
    df = rated_ix.df.sort_values(["Date", "OAS"])

    # Drop specified % of market value each day.
    drop_threshold = 1 - cascade_percent / 100

    def drop_cascade(df):
        mv_percentile = np.cumsum(df["MarketValue"]) / np.sum(df["MarketValue"])
        return df[mv_percentile < drop_threshold]

    gdf = df.groupby("Date").apply(drop_cascade)
    rated_ix_lowest_dropped = Index(gdf)

    if higher_rating is None:
        # No cascade down.
        return rated_ix_lowest_dropped

    # Get cascaded part of index with higher rating.
    higher_rated_ix = ix.subset(rating=higher_rating)
    # Sort by OAS within each date.
    df = higher_rated_ix.df.sort_values(["Date", "OAS"])
    # Drop specified % of market value each day.
    drop_threshold = 1 - cascade_percent / 100

    def get_cascade(df):
        mv_percentile = np.cumsum(df["MarketValue"]) / np.sum(df["MarketValue"])
        return df[mv_percentile >= drop_threshold]

    gdf = df.groupby("Date").apply(get_cascade)
    higher_rated_cascade_ix = Index(gdf)

    # Combine into new index.
    return rated_ix_lowest_dropped + higher_rated_cascade_ix


# %%
# Show that this methodology isn't just removing long bonds.
colors = ["#752333", "#0438A3", "#8E5C1D"]
n = 2
label = ["BBB-", "BB", "B"][n]
rating_color = colors[n]
df_list = [
    rated_ix.day(rated_ix.dates[0]),
    rated_ix.day(rated_ix.dates[55]),
    rated_ix.day(rated_ix.dates[-1]),
]

fig, axes = vis.subplots(1, 3, sharey=True, figsize=(12, 5))
for df, ax in zip(df_list, axes.flat):
    df.sort_values("OAS", inplace=True)
    df.plot.scatter(
        "OAD", "OAS", color=rating_color, alpha=0.5, s=25, ax=ax, label=label
    )
    mv_percentile = np.cumsum(df["OAS"]) / np.sum(df["OAS"])
    for i, color in zip([10, 20], ["black", "darkorchid"]):
        thresh = 1 - i / 100
        oas_line = df[mv_percentile < thresh]["OAS"].iloc[-1]
        ax.axhline(oas_line, color=color, label=f"Widest {i}%", lw=2, ls="--")
    date = df["Date"].iloc[0].strftime("%m/%d/%Y")
    ax.set_title(date)
axes[0].legend(loc="upper left")
# vis.show()
doc.add_figure(f"methodology_{label}", width=0.95, savefig=True)

# %%
colors = ["#752333", "#0438A3", "#8E5C1D"]
higher_ratings = [None, "BBB-", ("BB+", "BB-")]
ratings = ["BBB-", ("BB+", "BB-"), ("B+", "B-")]
names = ["BBB-", "BB", "B"]
iters = [ratings, higher_ratings, names, colors]

i = 0
for rating, higher_rating, name, color in zip(*iters):
    for casc, ls in zip([None, 10, 20], ["-", "--", ":"]):
        print(i)
        temp_ix = cascade(ix, rating, higher_rating, casc)
        i += 1


# %%
doc = Document("HY_Spread_Cascades", path="latex/HY/2020_spreads", fig_dir=True)
doc.add_preamble(margin={"left": 0.5, "right": 0.5, "top": 1, "bottom": 0.2})

fig, axes = vis.subplots(5, 1, sharex=True, figsize=(12, 15))
for rating, higher_rating, name, color in zip(*iters):
    for casc, ls in zip([None, 10, 20], ["-", "--", ":"]):
        if casc is not None:
            label = f"{name} {casc}% cascade"
        else:
            label = name
        temp_ix = cascade(ix, rating, higher_rating, casc)
        oas = temp_ix.market_value_weight("OAS")
        price = temp_ix.market_value_weight("DirtyPrice")
        ytw = temp_ix.market_value_weight("YieldToWorst") / 100
        oad = temp_ix.market_value_weight("OAD")
        mv = temp_ix.total_value() / 1e3
        for y, ax in zip([oas, price, ytw, oad, mv], axes.flat):
            ylabel = {
                "YieldToWorst": "Yield",
                "DirtyPrice": "Price",
                "OAS": "OAS (bp)",
                "OAD": "OAD (yr)",
                "total_value": "Market Value",
            }[y.name]

            fmt = None
            legend = None
            if ylabel == "Yield":
                fmt = "{x:.0%}"
            elif ylabel == "Price":
                fmt = "${x:.0f}"
            elif ylabel == "OAS (bp)":
                legend = {"loc": "upper left"}
            elif ylabel == "Market Value":
                fmt = "${x:.0f}B"

            vis.plot_timeseries(
                y,
                xtickfmt="auto",
                ytickfmt=fmt,
                ylabel=ylabel,
                ls=ls,
                color=color,
                alpha=0.8,
                label=label,
                legend=legend,
                ax=ax,
            )
# fig.suptitle('Ex-Energy')
doc.add_figure("cascade", savefig=True, width=0.95)
# doc.add_figure('cascade_ex_energy', savefig=True, width=0.95)
# doc.save_tex()

# %%
rating = "BBB-"
higher_rating = None
rating = ("BB+", "BB-")
higher_rating = "BBB-"
rating = ("B+", "B-")
higher_rating = ("BB+", "BB-")


d = defaultdict(list)
for casc in [None, 10, 20]:
    temp_ix = cascade(ix, rating, higher_rating, casc)
    oas = temp_ix.market_value_weight("OAS")
    price = temp_ix.market_value_weight("DirtyPrice").round(2)
    ytw = temp_ix.market_value_weight("YieldToWorst")
    oad = temp_ix.market_value_weight("OAD")
    mv = temp_ix.total_value() / 1e3
    d["OAS (bp)"].append(int(oas[-1]))
    d["Price (\\$)"].append(np.round(float(price[-1]), 2))
    d["Yield (%)"].append(ytw[-1].round(2))
    d["OAD (yr)"].append(oad[-1].round(3))
    d["Market Value (\\$B)"].append(int(mv[-1]))

table_df = pd.DataFrame(d, index=["Current", "10% Cascade", "20% Cascade"])
# print(latex_table(table_df, col_fmt='lccccc', caption='B Ex-Energy'))
print(latex_table(table_df, col_fmt="lccccc", caption="B"))
