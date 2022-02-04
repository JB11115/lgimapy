import numpy as np
import pandas as pd

from lgimapy.data import Database
from lgimapy import vis

vis.style()
# %%
db = Database()
db.load_market_data(start="1/1/2019", local=True)
# %%
def get_tail_isins(ix, start, end, col="PX_Adj_OAS", threshold=10):
    start_df = ix.subset(date=db.nearest_date(start)).ticker_df
    end_df = ix.subset(date=db.nearest_date(end)).ticker_df
    start_df["start_col"] = start_df[col]
    chg_df = pd.concat((end_df, start_df["start_col"]), axis=1, join="inner")
    chg_df["diff"] = chg_df[col] - chg_df["start_col"]
    chg_df.sort_values("diff", inplace=True)
    chg_df["cum_mv"] = (
        np.cumsum(chg_df["MarketValue"]) / chg_df["MarketValue"].sum()
    )
    thresh = 1 - threshold / 100
    tail = chg_df[chg_df["cum_mv"] > thresh]
    return tail.index
    # return tail.index


ratio_date = db.nearest_date("9/19/2019")

ix = db.build_market_index(
    in_stats_index=True, OAS=(0, 3000), maturity=(10, None)
)
ix.calc_dollar_adjusted_spreads()
ixs = {}
bbb_ix = ix.subset(rating=("BBB+", "BBB-"))
ixs["A Rated"] = ix.subset(rating=("AAA", "A-"))
tail = get_tail_isins(bbb_ix, ratio_date, "3/23/2020")
ixs["BBB Ex-Tail"] = bbb_ix.subset(ticker=tail, special_rules="~Ticker")
ixs["BBB Tail"] = bbb_ix.subset(ticker=tail)
ixs["HY"] = db.build_market_index(rating="HY", in_hy_stats_index=True)
ixs["HY"].df["PX_Adj_OAS"] = ixs["HY"].df["OAS"]

# %%
fig, ax = vis.subplots(figsize=(8, 6))
colors = ["#23345C", "#BD8A44", "#D75B66", "#8D2D56"]
ax.axhline(1, color="k", ls="--", lw=1, label="_nolegend_")
for color, (name, ix) in zip(colors, ixs.items()):
    price = ix.market_value_weight("DirtyPrice")
    spread = ix.market_value_weight("PX_Adj_OAS")
    current_spread = spread[-1]
    vis.plot_timeseries(
        spread - spread.loc[ratio_date],
        color=color,
        start=ratio_date,
        alpha=0.9,
        lw=3,
        ax=ax,
        label=name,
        ylabel=f"Spread Change from Sept. 2019 (bp)",
    )
ax.legend(loc="upper left", fancybox=True, shadow=True)
ax.set_ylim((-60, 550))

vis.savefig("px_adj_spread_ts_abs")
# vis.show()
