from collections import defaultdict

import numpy as np
import pandas as pd

from lgimapy.data import Database
from lgimapy.utils import load_json

from lgimapy import vis

# %%

db_old = Database()
db = Database()
index = load_json("indexes")
db_old.load_market_data(start="1/1/2015", end="1/1/2017", local=True)
db.load_market_data(local=True)

sector_kwargs = {
    "All": {},
    "A": {"rating": ("AAA", "A-")},
    "BBB": {"rating": ("BBB+", "BBB-")},
}
colors = ["navy", "darkorchid", "firebrick"]

sectors = [
    "ENERGY",
    "MIDSTREAM",
    "INDEPENDENT",
    # "REFINING",
    "OIL_FIELD_SERVICES",
    "INTEGRATED",
]
sector = "REFINING"
# %%
for sector in sectors:
    # %%
    ix_old = db_old.build_market_index(**index[sector])
    ix_curr = db.build_market_index(**index[sector])

    pct_list = []
    oas_list = []
    current_oas = []
    for name, kwargs in sector_kwargs.items():
        ix = ix_old.subset(**kwargs)
        ix_curr_sub = ix_curr.subset(**kwargs)
        oas = ix.market_value_weight("OAS").rename(name)
        oas_list.append(oas)
        curr_spread = ix_curr_sub.market_value_weight("OAS")[0]
        current_oas.append(curr_spread)
        pct_list.append(curr_spread / np.max(oas))
    df = pd.concat(oas_list, axis=1, sort=True)

    # %%
    vis.style()
    fig, ax = vis.subplots(figsize=(12, 8))
    vis.plot_multiple_timeseries(
        df,
        ylabel="OAS",
        title=index[sector]["name"],
        c_list=colors,
        ax=ax,
        legend=False,
    )

    for spread, pct, color in zip(current_oas, pct_list, colors):
        ax.axhline(
            spread,
            lw=1.5,
            ls="--",
            color=color,
            label=f"Current Level: {spread:.0f}, ({pct:.0%})",
        )
    ax.legend()

    vis.savefig(sector.lower())


# %%
ix_old = db_old.build_market_index(**index["ENERGY"])
ix_curr = db.build_market_index(**index["ENERGY"])
tickers = set(ix_old.tickers) & set(ix_curr.tickers)

d = defaultdict(list)
for ticker in tickers:
    ix = ix_old.subset(ticker=ticker)
    ix_curr_sub = ix_curr.subset(ticker=ticker)
    oas = ix.market_value_weight("OAS").rename(name)
    max_spread = np.max(oas)
    curr_spread = ix_curr_sub.market_value_weight("OAS")[0]
    d["current spread"].append(curr_spread)
    d["max spread in 2016"].append(max_spread)
    d["pct of max"].append(curr_spread / max_spread)


df = pd.DataFrame(d, index=tickers).sort_values("pct of max")
df.to_csv("energy_comp_to_2016_by_ticker.csv")
# %%

ix_old = db_old.build_market_index(**index["ENERGY"], maturity=(8.5, 12))
ix_curr = db.build_market_index(**index["ENERGY"], maturity=(8.5, 12))
tickers = set(ix_old.tickers) & set(ix_curr.tickers)

d = defaultdict(list)
for ticker in tickers:
    ix = ix_old.subset(ticker=ticker)
    ix_curr_sub = ix_curr.subset(ticker=ticker)
    oas = ix.market_value_weight("OAS")
    max_spread = np.max(oas)
    curr_spread = ix_curr_sub.market_value_weight("OAS")[0]
    d["current spread"].append(int(curr_spread))
    d["max spread in 2016"].append(int(max_spread))
    d["pct of max"].append(np.round(curr_spread / max_spread, 2))


df = pd.DataFrame(d, index=tickers).sort_values("pct of max")
df.to_csv("energy_comp_to_2016_by_ticker_10y.csv")
