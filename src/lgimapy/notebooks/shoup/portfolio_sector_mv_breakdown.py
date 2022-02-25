from collections import defaultdict

import pandas as pd

from lgimapy.data import Database

# %%
strategy_map = {
    "US Credit": ("CITMC", None),
    "US Corp": ("RSURC", None),
    "US Corp ex Subordinated": (
        "RSURC",
        {"collateral_type": "SUBORDINATED", "special_rules": "~CollateralType"},
    ),
}

db = Database()
sectors = db.IG_sectors()
d = defaultdict(list)
for strategy, (account, kwargs) in strategy_map.items():
    port = db.load_portfolio(account=account)
    if kwargs is not None:
        port = port.subset(**kwargs)
        port._constraints = {}

    bm_mv = port.bm_df["MarketValue"].sum()
    for sector in sectors:
        kws = db.index_kwargs(
            sector, unused_constraints=["OAS", "in_stats_index"]
        )
        sector_port = port.subset(**kws)
        sector_mv = sector_port.bm_df["MarketValue"].sum()
        d[sector_port.name].append(sector_mv / bm_mv)

df = pd.DataFrame(d, index=strategy_map.keys()).T
df.to_csv("Sector_MV_pct_by_Strategy.csv")
