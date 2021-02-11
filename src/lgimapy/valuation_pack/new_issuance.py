import numpy as np
import pandas as pd

import lgimapy.vis as vis
from lgimapy.data import Database

# %%


# %%
db = Database()
db.load_market_data()
ix_rets = db.build_market_index(in_returns_index=True)
ix_stat = db.build_market_index(in_stats_index=True)
ix_ni = ix_stat.subset(
    isin=ix_rets.isins, issue_years=(None, 40 / 365), special_rules="~ISIN"
)
new_issuance = ix_ni.df["AmountOutstanding"].sum()
print(f"${new_issuance/1e3:.0f} B")
