from collections import defaultdict
from datetime import datetime as dt
from inspect import cleandoc
from shutil import copy

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from lgimapy import vis
from lgimapy.data import Database
from lgimapy.latex import Document
from lgimapy.utils import root, get_ordinal, mkdir

vis.style()

# %%
db = Database()
db.load_market_data(start=db.date('5y'))

# %%
ix_d = {}
ix_d["mc"] = db.build_market_index(in_stats_index=True)
ix_d["lc"] = db.build_market_index(in_stats_index=True, maturity=(10, None))
ix_d["10y"] = db.build_market_index(in_stats_index=True, maturity=(8, 12))
ix_d["30y"] = db.build_market_index(in_stats_index=True, maturity=(25, 32))

# %%
bbg_df = db.load_bbg_data(
    ["US_HY", "CDX_IG", "CDX_HY"], "OAS", start=db.date("5y")
)
ix_mc = ix_d["mc"].subset(start=db.date("5y"))
oas = ix_mc.market_value_median("OAS").rename("US_IG")
df = pd.concat([bbg_df, oas], axis=1, sort=True).dropna(how="any")
df["HY/IG Cash"] = df["US_HY"] / df["US_IG"]
df["HY/IG CDX"] = df["CDX_HY"] / df["CDX_IG"]

left_last = 100 * df["HY/IG Cash"].rank(pct=True).iloc[-1]

# Plot
fig, ax = vis.subplots(figsize=(10, 5))

vis.plot_timeseries(df["HY/IG Cash"], color="navy", alpha=0.9, lw=2, ax=ax, label='_')
ax.set_ylabel("Ratio")
ax.axhline(np.median(df["HY/IG Cash"]), ls=":", lw=1.5, color="navy", label='Median')

pct = {x: np.percentile(df["HY/IG Cash"], x) for x in [5, 95]}
pct_line_kwargs = {"ls": "--", "lw": 1.5, "color": "dimgrey"}
ax.axhline(pct[5], label="5/95 %tiles", **pct_line_kwargs)
ax.axhline(pct[95], label="_nolegend_", **pct_line_kwargs)
ax.set_title("HY:IG spread ratio", fontweight="bold")
vis.format_xaxis(ax_right, df["HY/IG CDX"], "auto")
vis.legend(ax, loc="upper left")
plt.tight_layout()
plt.savefig('HY_IG_cash_ratio.tiff', dpi=300, bbox_inches="tight")

df['HY/IG Cash'].to_frame().to_csv("HY_IG_cash_ratio.csv")


pct
np.median(df["HY/IG Cash"])
