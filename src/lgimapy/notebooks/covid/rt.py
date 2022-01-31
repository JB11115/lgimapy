"""
Data from
https://rt.live
"""

import numpy as np
import pandas as pd
import seaborn as sns

from lgimapy import vis
from lgimapy.data import Database
from lgimapy.utils import root

vis.style()

# %%
db = Database()

fid = root("data/covid/rt.csv")
df = pd.read_csv(
    fid, index_col=0, parse_dates=True, infer_datetime_format=True
).rename_axis(None)
states = sorted(df["region"].unique())
states.remove("DC")
df = df.pivot(columns="region", values="mean")[states]
df = df[df.index >= pd.to_datetime("4/15/2020")]

# %%
prev_date = db.date("3w")
curr_date = df.index[-1]
curr = df.loc[curr_date]
prev = df.loc[prev_date]

n = len(states)
pal = "dark"
sns.set_palette(pal, n)
fig, ax = vis.subplots(figsize=(8, 6))
for state in states:
    ax.plot(prev.loc[state], curr.loc[state], "o", ms=8, alpha=0.8)

low = min(prev.append(curr))
high = max(prev.append(curr))
ax.plot([low, high], [low, high], "--", c="k", lw=1, alpha=0.6)
ax.set_xlabel(f"$R_t$ on {prev_date.strftime('%m/%d')}")
ax.set_ylabel(f"$R_t$ on {curr_date.strftime('%m/%d')}")
# ax.set_xlim(0.95, 1.3)
# ax.set_ylim(0.95, 1.3)
rt_chg = np.sign(curr - prev)
ax.set_title(
    (
        "Instantaneous Reproductive Number in Each State \n"
        f"$\mathbf{{R_t}}$ > 1 in {np.sum(curr > 1)} states; "
        f"$\mathbf{{R_t}}$ growing in {np.sum(rt_chg > 0)} states"
    ),
    fontweight="bold",
    fontsize=14,
)
vis.savefig("covid_Rt")


chg_df = (curr - prev).sort_values()

# %%
chg_df[chg_df < 0].round(3)
chg_df[chg_df > 0].round(3).sort_values(ascending=False)


curr.sort_values(ascending=False)
