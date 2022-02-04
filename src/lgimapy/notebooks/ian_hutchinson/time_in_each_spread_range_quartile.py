import numpy as np
import pandas as pd

from lgimapy import vis
from lgimapy.data import Database

vis.style()

# %%
oas = db.load_bbg_data("GBP_IG", "OAS", start="2003")
bad_dates = pd.to_datetime(["2/12/2010", "2/15/2010"])
oas = oas[~oas.index.isin(bad_dates)]
# oas = oas[oas.index > pd.to_datetime('2008')]
diff = np.max(oas) - np.min(oas)
q = {0: np.min(oas)}
for q_i in [25, 50, 75]:
    q[q_i] = q_i / 100 * diff + q[0]
q[100] = np.max(oas)
colors = sns.color_palette("coolwarm", 4).as_hex()
quads = {
    "Q1": (colors[0], 0, 25),
    "Q2": (colors[1], 25, 50),
    "Q3": (colors[2], 50, 75),
    "Q4": (colors[3], 75, 100),
}
# %%
fig, ax = vis.subplots(figsize=(8, 6))
vis.plot_timeseries(oas, ax=ax, color="k", label="GBP IG", ylabel="OAS")
for quad, (color, low, high) in quads.items():
    quad_oas = oas[(oas < q[high]) & (oas >= q[low])]
    pct = len(quad_oas) / len(oas)
    ax.fill_between(
        oas.index,
        q[low],
        q[high],
        color=color,
        alpha=0.5,
        label=f"{quad}: {pct:.0%}",
    )

ax.legend(bbox_to_anchor=(1, 1), loc=2)
ax.set_title(
    "Time Spent in each Spread Range Quadrant",
    fontweight="bold",
    fontsize=15,
)
vis.savefig("since_2008")

oas.to_frame().to_csv("GBP_IG_OAS.csv")
for k, v in q.items():
    print(f"{k} %tile: {v:.0f}")
