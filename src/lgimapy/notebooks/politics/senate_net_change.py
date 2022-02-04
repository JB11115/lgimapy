import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lgimapy import vis

vis.style()

# %%
df = pd.DataFrame(
    (
        (4, 1),
        (3, 1),
        (2, 1),
        (1, 1),
        (0, 7.5),
        (-1, 58),
        (-2, 18),
        (-3, 20),
        (-4, 1),
        (-5, 1),
        (-6, 1),
        (-7, 1),
    ),
    columns=["net_chg", "pct"],
)
df["pct"] = df["pct"] / df["pct"].sum()
df["lead"] = df["net_chg"] + 3
cum_r = df[df["lead"] > 0].sort_values("lead", ascending=False)
cum_r["cum_pct"] = np.cumsum(cum_r["pct"])
cum_d = df[df["lead"] < 0].sort_values("lead")
cum_d["cum_pct"] = np.cumsum(cum_d["pct"])
cum_unched = df[df["lead"] == 0].copy()
cum_unched["cum_pct"] = cum_unched["pct"]
df = pd.concat([cum_r, cum_unched, cum_d]).sort_values(
    "net_chg", ascending=False
)
# %%
plt.rcParams["hatch.color"] = "steelblue"
plt.rcParams["hatch.linewidth"] = 4
fig, ax = vis.subplots(figsize=(8, 6))
kwargs = {"alpha": 0.8, "width": 1, "edgecolor": "k", "linewidth": 1}
ax.bar(cum_d["lead"], cum_d["cum_pct"], color="steelblue", **kwargs)
ax.bar(cum_r["lead"], cum_r["cum_pct"], color="firebrick", **kwargs)
ax.bar(
    cum_unched["lead"],
    cum_unched["cum_pct"],
    hatch="//",
    color="firebrick",
    alpha=kwargs["alpha"],
    width=kwargs["width"],
)
ax.bar(
    cum_unched["lead"], cum_unched["cum_pct"], color="none", **kwargs,
)

vis.format_yaxis(ax, ytickfmt="{x:.0%}")
ax.grid(False, axis="x")
xticks = [f"{x:+}" if x != 0 else 0 for x in df["lead"].sort_values().abs()]
ax.set_xticks(df["lead"].sort_values())
ax.set_xticklabels(xticks)
color = "steelblue"
for tick in ax.xaxis.get_ticklabels():
    if tick.get_text() == "0":
        color = "grey"
    tick.set_color(color)
    if tick.get_text() == "0":
        color = "firebrick"

ax.set_title(
    "Probability each Party has at least\n __ Seat Margin in the Senate",
    fontweight="bold",
)
vis.savefig("senate_election_outcome")

# %%
