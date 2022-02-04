from lgimapy import vis
from lgimapy.bloomberg import bdh

vis.style()

# %%

d = {}
d["5y"] = bdh("USSW5", "Curncy", "PX_LAST", start="1/1/2010") / 100
d["10y"] = bdh("USSW10", "Curncy", "PX_LAST", start="1/1/2010") / 100
d["5y5y fwd"] = ((1 + d["10y"]) ** 10 / (1 + d["5y"]) ** 5) ** (1 / 5) - 1

# %%
fig, ax = vis.subplots(figsize=(8, 6))
colors = {"5y": "k", "10y": "navy", "5y5y fwd": "orchid"}
for label, s in d.items():
    vis.plot_timeseries(s, color=colors[label], lw=1.5, label=label, ax=ax)
vis.format_yaxis(ax, "{x:.0%}")
ax.legend(shadow=True, fancybox=True, fontsize=14)
vis.savefig("swaps")

# %%
