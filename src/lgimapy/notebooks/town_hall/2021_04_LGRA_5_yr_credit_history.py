from inspect import cleandoc

import numpy as np

from lgimapy import vis
from lgimapy.data import Database
from lgimapy.utils import get_ordinal

vis.style()


# %%
index = 'US_IG'
start= 5

def plot_index_oas(index, start, fid=None, color='navy'):
    db = Database()
    fig, ax = vis.subplots(figsize=(8, 6))
    oas = db.load_bbg_data(index, "OAS", start=db.date(f"{start}y")).dropna()
    med = np.median(oas)
    pct = {x: np.percentile(oas, x) for x in [5, 95]}


    pctile = int(np.round(100 * oas.rank(pct=True)[-1]))
    ordinal = get_ordinal(pctile)
    lbl = cleandoc(
        f"""
        Historical Stats Index
        Last: {oas[-1]:.0f} bp ({pctile:.0f}{ordinal} %tile)
        Range: [{np.min(oas):.0f}, {np.max(oas):.0f}]
        """
    )

    vis.plot_timeseries(
        oas,
        color=color,
        # bollinger=True,
        ylabel="OAS",
        ax=ax,
        label=lbl,
    )
    ax.axhline(
        med, ls="--", lw=1.5, color="firebrick", label=f"Median: {med:.0f}"
    )
    pct_line_kwargs = {"ls": "--", "lw": 1.5, "color": "dimgrey"}
    ax.axhline(pct[5], label="5/95 %tiles", **pct_line_kwargs)
    ax.axhline(pct[95], label="_nolegend_", **pct_line_kwargs)

    title = f"$\\bf{{{start}yr}}$ $\\bf{{Stats}}$"
    ax.legend(fancybox=True, title=title, shadow=True)
    if fid is None:
        vis.show()
    else:
        vis.savefig(fid)

# plot_index_oas('US_IG', start=5, fid='US_IG')
plot_index_oas('US_HY', start=5, fid='US_HY', color='darkorchid')
