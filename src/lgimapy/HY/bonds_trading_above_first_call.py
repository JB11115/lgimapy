from collections import defaultdict
from inspect import cleandoc

import numpy as np
import pandas as pd

from lgimapy import vis
from lgimapy.data import Database, Index
from lgimapy.latex import Document
from lgimapy.utils import root, get_ordinal

vis.style()
# %%

db = Database()
db.load_market_data(start='1/1/2000')
ix = db.build_market_index(in_hy_stats_index=True)

# %%

rating_kwargs = {"BB": ("BB+", "BB-"), "B": ("B+", "B-"), "CCC": "CCC"}
neg_convexity = {}
for rating, rating_kws in rating_kwargs.items():
    rating_ix = db.build_market_index(rating=rating_kws)
    mv = rating_ix.total_value()
    df = rating_ix.df.copy()
    df["CallPrice"] = 100 + df["CouponRate"] / 2
    ix_neg_conv = Index(df[df["DirtyPrice"] >= df["CallPrice"]])
    neg_convexity[rating] = ix_neg_conv.total_value() / mv


oas = db.load_bbg_data('US_HY', 'OAS', start='1/1/2000')


# %%
colors = dict(zip(rating_kwargs.keys(), vis.colors("ryb")))
fig, axes = vis.subplots(2, 1, figsize=(14, 8), sharex=True)
for rating, neg_conv in neg_convexity.items():
    c = colors[rating]
    vis.plot_timeseries(
        neg_conv,
        color=c,
        lw=2,
        # alpha=0.5,
        median_line=True,
        median_line_kws={"color": c, 'alpha': 0.8, 'label': "_nolegend_"},
        label=f"{rating}: {neg_conv.iloc[-1]:.0%}",
        ax=axes[0],
        title='Bonds Trading Above First Call\n',
    )



vis.format_yaxis(axes[0], ytickfmt='{x:.0%}')
axes[0].legend(fancybox=True, shadow=True, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.06))
vis.plot_timeseries(oas, ylabel='OAS', ax=axes[1], color='k', lw=2)
vis.show()
