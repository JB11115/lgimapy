import numpy as np
import pandas as pd

from lgimapy import vis
from lgimapy.bloomberg import bdh
from lgimapy.data import Database

vis.style()

# %%
start = "1/1/2001"
db = Database()
df = bdh(["CONSSENT", "CONCCONF"], "Index", "PX_LAST", start=start).sort_index()
df.columns = db.columns = ["U Mich Consumer Confidence", "Conference Board"]

u_mich = np.nan
bbg = np.nan
today = pd.to_datetime("today").date()
s = pd.Series([u_mich, bbg], index=df.columns, name=today)
df = pd.concat((df, s.to_frame().T))
df.index = pd.to_datetime(df.index)
df.tail()
# %%
vis.plot_double_y_axis_timeseries(
    df["U Mich Consumer Confidence"].dropna(),
    df["Conference Board"].dropna(),
    # start="1/1/2020",
    ytickfmt_left="{x:.0f}",
    ytickfmt_right="{x:.0f}",
    color_left="k",
    color_right="darkgreen",
    plot_kws={"alpha": 0.7},
)
vis.savefig("consumer_confidence")
vis.show()

# %%
