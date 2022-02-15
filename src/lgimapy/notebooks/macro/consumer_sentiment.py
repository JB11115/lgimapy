import numpy as np
import pandas as pd

from lgimapy import vis
from lgimapy.bloomberg import bdh
from lgimapy.data import Database

vis.style()

# %%
start = "1/1/2001"
db = Database()
df = bdh(["CONSSENT", "COMFCOMF"], "Index", "PX_LAST", start=start).sort_index()
df.columns = db.columns = ["U Mich Consumer Confidence", "BBG Consumer Comfort"]

u_mich = 61.7
bbg = np.nan
today = pd.to_datetime("today").date()
s = pd.Series([u_mich, bbg], index=df.columns, name=today)
df = df.append(s)
df.index = pd.to_datetime(df.index)
df.tail()
# %%
vis.plot_double_y_axis_timeseries(
    df["U Mich Consumer Confidence"].dropna(),
    df["BBG Consumer Comfort"].dropna(),
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
