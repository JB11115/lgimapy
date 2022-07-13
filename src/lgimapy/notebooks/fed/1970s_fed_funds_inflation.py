import pandas as pd

from lgimapy import vis
from lgimapy.bloomberg import bdh

vis.style()

# %%

start = "1/1/1970"
fed_funds = bdh("FDTR", "Index", "PX_LAST", start=start).squeeze()
cpi = bdh("CPI YOY", "Index", "PX_LAST", start=start).squeeze()

# %%
df = (
    pd.concat(
        (
            fed_funds.rename("Fed Funds"),
            cpi.rename("CPI YoY"),
        ),
        axis=1,
    )
    .fillna(method="ffill")
    .fillna(method="bfill")
)


vis.plot_double_y_axis_timeseries(
    df['Fed Funds'],
    df['CPI YoY'],
    color_left='k',
    color_right='dodgerblue',
    figsize=(20, 12),
    lw=3,
)

vis.show()
