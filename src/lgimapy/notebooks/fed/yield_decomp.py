from lgimapy import vis
from lgimapy.bloomberg import bdh
from lgimapy.data import Database

vis.style()

# %%
db = Database()
df = db.load_bbg_data(["UST_10Y_BE", "UST_10Y_RY"], "YTW", start="2010")

vis.plot_double_y_axis_timeseries(
    df["UST_10Y_BE"].rename("10Y Breakeven"),
    df["UST_10Y_RY"].rename("10Y Real Yield"),
    ytickfmt_left="{x:.1%}",
    ytickfmt_right="{x:.1%}",
)
vis.show()
