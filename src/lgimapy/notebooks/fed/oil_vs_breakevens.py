import pandas as pd

from lgimapy import vis
from lgimapy.data import Database

vis.style()

# %%
start = "2015"
db = Database()
oil = db.load_bbg_data("OIL", "price", start=start)
be = db.load_bbg_data(["UST_10Y_BE", "UST_5Y5Y_BE"], "YTW", start=start)
df = pd.concat((be, oil), axis=1)
df.columns = db.bbg_names(df.columns)

vis.plot_double_y_axis_timeseries(
    # df["UST 10Y Breakeven"],
    df["UST 5Y5Y Breakeven"].dropna(),
    df["WTI Crude"].dropna(),
    ytickfmt_left="{x:.2%}",
    ytickfmt_right="${x:.0f}",
    color_left="k",
    color_right="darkgreen",
    plot_kws={"alpha": 0.7},
)
# vis.savefig("10y_breakeven_vs_oil")
# vis.savefig("5y5y_breakeven_vs_oil")
vis.show()

# %%
