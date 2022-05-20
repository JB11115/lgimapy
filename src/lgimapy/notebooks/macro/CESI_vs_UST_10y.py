import pandas as pd

from lgimapy import vis
from lgimapy.data import Database

vis.style()

# %%
db = Database()
cesi = "GLOBAL_ECO_SURP"
start = "2010"
df = db.load_bbg_data([cesi, "UST_10Y"], ["LEVEL", "YTW"], nan="drop")


df["UST_chg"] = df["UST_10Y"] - df["UST_10Y"].shift(12 * 5)
# %%
# vis.plot_double_y_axis_timeseries(
#     df["UST_chg"].rename("$\Delta$UST 10Y (12w)"),
#     df[cesi].rename("Citi Economic Surprise Index"),
#     color_left="k",
#     color_right="forestgreen",
#     ytickfmt_left="{x:.1%}",
#     figsize=(12, 5),
#     lw=2,
#     alpha=0.8,
# )
# # vis.savefig('UST_10Y_vs_CESI')
# vis.show()
fig, ax = vis.subplots()
kwargs = {"lw": 1.8, "alpha": 0.8, "ax": ax}
vis.plot_timeseries(
    1e4 * df["UST_chg"], color="k", label="$\Delta$UST 10y (12w; bp)", **kwargs
)
vis.plot_timeseries(
    df[cesi],
    color="forestgreen",
    label="Citi Economic Surprise Index",
    **kwargs
)
vis.legend(ax)
vis.savefig("UST_10Y_vs_CESI")
