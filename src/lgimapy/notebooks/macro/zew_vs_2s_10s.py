import pandas as pd

from lgimapy import vis
from lgimapy.data import Database

vis.style()

# %%

db = Database()
zew = db.load_bbg_data("US_ZEW", "LEVEL")
twos_tens = db.load_bbg_data(["UST_2Y", "UST_10Y", "UST_10Y_RY"], "YTW")
comms = db.load_bbg_data(["GOLD", "COPPER"], "PRICE")

df = pd.concat((zew, twos_tens, comms), axis=1).fillna(method="ffill").dropna()
df["2s10s"] = df["UST_10Y"] - df["UST_2Y"]
df["C/G"] = df["COPPER"] / df["GOLD"]
# %%
vis.plot_double_y_axis_timeseries(
    df["2s10s"],
    df["US_ZEW"],
    ylabel_left="UST 10y - 2y",
    ylabel_right="US ZEW Expectations",
    plot_kws_left={"color": "k"},
    plot_kws_right={"color": "darkgreen"},
    figsize=(12, 6),
    ytickfmt_left="{x:.1%}",
)
vis.savefig("US_ZEW_vs_2s_10s")


# %%
vis.plot_double_y_axis_timeseries(
    df["UST_10Y"],
    df["C/G"],
    start="1/1/2010",
    ylabel_left="UST 10y",
    ylabel_right="Copper/Gold Ratio",
    plot_kws_left={"color": "k"},
    plot_kws_right={"color": "#b87333"},
    figsize=(12, 6),
    ytickfmt_left="{x:.1%}",
)
vis.savefig("Copper_Gold_Ratio_vs_10y")

# %%
vis.plot_double_y_axis_timeseries(
    df["UST_10Y_RY"],
    df["GOLD"],
    start="1/1/2012",
    ylabel_left="UST 10y Real Yield",
    ylabel_right="Gold (Inverted)",
    plot_kws_left={"color": "k"},
    plot_kws_right={"color": "goldenrod"},
    figsize=(12, 6),
    ytickfmt_left="{x:.1%}",
    ytickfmt_right="${x:,.0f}",
    invert_right_axis=True,
)
vis.savefig("Gold_vs_10y_Real_Yield")
