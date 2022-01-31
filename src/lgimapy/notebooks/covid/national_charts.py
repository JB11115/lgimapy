"""
Data from https://covidtracking.com/data/download
"""


import pandas as pd

from lgimapy import vis
from lgimapy.utils import root

vis.style()

# %%

fid = root("data/covid/national-history.csv")
df = pd.read_csv(fid, index_col=0).rename_axis(None)
df.index = pd.to_datetime(df.index)
df.sort_index(inplace=True)
df["currently_hospitalized"] = df["hospitalizedCurrently"] / 1e3
df["new_cases"] = df["positiveIncrease"] / 1e3
df["new_tests"] = df["totalTestResultsIncrease"] / 1e6
df["positive_pct"] = df["positiveIncrease"] / df["totalTestResultsIncrease"]
df["cases_7day"] = df["new_cases"].rolling(7).mean()
df["tests_7day"] = df["new_tests"].rolling(7).mean()
df["positive_pct_7day"] = df["positive_pct"].rolling(7).mean()
df = df[df.index > pd.to_datetime("3/15/2020")].copy()
# %%


# fig, ax_left = vis.subplots()
# ax_right = ax_left.twinx()
# df.plot.bar(y='new_tests', rot=0, ax=ax_left, color='steelblue', alpha=0.5, label='_nolegend_')
# # df.plot.bar(y='new_cases', rot=0, ax=ax_left, color='steelblue', label='_nolegend_')
# # vis.plot_timeseries(df['positive_%'], label='_nolegend_', color='darkorange', ax=ax_right)
# ax_left.get_legend().remove()
# vis.format_yaxis(ax_left, ytickfmt='{x:.0f}M')
# vis.format_yaxis(ax_right, ytickfmt='{x:.0%}')
# vis.make_patch_spines_invisible(ax_left)
# vis.make_patch_spines_invisible(ax_right)
# ax_right.grid(False)
# ax_right.spines["right"].set_position(("axes", 1.05))
# ax_right.spines["right"].set_visible(True)
# ax_right.spines["right"].set_linewidth(1.5)
# ax_right.spines["right"].set_color("lightgrey")
# ax_right.tick_params(right="on", length=5)
# ax_left.set_ylabel('Tests', color='steelblue')
# ax_right.set_ylabel('% Positive', color='darkorange')
# ax_left.tick_params(axis="y", colors='steelblue')
# ax_right.tick_params(axis="y", colors='darkorange')
# # vis.format_xaxis(ax_left, s=df['new_cases'])
# vis.show()

# %%
vis.plot_triple_y_axis_timeseries(
    df["currently_hospitalized"],
    df["tests_7day"],
    df["positive_pct_7day"],
    ylabel_left="Currently Hospitalized",
    ylabel_right_inner="# Tests (7 Day Avg)",
    ylabel_right_outer="% Positive Tests (7 Day Avg)",
    ytickfmt_left="{x:.0f}k",
    ytickfmt_right_inner="{x:.1f}M",
    ytickfmt_right_outer="{x:.0%}",
    lw=4,
)
# vis.show()
vis.savefig("covid_update")

# %%
df["hosp_7day"] = df["currently_hospitalized"].rolling(7).mean()
df["hosp_1st_derivative"] = df["hosp_7day"].diff()
vis.plot_double_y_axis_timeseries(
    df["hosp_7day"],
    df["hosp_1st_derivative"],
    ylabel_left="Currently Hospitalized",
    ylabel_right="1st Derivative",
    ytickfmt_left="{x:.0f}k",
    plot_kws_left={"color": "steelblue"},
    plot_kws_right={"color": "coral"},
    lw=4,
    figsize=(8, 5),
)
vis.savefig("hospitilizations")
