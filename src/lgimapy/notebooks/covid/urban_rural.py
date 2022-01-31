"""
Data from
https://covid.cdc.gov/covid-data-tracker/#pop-factors_newcases
"""

import pandas as pd
import seaborn as sns

from lgimapy import vis
from lgimapy.utils import root

vis.style()

# %%
fid = root("data/covid/urban_rural.csv")
df_full = pd.read_csv(fid, skiprows=2)
df_NCHS = df_full[df_full["Classification"] == "NCHS_class"].set_index("Date")
df_NCHS.index = pd.to_datetime(df_NCHS.index)

col = "7-Day Avg New Cases per 100K"
classifications = {
    1: "Large Central Metro",
    2: "Large Fringe Metro",
    3: "Medium Metro",
    4: "Small Metro",
    5: "Micropolitan",
    6: "Most Rural",
}
df_list = []
for n, classification in classifications.items():
    df_temp = df_NCHS[
        (df_NCHS["Category"] == n) & (df_NCHS["Location"] == "United States")
    ][col].rename(classification)
    df_list.append(df_temp)

df = pd.concat(df_list, axis=1).rename_axis(None)
df = df[df.index >= pd.to_datetime("3/1/2020")].copy()
# %%
fig, ax = vis.subplots(figsize=(10, 6))
colors = sns.color_palette("cubehelix", len(df.columns) + 1).as_hex()
for col, color in zip(df.columns, colors):
    vis.plot_timeseries(
        df[col],
        alpha=0.8,
        color=color,
        label=col,
        ax=ax,
        lw=2,
        start=None,
    )
ax.set_ylabel("New Cases per 100k")
ax.legend(shadow=True, fancybox=True)
vis.savefig("urban_rural_covid_cases")
