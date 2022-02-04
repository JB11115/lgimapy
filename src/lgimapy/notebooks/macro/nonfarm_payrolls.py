import warnings

import pandas as pd

from lgimapy import vis
from lgimapy.bloomberg import bdh

vis.style()

# %%

df = pd.DataFrame()
indexes = {"NSA": "NFP N", "SA": "NFP T"}
for key, index in indexes.items():
    df[key] = (
        bdh(index, "Index", "PX_LAST", start="1/1/2015").squeeze().diff()[1:]
    )

# %%
df_monthly = (df / 1e3).round(2)
df_monthly["month"] = pd.Series(df.index).dt.month.values
df_monthly["adj"] = df_monthly["SA"] - df_monthly["NSA"]
monthly_adjs = (
    df_monthly[df_monthly.index <= pd.to_datetime("1/1/2020")]
    .groupby("month")
    .mean()["adj"]
    .round(3)
    .to_dict()
)

# %%

df_2021 = df_monthly[df_monthly.index >= pd.to_datetime("/1/1/2021")].copy()
df_2021["sign"] = ((df_2021["adj"] * df_2021["NSA"]) > 0).astype(int)
df_2021 = df_2021.set_index("month").rename_axis(None).copy()

df_2021

df_rest_of_2021 = pd.DataFrame(columns=df_2021.columns)
df_rest_of_2021["adj"] = pd.Series(range(1, 13)).map(monthly_adjs)
df_rest_of_2021.index = range(1, 13)
df_rest_of_2021 = df_rest_of_2021[~df_rest_of_2021.index.isin(df_2021.index)]
df_2021 = df_2021.append(df_rest_of_2021)

nsa = []
for _, row in df_2021.iterrows():
    if row["sign"] == 1:
        nsa.append(row["NSA"] + row["adj"])
    elif row["sign"] == 0:
        nsa.append(row["NSA"])
    else:
        nsa.append(0)


df_2021["Non Seasonally Adjusted Payrolls"] = nsa
df_2021["Seasonal Adjustment"] = df_2021["adj"]
df_2021["Seasonally Adjusted Payrolls"] = df_2021["SA"]
df_2021["_nolegend_"] = df_2021["SA"]
df_2021["month_name"] = [
    pd.to_datetime(f"{month}/15/2021").strftime("%b") for month in df_2021.index
]
df_2021 = df_2021.set_index("month_name").rename_axis(None)
# %%
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fig, ax = vis.subplots()
    df_2021[
        ["Non Seasonally Adjusted Payrolls", "Seasonally Adjusted Payrolls"]
    ].plot.bar(ax=ax, color=["skyblue", "navy"], rot=0)
    df_2021[["Seasonal Adjustment", "_nolegend_"]].plot.bar(
        ax=ax, color=["darkorange", "navy"], rot=0
    )
    ax.set_ylabel("US Employees on Nonfarm Payrolls\nMoM Net Change")
    vis.format_yaxis(ax, "{x:.0f}M")
    ax.axhline(0, color="grey", alpha=0.8, lw=1)
    ax.legend(fancybox=True, shadow=True)
    ax.grid(False, axis="x")
    vis.savefig("Payrolls_2021")

df_2021
