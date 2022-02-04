import pandas as pd

from lgimapy import vis
from lgimapy.utils import root

vis.style()
# %%

fid = root("data/fed/CIE.csv")
df = (
    pd.read_csv(
        fid, index_col=0, parse_dates=True, infer_datetime_format=True
    ).rename_axis(None)
    / 100
)
df.columns = ["SPF 10y ahead PCE", "U Mich 5-10y ahead"]
# %%
fig, ax = vis.subplots(figsize=(10, 6))
colors = ["navy", "skyblue"]
for col, color in zip(df.columns, colors):
    s = df[col]
    label = f"{col}: {s.iloc[-1]:.2%}"
    vis.plot_timeseries(s, color=color, label=label, ax=ax)
vis.format_yaxis(ax, ytickfmt="{x:.1%}")
ax.legend(
    shadow=True,
    fancybox=True,
    ncol=2,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.1),
)
ax.set_title(
    "Fed Common Inflation Expectations\n\n", fontweight="bold", fontsize=16
)
vis.savefig("Fed_common_inflation_expectations")

# %%
