import numpy as np
import pandas as pd

from lgimapy import vis
from lgimapy.data import Database
from lgimapy.utils import root

vis.style()

# %%
db = Database()
db.load_market_data(start=db.date("HY_START"))

ix = db.build_market_index(in_H4UN_index=True)
oas = ix.OAS()
ytw = ix.market_value_weight("YieldToWorst") / 100

baml_df = (
    pd.read_csv(
        root("data/HY/h4un_oas_ytw.csv"),
        index_col=0,
        parse_dates=True,
        infer_datetime_format=True,
        usecols=[0, 1, 3],
        skiprows=1,
    )
    .dropna(how="all")
    .rename_axis(None)
)
baml_df.columns = ["OAS", "YTW"]
baml_df = baml_df[baml_df.index >= oas.index[0]]
baml_df["YTW"] /= 100
# %%
# Compare OAS.
diff = (baml_df["OAS"] - oas).fillna(method="ffill")
med = np.median(diff)
fig, ax = vis.subplots()
rax = ax.twinx()
rax.plot(diff, color="grey", alpha=0.6, lw=1.5)
rax.fill_between(diff.index, 0, diff.values, color="grey", alpha=0.2)
rax.set_ylabel("Difference", color="grey")
rax.tick_params(axis="y", colors="grey")
med_kwargs = {
    "color": "grey",
    "ls": "--",
    "lw": 2,
    "label": f"Median: {med:.0f} bp",
}
rax.axhline(med, **med_kwargs)
rax.grid(False)
rax.spines["right"].set_position(("axes", 1.05))
rax.spines["right"].set_visible(True)
rax.spines["right"].set_linewidth(1.5)
rax.spines["right"].set_color("lightgrey")
rax.tick_params(right="on", length=5)

kwargs = {"ax": ax, "alpha": 0.9}
med_kwargs.update({'alpha': 0})
vis.plot_timeseries(
    oas,
    color="navy",
    label="BBG",
    title="H4UN Index OAS",
    ylabel="OAS",
    median_line=True,
    median_line_kws=med_kwargs,
    **kwargs,
)
vis.plot_timeseries(baml_df["OAS"], color="firebrick", label="BAML", **kwargs)

vis.savefig('H4UN_OAS')

# %%
# Compare YTW.
diff = (baml_df["YTW"] - ytw).fillna(method="ffill")
med = np.median(diff)
fig, ax = vis.subplots()
rax = ax.twinx()
rax.plot(diff, color="grey", alpha=0.6, lw=1.5)
rax.fill_between(diff.index, 0, diff.values, color="grey", alpha=0.2)
rax.set_ylabel("Difference", color="grey")
rax.tick_params(axis="y", colors="grey")
med_kwargs = {
    "color": "grey",
    "ls": "--",
    "lw": 2,
    "label": f"Median: {med:.2%}",
}
rax.axhline(med, **med_kwargs)
rax.grid(False)
rax.spines["right"].set_position(("axes", 1.05))
rax.spines["right"].set_visible(True)
rax.spines["right"].set_linewidth(1.5)
rax.spines["right"].set_color("lightgrey")
rax.tick_params(right="on", length=5)

kwargs = {"ax": ax, "alpha": 0.9}
med_kwargs.update({'alpha': 0})
vis.plot_timeseries(
    ytw,
    color="navy",
    label="BBG",
    title="H4UN Index YTW",
    ylabel="YTW",
    median_line=True,
    median_line_kws=med_kwargs,
    **kwargs,
)

vis.plot_timeseries(baml_df["YTW"], color="firebrick", label="BAML", **kwargs)
vis.format_yaxis(ax, ytickfmt="{x:.1%}")
vis.format_yaxis(rax, ytickfmt="{x:.2%}")
vis.savefig('H4UN_YTW')
