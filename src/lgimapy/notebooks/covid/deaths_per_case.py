import pandas as pd

from lgimapy import vis
from lgimapy.bloomberg import bdh
from lgimapy.utils import root

vis.style()

# %%
offset = 20

df = pd.DataFrame()
cols = {
    "NCOVUSCA": "US_cases",
    "NCOVUSDE": "US_deaths",
    "NCOVGBCA": "UK_cases",
    "NCOVGBDE": "UK_deaths",
}
df = (
    bdh(cols.keys(), "Index", "PX_LAST", start="1/1/2020")
    .diff()
    .iloc[:-1, :]
    .rename(columns=cols)
)
df = df.rolling(7).mean()
df["US_deaths_offset"] = df["US_deaths"].shift(-offset)
df["UK_deaths_offset"] = df["UK_deaths"].shift(-offset)
df["UK_deaths_per_case"] = df["UK_deaths_offset"] / df["UK_cases"]
df["US_deaths_per_case"] = df["US_deaths_offset"] / df["US_cases"]

# %%
df_norm = df / df.max()
fig, axes = vis.subplots(1, 2, figsize=(10, 6), sharey=True)
for country, ax in zip(["UK", "US"], axes.flat):
    vis.plot_timeseries(
        df_norm[f"{country}_cases"],
        color="navy",
        ax=ax,
        label=f"Cases" if country == "UK" else "_",
        legend=False,
    )
    vis.plot_timeseries(
        df_norm[f"{country}_deaths_offset"],
        color="darkorchid",
        ax=ax,
        label=f"Deaths\n(offset 20 days)" if country == "UK" else "_nolegend_",
        legend=False,
    )
    vis.format_yaxis(ax, "{x:.0%}")
vis.legend(axes[0], fontsize=10, loc="upper left")
axes[0].set_ylabel("Percent of Peak")
axes[0].set_title("UK", fontweight="bold")
axes[1].set_title("US", fontweight="bold")
vis.savefig("covid_cases_vs_deaths")


# %%
df_recent = df[df.index >= pd.to_datetime("8/1/2020")]
fig, ax = vis.subplots()
vis.plot_timeseries(
    df_recent["UK_deaths_per_case"],
    color="skyblue",
    label="UK",
    ylabel="Case Fatality Rate",
    ytickfmt="{x:.1%}",
    ax=ax,
)
vis.plot_timeseries(
    df_recent["US_deaths_per_case"],
    color="navy",
    label="US",
    ylabel="Case Fatality Rate",
    ytickfmt="{x:.1%}",
    ax=ax,
)
vis.legend(ax)
vis.savefig("covid_case_fatality_rate")
