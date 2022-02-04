"""
Data from https://gis.cdc.gov/grasp/covidnet/COVID19_5.html
"""

from datetime import date, timedelta

import pandas as pd

from lgimapy import vis
from lgimapy.utils import root

vis.style()

# %%
fid = root("data/covid/Weekly_Data_Counts.csv")
df = pd.read_csv(fid, skiprows=2)
df = df[df.isna().sum(axis=1) < 4].iloc[:-1].copy()


def get_date(week):
    year, week = map(int, week.split("-"))
    first = date(year, 1, 1)
    base = 1 if first.isocalendar()[1] == 1 else 8
    return first + timedelta(
        days=base - first.isocalendar()[2] + 7 * (week - 1)
    )


df["Date"] = pd.to_datetime([get_date(week) for week in df["WEEK_NUMBER"]])
df.set_index("Date", inplace=True)
cols = [c for c in df.columns if c.endswith("YR")]
df = df[cols]
cols
# %%
df_sa = df[
    (df.index >= pd.to_datetime("6/1/2020"))
    * (df.index <= pd.to_datetime("10/1/2020"))
]
df_sa_normed = df_sa.divide(df_sa.max(), axis=1)

# %%
col_colors = {"18-49 YR": "magenta", "65+ YR": "navy"}
plot_cols = list(col_colors.keys())

# %%
df_sa = df[
    (df.index >= pd.to_datetime("6/15/2020"))
    * (df.index <= pd.to_datetime("10/1/2020"))
]
df_sa_normed = df_sa.divide(df_sa.max(), axis=1)
df_july = df_sa_normed[df_sa_normed.index >= "7/13/2020"]

col_colors = {"18-49 YR": "magenta", "65+ YR": "navy"}
fig, ax = vis.subplots(figsize=(7, 6))
for col, color in col_colors.items():
    vis.plot_timeseries(
        df_sa_normed[col],
        label=col.lower(),
        ylabel='Hospitilizations (% of Peak)',
        color=color,
        ax=ax,
        # title="Summer/Autumn, before Vaccines",
    )
ax.fill_between(
    df_july.index,
    df_july[plot_cols[0]],
    df_july[plot_cols[1]],
    color="grey",
    alpha=0.3,
)
ax.annotate(
    "Younger\ndecline\nquicker",
    xy=(pd.to_datetime("8/15/2020"), 0.65),
    xytext=(pd.to_datetime("9/20/2020"), 0.8),
    fontweight="bold",
    ha="center",
    va="bottom",
    bbox=dict(boxstyle="round,pad=0.4", fc="w", alpha=0.3),
    arrowprops=dict(
        arrowstyle="-|>, head_width=0.6, head_length=0.6",
        color="k",
        lw=2,
        connectionstyle="arc3,rad=-0.5",
    ),
)
vis.format_yaxis(ax, ytickfmt="{x:.0%}")
ax.legend(fancybox=True, shadow=True)
vis.savefig('summer_hospitalizations')

# %%
df_curr = df[df.index >= pd.to_datetime("11/22/2020")]
df_curr_normed = df_curr.divide(df_curr.max(), axis=1)
fig, ax = vis.subplots(figsize=(7, 6))
for col, color in col_colors.items():
    vis.plot_timeseries(
        df_curr_normed[col],
        label=col.lower(),
        color=color,
        ylabel='Hospitilizations (% of Peak)',
        ax=ax,
        # title="Winter/Spring, Impact of Vaccines",
    )
vis.format_yaxis(ax, ytickfmt="{x:.0%}")

df_vax = df_curr_normed[df_curr_normed.index >= pd.to_datetime("2/1/2021")]
ax.fill_between(
    df_vax.index,
    df_vax[plot_cols[0]],
    df_vax[plot_cols[1]],
    color="limegreen",
    alpha=0.4,
)

vlines = {
    "Vaccinations Start": pd.to_datetime("12/14/2020"),
    "10% of 65+ Vaccinated": pd.to_datetime("1/14/2021"),
    "50% of 65+ Vaccinated": pd.to_datetime("2/26/2021"),
}
for label, date in vlines.items():
    ax.axvline(date, ls="--", color="k", lw=1)
    height = 0.5 if label.startswith("50") else 0.17
    ax.annotate(
        label,
        xy=(date + timedelta(1), height),
        ha="left",
        va="bottom",
        fontweight='bold',
        fontsize=10,
        rotation=90,
    )

ax.annotate(
    "Vaccine\nImpact",
    xy=(pd.to_datetime("3/15/2021"), 0.35),
    ha="center",
    va="center",
    fontsize=12,
    fontweight="bold",
    color="darkgreen",
)

ax.legend(fancybox=True, shadow=True)
vis.savefig('current_hospitalizations')


# %%
col_colors = {"18-49 YR": "magenta", "65+ YR": "navy"}
plot_cols = list(col_colors.keys())

# %%
fig, axes = vis.subplots(1 , 2, figsize=(14, 6), sharey=False)

df_sa = df[
    (df.index >= pd.to_datetime("6/15/2020"))
    * (df.index <= pd.to_datetime("10/1/2020"))
]
df_sa_normed = df_sa.divide(df_sa.max(), axis=1)
df_july = df_sa_normed[df_sa_normed.index >= "7/13/2020"]

col_colors = {"18-49 YR": "magenta", "65+ YR": "navy"}
for col, color in col_colors.items():
    vis.plot_timeseries(
        df_sa_normed[col],
        label=col.lower(),
        ylabel='Hospitilizations (% of Peak)',
        color=color,
        ax=axes[0],
        # title="Summer/Autumn, before Vaccines",
    )
axes[0].fill_between(
    df_july.index,
    df_july[plot_cols[0]],
    df_july[plot_cols[1]],
    color="grey",
    alpha=0.3,
)
axes[0].annotate(
    "Older\ndecline\nslower",
    xy=(pd.to_datetime("8/15/2020"), 0.65),
    xytext=(pd.to_datetime("9/20/2020"), 0.8),
    fontweight="bold",
    ha="center",
    va="bottom",
    bbox=dict(boxstyle="round,pad=0.4", fc="w", alpha=0.3),
    arrowprops=dict(
        arrowstyle="-|>, head_width=0.6, head_length=0.6",
        color="k",
        lw=2,
        connectionstyle="arc3,rad=-0.5",
    ),
)
vis.format_yaxis(axes[0], ytickfmt="{x:.0%}")

df_curr = df[df.index >= pd.to_datetime("11/22/2020")]
df_curr_normed = df_curr.divide(df_curr.max(), axis=1)
for col, color in col_colors.items():
    vis.plot_timeseries(
        df_curr_normed[col],
        label=col.lower(),
        color=color,
        ax=axes[1],
        ylabel='Hospitilizations (% of Peak)',
        # title="Winter/Spring, Impact of Vaccines",
    )
vis.format_yaxis(axes[1], ytickfmt="{x:.0%}")

df_vax = df_curr_normed[df_curr_normed.index >= pd.to_datetime("2/1/2021")]
axes[1].fill_between(
    df_vax.index,
    df_vax[plot_cols[0]],
    df_vax[plot_cols[1]],
    color="limegreen",
    alpha=0.4,
)

vlines = {
    "Vaccinations Start": pd.to_datetime("12/14/2020"),
    "10% of 65+ Vaccinated": pd.to_datetime("1/14/2021"),
    "50% of 65+ Vaccinated": pd.to_datetime("2/26/2021"),
}
for label, date in vlines.items():
    axes[1].axvline(date, ls="--", color="k", lw=1)
    height = 0.5 if label.startswith("50") else 0.17
    axes[1].annotate(
        label,
        xy=(date + timedelta(1), height),
        ha="left",
        va="bottom",
        fontweight='bold',
        fontsize=10,
        rotation=90,
    )

axes[1].annotate(
    "Vaccine\nImpact",
    xy=(pd.to_datetime("3/15/2021"), 0.35),
    ha="center",
    va="center",
    fontsize=12,
    fontweight="bold",
    color="darkgreen",
)
axes[1].get_legend().remove()
axes[0].legend(fancybox=True, shadow=True, loc='lower left')
vis.savefig('vaccine_impact_split_axis')
