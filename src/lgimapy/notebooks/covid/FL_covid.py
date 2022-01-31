import numpy as np
import pandas as pd

import lgimapy.vis as vis
from lgimapy.data import Database
from lgimapy.utils import root

vis.style()

# %%
db = Database()

fid = root("data/covid/FL_covid_data.csv")
full_df = pd.read_csv(fid)
full_df["date"] = pd.to_datetime(
    pd.to_datetime(full_df["EventDate"]).apply(lambda x: x.date())
)
full_df.set_index("date", inplace=True)
full_df.sort_index(inplace=True)
full_df["Date"] = full_df.index
# %%

list(full_df)
# %%
median_age = full_df["Age"].resample("2w").median()
vis.plot_timeseries(median_age, xtickfmt="auto", ylabel="Median Age of Cases")
vis.show()

# %%
def get_death_rates(df):
    per_case, per_hosp = {}, {}
    for age, age_df in df.groupby("Age_group"):
        n_died = len(age_df[age_df["Died"] == "Yes"])
        n_hosp = len(age_df[age_df["Hospitalized"] == "YES"])
        n_cases = len(age_df)
        per_case[age] = 0 if n_cases == 0 else n_died / n_cases
        per_hosp[age] = 0 if n_hosp == 0 else n_died / n_hosp

    ages = [
        "0-4 years",
        "5-14 years",
        "15-24 years",
        "25-34 years",
        "35-44 years",
        "45-54 years",
        "55-64 years",
        "65-74 years",
        "75-84 years",
        "85+ years",
    ]
    return pd.Series(per_case).reindex(ages), pd.Series(per_hosp).reindex(ages)


df_march = full_df[(full_df.index >= "3/1/2020") & (full_df.index < "4/1/2020")]
march_case, march_hosp = get_death_rates(df_march)

df_rec = full_df[full_df.index >= db.date("2w")]
rec_case, rec_hosp = get_death_rates(df_rec)

# %%
fig, ax = vis.subplots(figsize=(12, 8))
w = 0.3
ax.bar(
    range(len(march_case)),
    march_case.values,
    width=w,
    color="navy",
    label="March",
)
ax.bar(
    np.arange(len(rec_case)) + w,
    rec_case.values,
    width=w,
    color="darkorchid",
    label="Past 2 Weeks",
)
ax.legend()
ax.set_xticks(np.arange(len(march_case)))
ax.set_xticklabels(rec_case.index, rotation=-45, ha="left")
ax.set_ylabel("Fatalities per Confirmed Case")
ax.grid(False, axis="x")
vis.format_yaxis(ax, ytickfmt="{x:.0%}")
# vis.savefig("deaths_per_case")
vis.show()

# %%

# %%
fig, ax = vis.subplots(figsize=(12, 8))
w = 0.3
ax.bar(
    range(len(march_hosp)),
    march_hosp.values,
    width=w,
    color="navy",
    label="March",
)
ax.bar(
    np.arange(len(rec_hosp)) + w,
    rec_hosp.values,
    width=w,
    color="darkorchid",
    label="Past 2 Weeks",
)
ax.legend()
ax.set_xticks(np.arange(len(march_hosp)))
ax.set_xticklabels(rec_hosp.index, rotation=-45, ha="left")
ax.set_ylabel("Fatalities per Hospitilization")
ax.grid(False, axis="x")
vis.format_yaxis(ax, ytickfmt="{x:.0%}")
vis.savefig("FL_fatalities_per_hosp")
# vis.show()
