"""
Export covid case data from
https://data.cdc.gov/Case-Surveillance/United-States-COVID-19-Cases-and-Deaths-by-State-o/9mfq-cb36
"""
import os

from collections import defaultdict
import pandas as pd

from lgimapy import vis
from lgimapy.data import Database
from lgimapy.utils import nearest_date, root

vis.style()
# %%
db = Database()
url = (
    "https://raw.githubusercontent.com/owid/covid-19-data/master/public"
    "/data/vaccinations/us_state_vaccinations.csv"
)
proxy = "http://JB11115:$Jrb1236463716@proxych:8080"

os.environ["http_proxy"] = proxy
os.environ["https_proxy"] = proxy
vax_df = pd.read_csv(url).set_index("date").rename_axis(None)
vax_df.index = pd.to_datetime(vax_df.index)

state_vax_dfs = {state: df for state, df in vax_df.groupby("location")}


fid = "United_States_COVID-19_Cases_and_Deaths_by_State_over_Time.csv"
df = pd.read_csv(root(f"data/covid/{fid}"))
df["date"] = pd.to_datetime(df["submission_date"])
state_case_dfs = {state: df for state, df in df.groupby("state")}


# %%
d = defaultdict(list)
for state, df in state_vax_dfs.items():
    state_code = db.convert_state_codes(state)
    if pd.isna(state_code):
        continue
    vaccinated = df["people_vaccinated_per_hundred"].dropna()
    if not len(vaccinated):
        continue
    prev_date = nearest_date(db.date("1w"), vaccinated.index)

    d["cases_per_day"].append(
        state_case_dfs[state_code]
        .set_index("date")
        .sort_values("date")["new_case"]
        .dropna()
        .iloc[-7:]
        .mean()
    )
    d["state"].append(state_code)
    d["vax"].append(vaccinated.iloc[-1] - vaccinated.loc[prev_date])
    d["population"].append(db.state_populations[state_code])


df = pd.DataFrame(d)
df["cases_per_capita"] = df["cases_per_day"] / df["population"]
df["cases_per_100k"] = df["cases_per_capita"] * 1e5


# %%
fig, ax = vis.subplots(figsize=(10, 8))
ax.plot(df["cases_per_100k"], df["vax"] / 100, "o", alpha=0.5)
for _, row in df.iterrows():
    ax.annotate(
        row["state"], (row["cases_per_100k"], row["vax"] / 100), fontsize=12
    )
ax.set_xlabel("Current Cases per 100k")
ax.set_ylabel("Population Vaccinated in Past Week")
vis.format_yaxis(ax, ytickfmt="{x:.1%}")
vis.savefig("US_state_recent_vaccinations")
