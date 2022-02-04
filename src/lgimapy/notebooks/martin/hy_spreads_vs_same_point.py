import numpy as np
import pandas as pd

import lgimapy.vis as vis
from lgimapy.data import Database
from lgimapy.utils import EventDistance
from lgimapy.latex import Document

db = Database()
cols = ["HY", "BB", "B"]
colors = ["#04060F", "#0294A5", "#C1403D"]
cmap = {c: color for c, color in zip(cols, colors)}

df = db.load_bbg_data([f"US_{col}" for col in cols], "OAS")
df.columns = cols
doc = Document(
    "HY_historic_spreads", path="latex/HY/202004_historic_spreads", fig_dir=True
)

# %%
# Plot historic spreads.
fig, ax = vis.subplots(figsize=(10, 6))
vis.plot_multiple_timeseries(
    df, ylabel="OAS", c_list=cmap.values(), ax=ax, xtickfmt="auto",
)
for col in cols:
    ax.axhline(df[col][-1], color=cmap[col], ls="--", lw=1.5, alpha=0.7)
    ax.plot(df.index[-1], df[col][-1], "o", color=cmap[col], ms=5)
doc.add_figure("HY_spreads", savefig=True)

# %%

for cat in cols:
    s = df[cat]
    s_curr_level = s - s[-1]
    zero_crossings = np.where(np.diff(np.sign(s_curr_level)))[0]
    curr_level_dates = list(s.index[zero_crossings])
    ed = EventDistance(s, curr_level_dates, lookback="0d", lookforward="6m")

    # Plot how spreads moved over 6 months after reaching current levels.
    fig, ax = vis.subplots(figsize=(10, 5))
    df_b = pd.DataFrame(ed.values).T.fillna(method="ffill")
    for col in df_b.columns:
        ax.plot(df_b[col], color="gray", alpha=0.3, lw=2, label="_nolegend_")
    avg = df_b.mean(axis=1)
    med = df_b.median(axis=1)
    ax.plot(avg, c="k", label=f"Mean: {avg.iloc[-1]:.0f} bp", lw=3)
    ax.plot(med, c="steelblue", label=f"Median: {med.iloc[-1]:.0f} bp", lw=3)
    ax.legend()
    ax.set_xlabel("Weeks Since Spreads Reached Current Level")
    ax.set_title(f"{cat}, Current OAS: {s.iloc[-1]:.0f} bp")
    doc.add_figure(f"{cat}_spread_timeseries", savefig=True)

    # Plot distribution of spreads after 6 months.
    fig, ax = vis.subplots(figsize=(12, 6))
    vis.plot_hist(df_b.iloc[-1, :], ax=ax)
    ax.set_xlabel("OAS")
    ax.set_title("Distribution of Spreads 6 months after current levels")
    doc.add_figure(f"{cat}_spread_hist", savefig=True)


# %%
df_ret = np.log(df[1:] / df[:-1].values)
periods = {"1w": "1 Week", "1m": "1 Month", "3m": "3 Months", "6m": "6 Months"}
mean_df = pd.DataFrame(index=cols, columns=periods.values())
med_df = pd.DataFrame(index=cols, columns=periods.values())

for cat in cols:
    s = df_ret[cat]
    dates = list(s.sort_values().index[:50])
    for lb, period in periods.items():
        ed = EventDistance(df[cat], dates, lookback="0d", lookforward=lb)
        temp = pd.DataFrame(ed.values).T
        changes = []
        for col in temp.columns:
            if np.sum(temp[col].isna()) / len(temp) > 0.4:
                continue
            vals = temp[col].dropna().values
            changes.append(vals[-1] - vals[0])
        changes
        mean_df.loc[cat, period] = np.mean(changes)
        med_df.loc[cat, period] = np.median(changes)


doc.add_table(mean_df, prec=2, caption="Mean change in spreads")
doc.add_table(med_df, prec=2, caption="Median change in spreads")
# doc.save(save_tex=True)
