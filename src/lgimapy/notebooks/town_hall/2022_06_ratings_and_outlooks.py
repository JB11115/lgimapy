import numpy as np
import pandas as pd

from lgimapy import vis
from lgimapy.data import Database
from lgimapy.models import rating_migrations

vis.style()
# %%
excel_df = pd.DataFrame()


class RatingChange:
    def __init__(self, db):
        self._db = db

    def rating_change_history_df(self, freq=None, start=None, end=None):
        self._freq = freq
        df = self._db.rating_changes(start=start, end=end)
        self._df = df[df["USCreditReturnsFlag"] == True]
        df = pd.DataFrame(index=self._db.trade_dates(start=start, end=end))
        for action in ["upgrade", "downgrade"]:
            df[action] = self._get_n_actions_history(action)

        df.fillna(0, inplace=True)
        df["net_upgrade"] = df["upgrade"] - df["downgrade"]
        return df

    def _get_n_actions_history(self, action):
        if action == "upgrade":
            df = self._df[
                (self._df["MoodyRating_CHANGE"] > 0)
                | (self._df["SPRating_CHANGE"] > 0)
                | (self._df["FitchRating_CHANGE"] > 0)
            ]
        elif action == "downgrade":
            df = self._df[
                (self._df["MoodyRating_CHANGE"] < 0)
                | (self._df["SPRating_CHANGE"] < 0)
                | (self._df["FitchRating_CHANGE"] < 0)
            ]
        df = (
            df[["Date_PREV", "Issuer"]]
            .groupby(["Date_PREV", "Issuer"], observed=True)
            .count()
            .reset_index()
        )
        return df.groupby("Date_PREV").count().rename_axis(None).squeeze()


fig, axes = vis.subplots(2, 1, sharex=True, figsize=(12, 8))

db = Database()
self = RatingChange(db)

start = pd.to_datetime("1/1/2020")
df = self.rating_change_history_df(start=db.date("3m", start))
n_action_df = df.rolling(window=44).sum().dropna()
s = n_action_df["net_upgrade"]
s = s[s.index >= start]
excel_df["n_rating_change_net_upgrade_bias"] = s
vis.plot_timeseries(
    s,
    color="grey",
    lw=1.5,
    end_point=False,
    title="2 Month Rolling Net Upgrades in US IG",
    ylabel="# Issuers",
    ax=axes[0],
)
n = 50
alpha = 0.01
for i in np.linspace(0, 1, n):
    s_pos = s.copy()
    s_pos[s < 0] = 0
    s_neg = s.copy()
    s_neg[s > 0] = 0
    axes[0].fill_between(
        s_pos.index, 0 + i * s_pos, s_pos, color="steelblue", alpha=alpha
    )
    axes[0].fill_between(
        s_neg.index, 0 + i * s_neg, s_neg, color="firebrick", alpha=alpha
    )

axes[0].fill_between(s_pos.index, 0, s_pos, color="steelblue", alpha=0.15)
axes[0].fill_between(s_neg.index, 0, s_neg, color="firebrick", alpha=0.15)
axes[0].axhline(0, color="grey", lw=1)


full_d = db.load_json("citi_rating_outlook")
data_d_list = full_d["datasets"]["data-7c61d340b6d9bc16d05f5c6ace340703"]
d = {}
for data_d in data_d_list:
    date = pd.to_datetime(data_d["reportdate"])
    val = data_d["Net"]
    d[date] = val

s = pd.Series(d).sort_index()
s = s[s.index >= start]
excel_df["n_rating_outlook_net_upgrade_bias"] = s
vis.plot_timeseries(
    s,
    color="grey",
    lw=1.5,
    end_point=False,
    title="Net Upgrade Bias in Rating Agency Outlooks",
    ylabel="# Issuers",
    ax=axes[1],
)
n = 50
alpha = 0.01
for i in np.linspace(0, 1, n):
    s_pos = s.copy()
    s_pos[s < 0] = 0
    s_neg = s.copy()
    s_neg[s > 0] = 0
    axes[1].fill_between(
        s_pos.index, 0 + i * s_pos, s_pos, color="steelblue", alpha=alpha
    )
    axes[1].fill_between(
        s_neg.index, 0 + i * s_neg, s_neg, color="firebrick", alpha=alpha
    )

axes[1].fill_between(s_pos.index, 0, s_pos, color="steelblue", alpha=0.15)
axes[1].fill_between(s_neg.index, 0, s_neg, color="firebrick", alpha=0.15)
axes[1].axhline(0, color="grey", lw=1)
yticks = [-400, -300, -200, -100, 0, 100]
axes[1].set_yticks(yticks)
axes[1].set_yticklabels(yticks)
axes[1].set_ylim(-420, 105)
vis.savefig("IG_rating_changes")


excel_df.dropna().to_csv("IG_rating_changes.csv")
