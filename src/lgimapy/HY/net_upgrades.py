from collections import defaultdict

import numpy as np
import pandas as pd

from lgimapy.vis import vis
from lgimapy.data import Database
from lgimapy.latex import Document

vis.style()

# %%

db = Database()
db.display_all_columns()
doc = Document("HY_Downgrades", path="latex/HY/2020/", fig_dir=True)


# %%
def ytd_junk_downgrades():
    db = Database()
    agencies = ["SP", "Moody", "Fitch"]
    d = defaultdict(list)
    for month in range(1, 13):

        try:
            month_start = db.date("MONTH_START", f"{month}/15/2020")
        except IndexError:
            month_start = db.date("MONTH_START")
        try:
            month_end = db.date("month_end", month_start)
        except IndexError:
            month_end = db.date("today")

        db.load_market_data(month_start)
        ix = db.build_market_index(in_hy_returns_index=True)
        ix_mv = ix.total_value("AmountOutstanding").squeeze()
        isins = ix.isins
        ratings_df = db.rating_changes(month_start, month_end)
        ratings_df.head()
        agency = "Moody"
        d["Month"].append(month_start.strftime("%b"))
        for agency in agencies:
            downgrades = ratings_df[
                ratings_df["ISIN"].isin(isins)
                & (ratings_df[f"{agency}Rating_CHANGE"] < 0)
            ]["ISIN"]
            downgrade_ix = ix.subset(isin=downgrades)
            downgrade_mv = downgrade_ix.total_value(
                "AmountOutstanding"
            ).squeeze()
            downgrades = downgrade_mv / ix_mv
            if isinstance(downgrades, pd.Series):
                downgrades = 0
            d[agency].append(downgrades)

    return pd.DataFrame(d).set_index("Month", drop=True).rename_axis(None)


ytd_junk_df = ytd_junk_downgrades()
ytd_junk_df.to_csv("HY_downgrades_2020.csv")
# %%


colors = ["#752333", "#0438A3", "#8E5C1D"]
fig, ax = vis.subplots(figsize=(12, 6))
vis.format_yaxis(ax, "{x:.0%}")
ytd_junk_df.plot.bar(rot=0, color=colors, ax=ax, alpha=0.9)
ax.grid(False, axis="x")
ax.set_title("Downgrades (BBG LF98 US HY Index)", fontweight="bold")
doc.add_figure("hy_downgrades_2020", savefig=True)

# %%
def ytd_fallen_angels():
    db = Database()
    agencies = ["SP", "Moody", "Fitch"]
    months, fallen_angel_pct = [], []
    for month in range(1, 13):
        try:
            month_start = db.date("MONTH_START", f"{month}/15/2020")
        except IndexError:
            month_start = db.date("MONTH_START")
        try:
            month_end = db.date("month_end", month_start)
        except IndexError:
            month_end = db.date("today")

        months.append(month_start.strftime("%b"))

        db.load_market_data(month_start, local=True)
        ix = db.build_market_index(in_returns_index=True)
        ix_mv = ix.total_value("AmountOutstanding").squeeze()
        isins = ix.isins
        ratings_df = db.rating_changes(month_start, month_end)
        fallen_angels = ratings_df[
            ratings_df["ISIN"].isin(isins)
            & (
                ratings_df[f"NumericRating_PREV"]
                <= db.convert_letter_ratings("BBB-")
            )
            & (
                ratings_df[f"NumericRating_NEW"]
                >= db.convert_letter_ratings("BB+")
            )
        ]["ISIN"]
        fa_ix = ix.subset(isin=fallen_angels)
        fa_mv = fa_ix.total_value("AmountOutstanding").squeeze()
        fa_mv = 0 if isinstance(fa_mv, pd.Series) else fa_mv
        fallen_angel_pct.append(fa_mv / ix_mv)

    return pd.Series(fallen_angel_pct, index=months)


ytd_fa_df = ytd_fallen_angels()
ytd_fa_df.to_csv("Fallen_Angels_2020.csv")
# %%
fig, ax = vis.subplots(figsize=(12, 6))
vis.format_yaxis(ax, "{x:.1%}")
ytd_fa_df.plot.bar(rot=0, color="steelblue", ax=ax, alpha=0.9)
ax.grid(False, axis="x")
ax.set_title("Fallen Angels (BBG LUCR US IG Index)", fontweight="bold")
doc.add_figure("fallen_angel_downgrades_2020", savefig=True)

# %%

starting_ratings = ("BB+", "BB-")
start = "1/1/2020"
end = None

# %%
def get_net_fallen_angels(start=None, end=None, nonfin=False):
    db = Database()
    db.load_market_data(db.nearest_date(start), local=True)
    ix = db.build_market_index(
        rating=("BBB+", "BBB-"),
        in_stats_index=True,
    )
    # Find index value of rating category on start date.
    mv = ix.total_value("AmountOutstanding").squeeze()

    ratings_df = db.rating_changes(start, end)
    is_index_eligible = (ratings_df["USCreditReturnsFlag"] == 1) | (
        ratings_df["USHYReturnsFlag"] == 1
    )
    is_nonfin = ratings_df["FinancialFlag"] == 0
    is_fallen_angel = (
        ratings_df[f"NumericRating_PREV"] <= db.convert_letter_ratings("BBB-")
    ) & (ratings_df[f"NumericRating_NEW"] >= db.convert_letter_ratings("BB+"))
    is_rising_star = (
        ratings_df[f"NumericRating_PREV"] >= db.convert_letter_ratings("BB+")
    ) & (ratings_df[f"NumericRating_NEW"] <= db.convert_letter_ratings("BBB-"))
    df = ratings_df[
        is_index_eligible & is_nonfin & (is_fallen_angel | is_rising_star)
    ].copy()
    df["net_change"] = df["AmountOutstanding"] * np.sign(
        df["NumericRating_CHANGE"]
    )
    net_upgrades = (
        df[["Date_NEW", "net_change"]]
        .groupby("Date_NEW")
        .sum()
        .rename_axis(None)
        .squeeze()
        .resample("w")
        .sum()
        .dropna()
    )
    return np.cumsum(net_upgrades) / mv


def get_cumulative_net_upgrades(starting_ratings, start=None, end=None):
    db = Database()
    db.load_market_data(db.nearest_date(start), local=True)
    ix = db.build_market_index(
        rating=starting_ratings,
        in_stats_index=True,
        in_hy_stats_index=True,
        special_rules="USCreditStatisticsFlag | USHYStatisticsFlag",
    )
    # Find index value of rating category on start date.
    mv = ix.total_value("AmountOutstanding").squeeze()

    # Subset rating changes to specified ratings and index eligible bonds.
    ratings_df = db.rating_changes(start, end)
    numeric_ratings = db.convert_letter_ratings(starting_ratings)
    is_nonfin = ratings_df["FinancialFlag"] == 0
    is_index_eligible = (ratings_df["USCreditReturnsFlag"] == 1) | (
        ratings_df["USHYReturnsFlag"] == 1
    )
    rating_change_df_list = []
    for agency in ["SP", "Moody"]:
        in_specied_ratings = (
            ratings_df[f"{agency}Rating_PREV"] >= numeric_ratings[0]
        ) & (ratings_df[f"{agency}Rating_PREV"] <= numeric_ratings[1])
        df = ratings_df[
            is_index_eligible & in_specied_ratings & is_nonfin
        ].copy()
        chg_col = f"{agency}Rating_CHANGE"
        # Weight individual rating changes by 50% * number of notches
        # to balance the two agencies impact.
        df["weighted_par"] = 0.5 * df["AmountOutstanding"] * df[chg_col]
        rating_change_df_list.append(df[["Date_NEW", "weighted_par"]])

    net_upgrades = (
        pd.concat(rating_change_df_list)
        .groupby("Date_NEW")
        .sum()
        .rename_axis(None)
        .squeeze()
    )
    return np.cumsum(net_upgrades) / mv


# %%
events = {
    "GFC": "12/04/2008",
    "Energy Crisis": "2/11/2016",
    "Covid-19": "3/23/2020",
}
ratings = {
    "Fallen Angels/Rising Stars": None,
    "BB": ("BB+", "BB-"),
    "B": ("B+", "B-"),
    "CCC": ("CCC+", "CCC-"),
}
d = defaultdict(list)
for event, date in events.items():
    d["wides"].append(db.nearest_date(date))
    d["start"].append(db.date("6m", date))
    try:
        end = db.date("+2.5y", date)
    except IndexError:
        end = db.date("today")
    d["end"].append(end)

event_df = pd.DataFrame(d, index=events.keys())
event_df.to_csv("crisis_dates.csv")
# %%
cum_upgrades = defaultdict(dict)
for crisis, event in event_df.iterrows():
    for rating, rating_kws in ratings.items():
        if rating_kws is None:
            net_upgrades = get_net_fallen_angels(event["start"], event["end"])
        else:
            net_upgrades = get_cumulative_net_upgrades(
                rating_kws, event["start"], event["end"]
            )
        net_upgrades.index = [
            (d - event["wides"]).days / 365 for d in net_upgrades.index
        ]
        cum_upgrades[crisis][rating] = net_upgrades / 1e3


# %%
df_list = []
for crisis, rating_d in cum_upgrades.items():
    for rating, df in rating_d.items():
        name = {"Fallen Angels/Rising Stars": "FA_RS"}.get(rating, rating)
        df_list.append(df.rename(f"{crisis}_{name}"))

full_df = (
    pd.concat(df_list, axis=1).fillna(method="ffill").fillna(method="bfill")
)
full_df.to_csv("crises_net_upgrades_by_rating.csv")
# %%
color = {
    "GFC": "navy",
    "Energy Crisis": "darkgreen",
    "Covid-19": "darkorchid",
}
ls = {"Fallen Angels/Rising Stars": "-", "BB": "--", "B": "-.", "CCC": ":"}
fig, axes = vis.subplots(3, 1, sharex=True, figsize=(10, 10))
for ax, (crisis, rating_d) in zip(axes.flat, cum_upgrades.items()):
    # vis.format_yaxis(ax, ytickfmt="{x:.0%}")
    vis.format_yaxis(ax, ytickfmt="${x:.0f}B")
    ax.set_title(crisis, fontweight="bold")
    ax.axhline(0, c="darkgray", lw=3)
    for rating, s in rating_d.items():
        ax.plot(
            s, color=color[crisis], ls=ls[rating], lw=2, alpha=0.7, label=rating
        )
axes[-1].set_xlabel("Years From Crisis Wides")
axes[0].legend(
    loc="upper center",
    fontsize=12,
    ncol=4,
    fancybox=True,
    shadow=True,
    bbox_to_anchor=(0.5, 1.4),
)
ax = fig.add_subplot(111, frame_on=False)
ax.grid(False)
ax.tick_params(labelcolor="none", bottom=False, left=False)
ax.set_ylabel("Cumulative Net Upgrades\n\n")
doc.add_figure("net_upgrades_by_crisis", savefig=True)

# %%
fig, axes = vis.subplots(4, 1, sharex=True, figsize=(10, 10))
for crisis, rating_d in cum_upgrades.items():
    for ax, (rating, s) in zip(axes.flat, rating_d.items()):
        # vis.format_yaxis(ax, ytickfmt="{x:.0%}")
        vis.format_yaxis(ax, ytickfmt="${x:.0f}B")
        ax.set_title(rating, fontweight="bold")
        ax.axhline(0, c="darkgray", lw=2)
        ax.plot(
            s, color=color[crisis], ls=ls[rating], lw=2, alpha=0.7, label=crisis
        )
axes[-1].set_xlabel("Years From Crisis Wides")
axes[0].legend(
    loc="upper center",
    fontsize=12,
    ncol=3,
    fancybox=True,
    shadow=True,
    bbox_to_anchor=(0.5, 1.7),
)
ax = fig.add_subplot(111, frame_on=False)
ax.grid(False)
ax.tick_params(labelcolor="none", bottom=False, left=False)
ax.set_ylabel("Cumulative Net Upgrades\n\n")
doc.add_figure("net_upgrades_by_rating", savefig=True)


# %%
# doc.save_tex()


# %%
ratings = {
    "A-Rated": ("AAA", "A-"),
    "BBB": ("BBB+", "BBB-"),
    "Fallen Angels/Rising Stars": None,
}
cum_upgrades = defaultdict(dict)
for crisis, event in event_df.iterrows():
    for rating, rating_kws in ratings.items():
        if rating_kws is None:
            net_upgrades = get_net_fallen_angels(event["start"], event["end"])
        else:
            net_upgrades = get_cumulative_net_upgrades(
                rating_kws, event["start"], event["end"]
            )
        net_upgrades.index = [
            (d - event["wides"]).days / 365 for d in net_upgrades.index
        ]
        cum_upgrades[crisis][rating] = net_upgrades

# %%
ls = {"Fallen Angels/Rising Stars": ":", "A-Rated": "-", "BBB": "--"}
fig, axes = vis.subplots(3, 1, sharex=True, figsize=(10, 10))
for ax, (crisis, rating_d) in zip(axes.flat, cum_upgrades.items()):
    vis.format_yaxis(ax, ytickfmt="{x:.0%}")
    # vis.format_yaxis(ax, ytickfmt="${x:.0f}B")
    ax.set_title(crisis, fontweight="bold")
    ax.axhline(0, c="darkgray", lw=3)
    for rating, s in rating_d.items():
        ax.plot(
            s, color=color[crisis], ls=ls[rating], lw=2, alpha=0.7, label=rating
        )
axes[-1].set_xlabel("Years From Crisis Wides")
axes[0].legend(
    loc="upper center",
    fontsize=12,
    ncol=3,
    fancybox=True,
    shadow=True,
    bbox_to_anchor=(0.5, 1.4),
)
ax = fig.add_subplot(111, frame_on=False)
ax.grid(False)
ax.tick_params(labelcolor="none", bottom=False, left=False)
ax.set_ylabel("Cumulative Net Upgrades\n\n")
vis.savefig("IG_pct_nonfin_net_upgrades_by_crisis")
# vis.show()

# %%
colors = ["#752333", "#0438A3", "#8E5C1D"]

color = {
    "GFC": "k",
    "Energy Crisis": "darkgreen",
    "Covid-19": "darkorchid",
}
fig, axes = vis.subplots(3, 1, sharex=True, figsize=(10, 10))
for crisis, rating_d in cum_upgrades.items():
    for ax, (rating, s) in zip(axes.flat, rating_d.items()):
        vis.format_yaxis(ax, ytickfmt="{x:.0%}")
        # vis.format_yaxis(ax, ytickfmt="${x:.0f}B")
        ax.set_title(rating, fontweight="bold")
        ax.axhline(0, c="darkgray", lw=2)
        ax.plot(
            s,
            color=color[crisis],
            lw=2,
            alpha=0.7,
            label=crisis
            # s, color=color[crisis], ls=ls[rating], lw=2, alpha=0.7, label=crisis
        )
axes[-1].set_xlabel("Years From Crisis Wides")
axes[0].legend(
    loc="upper center",
    fontsize=12,
    ncol=3,
    fancybox=True,
    shadow=True,
    bbox_to_anchor=(0.5, 1.4),
)
ax = fig.add_subplot(111, frame_on=False)
ax.grid(False)
ax.tick_params(labelcolor="none", bottom=False, left=False)
ax.set_ylabel("Cumulative Net Upgrades\n\n")
vis.savefig("IG_pct_nonfin_net_upgrades_by_rating")
# vis.show()
