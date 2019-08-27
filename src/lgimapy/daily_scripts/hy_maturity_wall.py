from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

from lgimapy.data import Database
from lgimapy.utils import custom_sort, savefig

plt.style.use("fivethirtyeight")
# %%


def main():
    # %%
    db = Database()
    db.load_market_data(dev=True)

    # Get total HY market value.
    total_ix = db.build_market_index(rating="HY")
    total_mv = market_value(total_ix.df)

    # Store rating and maturity classifiers.
    ratings = ["BB", "B", "CCC"]
    ratings_d = {k: (f"{k}+", f"{k}-") for k in ratings}
    mats_d = {
        "<1": (0, 1),
        "1-2": (1, 2),
        "2-3": (2, 3),
        "3-4": (3, 4),
        "4-5": (4, 5),
        "5-10": (5, 10),
        ">10": (10, np.infty),
    }
    mats = custom_sort(mats_d.keys(), "<123456789>")

    pcts = defaultdict(list)
    for r in ratings:
        for mat in mats:
            ix = db.build_market_index(
                rating=ratings_d[r], maturity=mats_d[mat]
            )
            pcts[r].append(market_value(ix.df) / total_mv)

    # %%
    N = len(mats)
    ind = np.arange(N)
    width = 0.27

    fig, ax = plt.subplots(1, 1, figsize=[14, 6])

    bplots = {}
    colors = ["steelblue", "forestgreen", "firebrick"]
    for i, (r, c) in enumerate(zip(ratings, colors)):
        bplots[r] = ax.bar(ind + width * i, pcts[r], width, color=c, alpha=0.6)

    ax.set_ylabel("Market Value")
    ax.set_xlabel("Maturity (yrs)")
    ax.set_xticks(ind + width)
    ax.set_xticklabels(mats)
    ax.legend((bplots[r][0] for r in ratings), ratings)
    ax.xaxis.grid(False)
    tick = mtick.StrMethodFormatter("{x:.0%}")
    ax.yaxis.set_major_formatter(tick)

    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                1.05 * h,
                f"{h:.1%}\n${h*total_mv/1e3:.0f} B",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    for r in ratings:
        autolabel(bplots[r])

    savefig("maturity_wall_6-10-2019")
    plt.show()

    # %%

    ind = np.arange(len(mats))
    width = 0.27

    fig, ax = plt.subplots(1, 1, figsize=[14, 6])

    bplots = {}
    colors = ["steelblue", "forestgreen", "firebrick"]
    for i, (r, c) in enumerate(zip(ratings, colors)):
        bplots[r] = ax.bar(ind + width * i, pcts[r], width, color=c, alpha=0.6)

    ax.set_ylabel("Market Value")
    ax.set_xlabel("Maturity (yrs)")
    ax.set_xticks(ind + width)
    ax.set_xticklabels(mats)
    ax.legend((bplots[r][0] for r in ratings), ratings)
    ax.xaxis.grid(False)
    tick = mtick.StrMethodFormatter("{x:.0%}")
    ax.yaxis.set_major_formatter(tick)

    def autolabel(rects, fontsize=8):
        for rect in rects:
            h = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                1.05 * h,
                f"{h:.1%}\n${h*total_mv/1e3:.0f} B",
                ha="center",
                va="bottom",
                fontsize=fontsize,
            )

    for r in ratings:
        autolabel(bplots[r])

    savefig("maturity_wall_6-10-2019")
    plt.show()

    # %%
    fig, ax = plt.subplots(1, 1, figsize=[6, 6])

    y_pos = np.arange(len(ratings))
    heights = [sum(pcts[r]) for r in ratings]
    ax.set_ylabel("Market Value")
    ax.set_xlabel("Rating")
    ax.set_xticks(y_pos)
    ax.set_xticklabels(ratings)
    ax.xaxis.grid(False)
    ax.yaxis.set_major_formatter(tick)
    rect = ax.bar(y_pos, heights, color=colors, alpha=0.6)
    autolabel(rect, fontsize=14)

    savefig("rating_pct_6-10-2019")
    plt.show()
    # %%


def market_value(df):
    """Return total market value of DataFrame."""
    return np.sum(df["DirtyPrice"] * df["AmountOutstanding"]) / 100
