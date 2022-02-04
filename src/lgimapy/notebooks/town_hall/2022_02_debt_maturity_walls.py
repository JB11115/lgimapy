from collections import defaultdict

import numpy as np
import pandas as pd

from lgimapy import vis
from lgimapy.data import Database
from lgimapy.latex import Document
from lgimapy.utils import root


# %%


def get_maturity_wall_df(ix):
    db = Database()
    date = ix.df["Date"].iloc[0]
    total_mv = ix.total_value().iloc[0]
    period_labels = ["1-2y", "2-3y", "3-4y", "4-5y", "5-10y", "10y+"]
    period_mv = []
    for period in period_labels:
        if period.startswith("<"):
            # No starting date.
            end = db.date(f"+{period.strip('<')}", date, fcast=True)
            df = ix.df[ix.df["MaturityDate"] <= end].copy()
        elif period.endswith("+"):
            # No end date.
            start = db.date(f"+{period.strip('+')}", date, fcast=True)
            df = ix.df[ix.df["MaturityDate"] > start].copy()
        else:
            start, end = [
                db.date(f"+{year}y", date, fcast=True)
                for year in period.strip("y").split("-")
            ]
            df = ix.df[
                (ix.df["MaturityDate"] > start) & (ix.df["MaturityDate"] <= end)
            ]
        period_mv.append(df["AmountOutstanding"].sum())
    df = pd.Series(period_mv, index=period_labels, name="MV").to_frame()
    df["MV_%"] = df["MV"] / total_mv
    df["MV"] /= 1e3
    return df


def get_maturity_walls():
    db = Database()
    dates = {
        date.strftime("%-m/%d/%Y"): date
        for date in [db.date("5y"), db.date("today")]
    }
    d = {}
    for fmt_date, date in dates.items():
        # Load current index and bonds that were formerly in the index
        # with less than 1 year maturity.
        db.load_market_data(date=date)
        ix = db.build_market_index(in_H0A0_index=True, rating=("B+", "CCC-"))
        d[fmt_date] = get_maturity_wall_df(ix)
    return d


def plot_maturity_walls(maturity_wall_data):
    fig, ax = vis.subplots(figsize=(8, 6))
    df = pd.concat(
        (df["MV"].rename(date) for date, df, in maturity_wall_data.items()),
        axis=1,
    )
    df.plot.bar(color=["grey", "navy"], alpha=0.8, rot=0, ax=ax)
    ax.grid(False, axis="x")

    title = f"Maturity Walls ($)"
    vis.format_yaxis(ax, "${x:.0f}B")
    offset = 1
    ax.legend(
        fancybox=True,
        shadow=True,
        loc="center",
        bbox_to_anchor=(0.5, 1.07),
        ncol=2,
    )
    for p in ax.patches:
        ax.annotate(
            f"{offset * p.get_height():.0f}",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            va="bottom",
            ha="center",
            fontsize=9,
        )


vis.style()
# maturity_wall_data = get_maturity_walls()
plot_maturity_walls(maturity_wall_data)
vis.savefig("maturity_walls")
