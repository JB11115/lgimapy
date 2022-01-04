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
        date.strftime("%#m/%d/%Y"): date
        for date in [db.date("1y"), db.date("today")]
    }
    ratings = {"BB": ("BB+", "BB-"), "B": ("B+", "B-"), "CCC": ("CCC+", "CCC-")}
    d = defaultdict(dict)
    for fmt_date, date in dates.items():
        # Load current index and bonds that were formerly in the index
        # with less than 1 year maturity.
        db.load_market_data(date=date)
        ix = db.build_market_index(in_H0A0_index=True)
        for rating, rating_kws in ratings.items():
            ix_rating = ix.subset(rating=rating_kws)
            d[rating][fmt_date] = get_maturity_wall_df(ix_rating)
    return d


def plot_maturity_walls(mat_walls, doc):
    fig, axes = vis.subplots(3, 2, figsize=(10, 12))
    colors = dict(zip(mat_walls.keys(), vis.colors("ryb")))
    for i, (rating, date_d) in enumerate(mat_walls.items()):
        cols = ["MV", "MV_%"]
        d = defaultdict(list)
        for date, df in date_d.items():
            for col in cols:
                d[col].append(df[col].rename(date))

        for j, col in enumerate(cols):
            ax = axes[i, j]
            df = pd.concat(d[col], axis=1)
            df.plot.bar(color=["grey", colors[rating]], alpha=0.8, rot=0, ax=ax)
            ax.grid(False, axis="x")

            if col.endswith("%"):
                title = f"\n\n{rating} Maturity Walls (%)"
                vis.format_yaxis(ax, "{x:.0%}")
                offset = 100
            else:
                title = f"\n\n{rating} Maturity Walls ($)"
                vis.format_yaxis(ax, "${x:.0f}B")
                offset = 1
            ax.set_title(title, fontweight="bold")
            ax.legend(fancybox=True, shadow=True)
            for p in ax.patches:
                ax.annotate(
                    f"{offset * p.get_height():.0f}",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    va="bottom",
                    ha="center",
                    fontsize=9,
                )
    doc.add_figure("maturity_walls", savefig=True)


def update_maturity_walls(fid):
    vis.style()
    maturity_wall_data = get_maturity_walls()
    doc = Document(fid, path="reports/HY", fig_dir=True)
    doc.add_preamble(
        bookmarks=True,
        margin={"left": 0.5, "right": 0.5, "top": 1.5, "bottom": 1},
        footer=doc.footer(logo="LG_umbrella", height=-0.5, width=0.09),
    )
    doc.add_section("Maturity Walls")
    plot_maturity_walls(maturity_wall_data, doc)
    doc.save()


# %%

if __name__ == "__main__":
    fid = "Maturity_Walls"
    update_maturity_walls(fid)
