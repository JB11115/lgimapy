from collections import defaultdict
import pandas as pd

from lgimapy import vis
from lgimapy.data import Database
from lgimapy.latex import Document


# %%


def update_rating_performance(fid):
    vis.style()
    doc = Document(fid, path="reports/HY", fig_dir=True)
    doc.add_preamble(
        orientation="landscape",
        bookmarks=True,
        margin={
            "left": 0.5,
            "right": 0.5,
            "top": 1.5,
            "bottom": 1,
            "paperwidth": 6,
            "paperheight": 12,
        },
        footer=doc.footer(logo="LG_umbrella", height=-0.5, width=0.08),
    )
    figs = {"TRET": "Total Returns", "XSRET": "Excess Returns"}
    plot_fids = []
    doc.add_bookmark("Performance by Rating")
    for field, title in figs.items():
        df = get_rating_performance(field)
        plot_fid = plot_rating_performance(df, title, doc)
        plot_fids.append(plot_fid)
    doc.add_subfigures(figures=plot_fids)
    doc.save()


def get_rating_performance(field):
    rating_buckets = "A BBB BB B CCC".split()

    db = Database()
    ytd = db.date("YTD")
    last_month_end = db.date("LAST_MONTH_END")
    # last_month_end = pd.to_datetime("5/31/2022")
    last_month_start = db.date("MONTH_START", last_month_end)
    last_month = last_month_end.strftime("%B")
    d = defaultdict(list)
    for rating in rating_buckets:
        ytd_ret = db.load_bbg_data(
            f"US_{rating}", field, start=ytd, aggregate=True
        )
        last_month_ret = db.load_bbg_data(
            f"US_{rating}",
            field,
            start=last_month_start,
            end=last_month_end,
            aggregate=True,
        )
        d[last_month].append(last_month_ret)
        d["YTD"].append(ytd_ret)

    return pd.DataFrame(d, index=rating_buckets).round(3)


def plot_rating_performance(df, title, doc):
    fig, ax = vis.subplots(figsize=(12, 6))
    df.plot.bar(color=["darkorchid", "navy"], alpha=0.7, rot=0, ax=ax)
    ax.grid(False, axis="x")
    ax.legend(fancybox=True, shadow=True)
    vis.format_yaxis(ax, "{x:.0%}")
    ax.set_title(title, fontweight="bold")

    n_patches = len(ax.patches)
    for i, p in enumerate(ax.patches):
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        y_label_pos = y if y >= 0 else y - 0.0005
        ax.annotate(
            f"{100 * y:.1f}",
            (x, y_label_pos),
            va="bottom" if y >= 0 else "top",
            ha="center",
            fontweight="bold",
            fontsize=14,
            color="darkorchid" if i < n_patches / 2 else "navy",
        )
    fid = f"{title.replace(' ', '_')}_performance"
    vis.savefig(fid, path=doc.fig_dir, dpi=200)
    vis.close()
    return fid


# %%
if __name__ == "__main__":
    fid = "Rating_Performance"
    update_rating_performance(fid)
