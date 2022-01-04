from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from lgimapy import vis
from lgimapy.data import Database
from lgimapy.latex import Document

vis.style()

# %%

date = pd.to_datetime("7/19/2021")
date_fmt = date.strftime("%m/%d/%Y")
# %%
db = Database()
ratings = ["BB", "B", "CCC"]
securities = ratings + ["JNK"]
bbg_indexes = [f"US_{rating}" for rating in ratings]
data = defaultdict(dict)
for rating in ratings:
    for field in ["OAS", "TRET", "XSRET"]:
        s = db.load_bbg_data(f"US_{rating}", field, start="1/1/2000")
        if field == "TRET":
            data[rating][field] = s.pct_change().iloc[1:]
        elif field == "XSRET":
            data[rating][field] = s.iloc[1:] / 100
        else:
            data[rating][field] = s.iloc[1:]

data["JNK"]["TRET"] = (
    db.load_bbg_data("JNK", "LEVEL", start="1/1/2000").pct_change().iloc[1:]
)


# %%
security = "BB"
field = "XSRET"
subset_deciles = 20


def make_violin_plots(field, doc, subset_deciles=None):
    if field == "TRET":
        name = "Total"
        securities = ["BB", "B", "CCC", "JNK"]
        col_fmt = "lrrrr"
    else:
        name = "Excess"
        securities = ["BB", "B", "CCC"]
        col_fmt = "lrrr"

    df = pd.DataFrame()
    for security in securities:
        s = data[security][field]
        if subset_deciles is not None:
            sec = "US_HY" if security == "JNK" else f"US_{security}"
            oas = db.load_bbg_data(sec, "OAS", start="1/1/2000").rank(pct=True)
            decile_dates = set(oas[oas < subset_deciles / 100].index)
            decile_dates |= set([date])
            s = s[s.index.isin(decile_dates)]
        df[security] = s.copy()

    summary = df.describe().iloc[1:]
    date_pct = df.rank(pct=True).loc[date].to_frame().T
    date_pct.index = [f"{date_fmt} %tile"]
    summary = summary.append(date_pct)
    doc.add_table(
        summary,
        caption=f"Daily {name} Return Moves",
        prec={col: "2%" for col in summary.columns},
        col_fmt=col_fmt,
    )

    fig, ax = vis.subplots(figsize=(10, 8))
    sns.violinplot(
        data=df,
        ax=ax,
        inner="quartile",
        linewidth=0.8,
        color="skyblue",
        alpha=0.6,
    )
    sns.swarmplot(data=df.loc[date].to_frame().T, color="firebrick", size=8)
    legend_elements = [
        Line2D(
            [0],
            [0],
            color="w",
            markerfacecolor="firebrick",
            marker="o",
            label=date_fmt,
            markersize=8,
        )
    ]
    ax.legend(
        handles=legend_elements, fancybox=True, shadow=True, loc="upper left"
    )
    if subset_deciles is not None and field == "XSRET":
        vis.format_yaxis(ax, "{x:.1%}")
    else:
        vis.format_yaxis(ax, "{x:.0%}")
    ax.set_ylabel(f"Daily {name} Return")
    fig_fid = (
        f"daily_{field}_ret_deciles"
        if subset_deciles is not None
        else f"daily_{field}_ret"
    )
    doc.add_figure(fig_fid, savefig=True)
    doc.add_pagebreak()


# %%


def make_future_returns_table(field, doc, subset_deciles=None):
    if field == "TRET":
        name = "Total Returns"
        securities = ["BB", "B", "CCC", "JNK"]
        col_fmt = "lrrrr"
        prec = {sec: "2%" for sec in securities}
    elif field == "XSRET":
        name = "Excess Returns"
        securities = ["BB", "B", "CCC"]
        col_fmt = "lrrr"
        prec = {sec: "2%" for sec in securities}
    else:
        name = "Spread Change"
        securities = ["BB", "B", "CCC"]
        col_fmt = "lrrr"
        prec = {sec: "0f" for sec in securities}

    lookforwards = ["1W", "1M", "3M", "6M"]
    d = defaultdict(list)
    for security in securities:
        s = data[security]["TRET"]
        dates = s[s <= s.loc[date]].index
        if subset_deciles is not None:
            sec = "US_HY" if security == "JNK" else f"US_{security}"
            oas = db.load_bbg_data(sec, "OAS", start="1/1/2000").rank(pct=True)
            decile_dates = set(oas[oas < subset_deciles / 100].index)
            dates = set(dates) & decile_dates
        s = data[security][field]
        for lookforward in lookforwards:
            returns = []
            for start_date in dates:
                try:
                    next_date = db.date(f"+{lookforward}", start_date)
                except IndexError:
                    continue
                if next_date > date:
                    continue
                if field == "OAS":
                    try:
                        returns.append(s.loc[next_date] - s.loc[start_date])
                    except KeyError:
                        continue
                    continue
                if security == "JNK":
                    s_date = s[(s.index >= start_date) & (s.index <= next_date)]
                    returns.append(np.prod(s_date + 1) - 1)
                    continue
                else:
                    returns.append(
                        db.load_bbg_data(
                            f"US_{security}",
                            field,
                            start=start_date,
                            end=next_date,
                            aggregate=True,
                        )
                    )
            d[security].append(np.median(returns))
    summary_df = pd.DataFrame(d, index=lookforwards)
    doc.add_table(
        summary_df,
        caption=f"Median {name} after Similar Size Moves",
        prec=prec,
        col_fmt=col_fmt,
    )


# %%

fid = "Impact_of_Large_Daily_Moves_in_HY"
doc = Document(fid, path="reports/HY", fig_dir=True)
doc.add_preamble(table_caption_justification="c")

doc.add_section("Full Data")
for field in ["TRET", "XSRET"]:
    make_violin_plots(field, doc)
for field in ["TRET", "XSRET", "OAS"]:
    make_future_returns_table(field, doc)
doc.add_pagebreak()

doc.add_section("2 Tightest Deciles")
for field in ["TRET", "XSRET"]:
    make_violin_plots(field, doc, subset_deciles=20)
for field in ["TRET", "XSRET", "OAS"]:
    make_future_returns_table(field, doc, subset_deciles=20)

doc.save(save_tex=True)
